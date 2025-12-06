"""Query Denoising Generator for Co-DETR.

This module implements Collaborative Denoising (CDN) for DETR-based detectors.
Query denoising adds noisy ground truth queries during training to stabilize
convergence and improve final performance.

Key Features:
    - Label noise: Randomly flip class labels
    - Box noise: Add scaled perturbation to bounding box coordinates
    - Group-based denoising: Multiple noise groups with attention masking
    - Attention masks: Prevent information leakage between groups
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .box_ops import bbox_xyxy_to_cxcywh, inverse_sigmoid


class DnQueryGenerator:
    """Denoising Query Generator for DETR training.
    
    Generates noisy ground truth queries to train the decoder for denoising.
    This helps stabilize training and improves detection performance.
    
    Args:
        num_queries: Number of learnable object queries.
        hidden_dim: Hidden dimension of query embeddings.
        num_classes: Number of object classes.
        noise_scale: Dictionary with 'label' and 'box' noise scales.
            - 'label': Probability of flipping a label (default: 0.5).
            - 'box': Scale factor for box perturbation (default: 0.4).
        group_cfg: Dictionary for group configuration.
            - 'dynamic': If True, compute groups dynamically based on GT count.
            - 'num_groups': Fixed number of groups (if not dynamic).
            - 'num_dn_queries': Target number of denoising queries (if dynamic).
    
    Example:
        >>> dn_generator = DnQueryGenerator(
        ...     num_queries=300,
        ...     hidden_dim=256,
        ...     num_classes=80,
        ...     noise_scale={'label': 0.5, 'box': 0.4},
        ...     group_cfg={'dynamic': True, 'num_dn_queries': 100}
        ... )
        >>> label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
        ...     gt_bboxes, gt_labels, label_enc, img_metas
        ... )
    """
    
    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        num_classes: int,
        noise_scale: Optional[Dict[str, float]] = None,
        group_cfg: Optional[Dict] = None,
    ) -> None:
        """Initialize the denoising query generator."""
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Default noise scales
        if noise_scale is None:
            noise_scale = {'label': 0.5, 'box': 0.4}
        self.label_noise_scale = noise_scale.get('label', 0.5)
        self.box_noise_scale = noise_scale.get('box', 0.4)
        
        # Default group configuration
        if group_cfg is None:
            group_cfg = {'dynamic': True, 'num_dn_queries': 100}
        
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, (
                "num_dn_queries should be set when using dynamic dn groups"
            )
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, (
                "num_groups should be set when using static dn groups"
            )
            self.num_dn = group_cfg['num_groups']
        
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, (
            f"Expected num_dn to be a positive integer, got {self.num_dn}"
        )
    
    def get_num_groups(self, group_queries: Optional[int] = None) -> int:
        """Compute the number of denoising groups.
        
        Args:
            group_queries: Number of queries per group (for dynamic mode).
        
        Returns:
            Number of denoising groups.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, (
                "group_queries should be provided when using dynamic dn groups"
            )
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        
        if num_groups < 1:
            num_groups = 1
        
        return int(num_groups)
    
    def __call__(
        self,
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        label_enc: nn.Module,
        img_metas: List[Dict],
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """Generate denoising queries from ground truth.
        
        Args:
            gt_bboxes: List of ground truth bboxes per image.
                Each tensor has shape (num_gts, 4) in xyxy format.
            gt_labels: List of ground truth labels per image.
                Each tensor has shape (num_gts,).
            label_enc: Label embedding module (nn.Embedding).
            img_metas: List of image metadata dictionaries.
                Each dict should contain 'img_shape' (H, W, C).
        
        Returns:
            Tuple of:
                - input_query_label: Label embeddings (B, pad_size, hidden_dim).
                - input_query_bbox: Box embeddings (B, pad_size, 4).
                - attn_mask: Attention mask (tgt_size, tgt_size).
                - dn_meta: Metadata dict with 'pad_size' and 'num_dn_group'.
        """
        assert gt_labels is not None, "gt_labels is required"
        assert label_enc is not None, "label_enc is required"
        assert img_metas is not None, "img_metas is required"
        assert len(gt_bboxes) == len(gt_labels), (
            f"Length mismatch: gt_bboxes={len(gt_bboxes)}, gt_labels={len(gt_labels)}"
        )
        
        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device if len(gt_bboxes) > 0 and len(gt_bboxes[0]) > 0 else torch.device('cpu')
        
        # Handle empty ground truth case
        if all(len(boxes) == 0 for boxes in gt_bboxes):
            return self._get_empty_output(batch_size, device)
        
        # Normalize bboxes to [0, 1] in cxcywh format
        gt_bboxes_normalized = []
        for img_meta, bboxes in zip(img_metas, gt_bboxes):
            if len(bboxes) == 0:
                gt_bboxes_normalized.append(bboxes.new_zeros(0, 4))
                continue
            
            img_h, img_w = img_meta['img_shape'][:2]
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes) / factor
            gt_bboxes_normalized.append(bboxes_normalized)
        
        # Count known objects per image
        known_num = [len(labels) for labels in gt_labels]
        
        # Compute number of groups
        num_groups = self.get_num_groups(int(max(known_num)) if known_num else 0)
        
        # Concatenate all ground truth
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes_normalized)
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) 
            for i, t in enumerate(gt_labels)
        ])
        
        # Create indices for positive and negative groups
        num_total_gt = len(labels)
        known_indice = torch.arange(num_total_gt, device=device)
        
        # Repeat for 2 * num_groups (positive + negative for each group)
        known_indice = known_indice.repeat(2 * num_groups)
        known_labels = labels.repeat(2 * num_groups)
        known_bid = batch_idx.repeat(2 * num_groups)
        known_bboxs = boxes.repeat(2 * num_groups, 1)
        
        # Clone for modification
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        
        # Apply label noise
        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)
            ).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expand.scatter_(0, chosen_indice, new_label)
        
        # Compute padding size
        single_pad = int(max(known_num)) if known_num else 0
        pad_size = int(single_pad * 2 * num_groups)
        
        # Create positive and negative indices for box noise
        positive_idx = torch.arange(num_total_gt, device=device).unsqueeze(0).repeat(num_groups, 1)
        positive_idx += (torch.arange(num_groups, device=device) * num_total_gt * 2).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_total_gt
        
        # Apply box noise
        if self.box_noise_scale > 0 and num_total_gt > 0:
            # Convert cxcywh to xyxy for noise application
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            
            # Compute diff (half of width/height)
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            
            # Generate random noise
            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32
            )
            rand_sign = rand_sign * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            
            # Add extra noise for negative samples
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            
            # Apply noise
            known_bbox_ += torch.mul(rand_part, diff) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            
            # Convert back to cxcywh
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
        
        # Create embeddings
        m = known_labels_expand.long()
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand.clamp(1e-3, 1 - 1e-3))
        
        # Create padding tensors
        padding_label = torch.zeros(pad_size, self.hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)
        
        # Create output tensors
        input_query_label = padding_label.unsqueeze(0).repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Map known indices to padded positions
        if sum(known_num) > 0:
            map_known_indice = torch.cat([
                torch.arange(num, device=device) for num in known_num
            ])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(2 * num_groups)
            ]).long()
            
            # Fill in the embeddings
            input_query_label[known_bid.long(), map_known_indice] = input_label_embed
            input_query_bbox[known_bid.long(), map_known_indice] = input_bbox_embed
        
        # Create attention mask
        tgt_size = pad_size + self.num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, device=device).bool()
        
        # Match queries cannot see denoising queries
        attn_mask[pad_size:, :pad_size] = True
        
        # Denoising groups cannot see each other
        for i in range(num_groups):
            group_start = single_pad * 2 * i
            group_end = single_pad * 2 * (i + 1)
            
            # Mask everything before current group
            if i > 0:
                attn_mask[group_start:group_end, :group_start] = True
            
            # Mask everything after current group (within dn region)
            if i < num_groups - 1:
                attn_mask[group_start:group_end, group_end:pad_size] = True
        
        # Create metadata
        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': num_groups,
        }
        
        return input_query_label, input_query_bbox, attn_mask, dn_meta
    
    def _get_empty_output(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """Generate empty output when no ground truth is available.
        
        Args:
            batch_size: Batch size.
            device: Output device.
        
        Returns:
            Tuple of empty tensors and metadata.
        """
        input_query_label = torch.zeros(
            batch_size, 0, self.hidden_dim, device=device
        )
        input_query_bbox = torch.zeros(
            batch_size, 0, 4, device=device
        )
        attn_mask = torch.zeros(
            self.num_queries, self.num_queries, device=device
        ).bool()
        dn_meta = {
            'pad_size': 0,
            'num_dn_group': 0,
        }
        return input_query_label, input_query_bbox, attn_mask, dn_meta


class CdnQueryGenerator(DnQueryGenerator):
    """Collaborative Denoising Query Generator.
    
    This is a specialized version of DnQueryGenerator for CDN (Collaborative
    Denoising) used in Co-DETR and related models.
    
    Inherits all functionality from DnQueryGenerator.
    
    Args:
        Same as DnQueryGenerator.
    
    Example:
        >>> cdn_generator = CdnQueryGenerator(
        ...     num_queries=300,
        ...     hidden_dim=256,
        ...     num_classes=80,
        ... )
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize CDN query generator."""
        super().__init__(*args, **kwargs)


def build_dn_generator(dn_args: Optional[Dict]) -> Optional[DnQueryGenerator]:
    """Build a denoising generator from configuration.
    
    Factory function to create DnQueryGenerator or CdnQueryGenerator
    based on the 'type' key in the configuration dictionary.
    
    Args:
        dn_args: Configuration dictionary with keys:
            - 'type': Either 'DnQueryGenerator' or 'CdnQueryGenerator'.
            - Other keys are passed to the constructor.
    
    Returns:
        Instantiated generator, or None if dn_args is None.
    
    Raises:
        NotImplementedError: If the specified type is not supported.
    
    Example:
        >>> config = {
        ...     'type': 'CdnQueryGenerator',
        ...     'num_queries': 300,
        ...     'hidden_dim': 256,
        ...     'num_classes': 80,
        ... }
        >>> generator = build_dn_generator(config)
    """
    if dn_args is None:
        return None
    
    # Make a copy to avoid modifying the original
    dn_args = dn_args.copy()
    dn_type = dn_args.pop('type')
    
    if dn_type == 'DnQueryGenerator':
        return DnQueryGenerator(**dn_args)
    elif dn_type == 'CdnQueryGenerator':
        return CdnQueryGenerator(**dn_args)
    else:
        raise NotImplementedError(f"Denoising type '{dn_type}' is not supported")
