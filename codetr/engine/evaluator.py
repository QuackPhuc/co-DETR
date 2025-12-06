"""Detection evaluation metrics without pycocotools.

This module provides custom mAP (mean Average Precision) evaluation for
object detection, supporting COCO-style metrics without external dependencies.

Key features:
    - AP at multiple IoU thresholds [0.5:0.95:0.05]
    - Per-class AP computation
    - Precision-recall curve calculation
    - 101-point AP interpolation

Reference:
    COCO API: https://cocodataset.org/#detection-eval

Example:
    >>> evaluator = DetectionEvaluator(num_classes=80)
    >>> for images, targets in val_loader:
    ...     predictions = model(images)
    ...     evaluator.add_predictions(predictions, targets)
    >>> metrics = evaluator.compute_metrics()
    >>> print(f"mAP: {metrics['mAP']:.4f}")
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def compute_iou_matrix(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """Compute IoU matrix between predictions and ground truth boxes.
    
    Args:
        pred_boxes: Predicted boxes in xyxy format. Shape: (N, 4).
        gt_boxes: Ground truth boxes in xyxy format. Shape: (M, 4).
        
    Returns:
        IoU matrix. Shape: (N, M).
        
    Example:
        >>> pred = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> gt = torch.tensor([[0, 0, 10, 10]])
        >>> iou = compute_iou_matrix(pred, gt)
        >>> print(iou.shape)  # (2, 1)
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), 
                          device=pred_boxes.device)
    
    # Compute areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1])
    
    # Compute intersection
    lt = torch.max(pred_boxes[:, None, :2], gt_boxes[:, :2])  # (N, M, 2)
    rb = torch.min(pred_boxes[:, None, 2:], gt_boxes[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute union
    union = pred_area[:, None] + gt_area - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-8)
    
    return iou


def compute_ap(recalls: Tensor, precisions: Tensor) -> float:
    """Compute Average Precision using 101-point interpolation.
    
    Uses the COCO-style 101-point interpolation where AP is computed as
    the average of maximum precisions at 101 recall thresholds.
    
    Args:
        recalls: Recall values sorted in ascending order. Shape: (N,).
        precisions: Precision values corresponding to recalls. Shape: (N,).
        
    Returns:
        Average Precision value in [0, 1].
        
    Example:
        >>> recalls = torch.tensor([0.1, 0.2, 0.3, 0.4])
        >>> precisions = torch.tensor([1.0, 0.8, 0.6, 0.5])
        >>> ap = compute_ap(recalls, precisions)
    """
    if len(recalls) == 0:
        return 0.0
    
    # Add sentinel values at both ends
    recalls = torch.cat([torch.zeros(1, device=recalls.device), 
                        recalls,
                        torch.ones(1, device=recalls.device)])
    precisions = torch.cat([torch.zeros(1, device=precisions.device), 
                           precisions,
                           torch.zeros(1, device=precisions.device)])
    
    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    
    # 101-point interpolation
    recall_thresholds = torch.linspace(0, 1, 101, device=recalls.device)
    
    # Find precision at each recall threshold
    ap_sum = 0.0
    for r_thresh in recall_thresholds:
        # Find first recall >= threshold
        mask = recalls >= r_thresh
        if mask.any():
            # Get maximum precision at this recall level
            idx = mask.nonzero(as_tuple=True)[0][0]
            ap_sum += precisions[idx].item()
    
    return ap_sum / 101.0


def match_predictions_to_gt(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_threshold: float,
) -> Tuple[Tensor, Tensor]:
    """Match predictions to ground truth using greedy matching.
    
    Predictions are sorted by score (high to low) and matched to the
    highest IoU ground truth box that hasn't been matched yet.
    
    Args:
        pred_boxes: Predicted boxes in xyxy format. Shape: (N, 4).
        pred_scores: Prediction scores. Shape: (N,).
        pred_labels: Predicted class labels. Shape: (N,).
        gt_boxes: Ground truth boxes in xyxy format. Shape: (M, 4).
        gt_labels: Ground truth class labels. Shape: (M,).
        iou_threshold: Minimum IoU for a valid match.
        
    Returns:
        Tuple of:
            - tp: True positive mask. Shape: (N,). True if matched.
            - matched_gt: Matched GT index for each prediction. Shape: (N,).
              -1 if not matched.
    """
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]
    
    device = pred_boxes.device
    
    # Initialize outputs
    tp = torch.zeros(num_preds, dtype=torch.bool, device=device)
    matched_gt = torch.full((num_preds,), -1, dtype=torch.long, device=device)
    
    if num_gts == 0:
        return tp, matched_gt
    
    if num_preds == 0:
        return tp, matched_gt
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # (N, M)
    
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    # Track which GT boxes have been matched
    gt_matched = torch.zeros(num_gts, dtype=torch.bool, device=device)
    
    for pred_idx in sorted_indices:
        pred_label = pred_labels[pred_idx]
        
        # Find GT boxes with same class
        class_mask = gt_labels == pred_label
        
        if not class_mask.any():
            continue
        
        # Get IoUs with same-class GT boxes
        ious = iou_matrix[pred_idx]  # (M,)
        
        # Mask out already matched and different class
        available_mask = class_mask & ~gt_matched
        
        if not available_mask.any():
            continue
        
        # Find best matching GT (highest IoU among available)
        masked_ious = ious.clone()
        masked_ious[~available_mask] = -1
        
        best_gt_idx = masked_ious.argmax()
        best_iou = masked_ious[best_gt_idx]
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = True
            matched_gt[pred_idx] = best_gt_idx
            gt_matched[best_gt_idx] = True
    
    return tp, matched_gt


class DetectionEvaluator:
    """Evaluator for object detection with mAP metrics.
    
    Computes COCO-style Average Precision at multiple IoU thresholds.
    
    Args:
        num_classes: Number of object classes.
        iou_thresholds: List of IoU thresholds for matching.
            Default: [0.5, 0.55, ..., 0.95] (10 thresholds).
            
    Attributes:
        predictions: List of accumulated predictions.
        ground_truths: List of accumulated ground truths.
        
    Example:
        >>> evaluator = DetectionEvaluator(num_classes=80)
        >>> # Add predictions from validation loop
        >>> for batch_idx, (images, targets) in enumerate(val_loader):
        ...     with torch.no_grad():
        ...         predictions = model(images)
        ...     evaluator.add_predictions(predictions, targets)
        >>> # Compute final metrics
        >>> metrics = evaluator.compute_metrics()
        >>> print(f"mAP: {metrics['mAP']:.4f}")
        >>> print(f"AP50: {metrics['AP50']:.4f}")
    """
    
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[List[float]] = None,
    ) -> None:
        self.num_classes = num_classes
        
        if iou_thresholds is None:
            # COCO-style thresholds: [0.5, 0.55, 0.6, ..., 0.95]
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds
        
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions and ground truths."""
        self.predictions: List[Dict[str, Tensor]] = []
        self.ground_truths: List[Dict[str, Tensor]] = []
        self._image_counter = 0
    
    def add_predictions(
        self,
        predictions: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> None:
        """Add batch predictions and targets for evaluation.
        
        Args:
            predictions: List of prediction dicts, one per image, with keys:
                - 'boxes': Tensor of shape (N, 4) in xyxy format
                - 'scores': Tensor of shape (N,)
                - 'labels': Tensor of shape (N,)
            targets: List of target dicts, one per image, with keys:
                - 'boxes': Tensor of shape (M, 4) in cxcywh or xyxy format
                - 'labels': Tensor of shape (M,)
                
        Note:
            Target boxes should be in absolute xyxy format for proper IoU
            computation. If they are in normalized cxcywh format, you should
            convert them before calling this method.
        """
        for pred, target in zip(predictions, targets):
            # Store predictions with image ID
            pred_with_id = {
                'image_id': self._image_counter,
                'boxes': pred['boxes'].cpu() if pred['boxes'].numel() > 0 else torch.zeros((0, 4)),
                'scores': pred['scores'].cpu() if pred['scores'].numel() > 0 else torch.zeros(0),
                'labels': pred['labels'].cpu() if pred['labels'].numel() > 0 else torch.zeros(0, dtype=torch.long),
            }
            self.predictions.append(pred_with_id)
            
            # Store ground truth with image ID
            gt_with_id = {
                'image_id': self._image_counter,
                'boxes': target['boxes'].cpu() if target['boxes'].numel() > 0 else torch.zeros((0, 4)),
                'labels': target['labels'].cpu() if target['labels'].numel() > 0 else torch.zeros(0, dtype=torch.long),
            }
            self.ground_truths.append(gt_with_id)
            
            self._image_counter += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute mAP and related metrics.
        
        Returns:
            Dictionary containing:
                - 'mAP': Mean AP across all IoU thresholds and classes
                - 'AP50': AP at IoU threshold 0.5
                - 'AP75': AP at IoU threshold 0.75
                - 'AP_per_class': List of per-class AP values
        """
        if len(self.predictions) == 0:
            return {
                'mAP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_per_class': [0.0] * self.num_classes,
            }
        
        # Compute AP for each class and IoU threshold
        all_aps = []  # Shape will be (num_thresholds, num_classes)
        
        ap_at_50 = []
        ap_at_75 = []
        
        for class_id in range(self.num_classes):
            class_aps_per_threshold = []
            
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_class_ap(class_id, iou_thresh)
                class_aps_per_threshold.append(ap)
            
            all_aps.append(class_aps_per_threshold)
            
            # Store AP50 and AP75 for this class
            ap_at_50.append(class_aps_per_threshold[0])  # IoU=0.5
            if len(class_aps_per_threshold) >= 6:
                ap_at_75.append(class_aps_per_threshold[5])  # IoU=0.75
            else:
                ap_at_75.append(0.0)
        
        # Compute mean AP across classes and thresholds
        # Only include classes that have GT samples
        classes_with_gt = set()
        for gt in self.ground_truths:
            for label in gt['labels'].tolist():
                classes_with_gt.add(label)
        
        if len(classes_with_gt) == 0:
            return {
                'mAP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_per_class': [0.0] * self.num_classes,
            }
        
        # Compute per-class AP (averaged over thresholds)
        ap_per_class = []
        for class_id in range(self.num_classes):
            if class_id in classes_with_gt:
                class_ap = sum(all_aps[class_id]) / len(all_aps[class_id])
            else:
                class_ap = 0.0
            ap_per_class.append(class_ap)
        
        # Compute mAP (average over classes with GT)
        valid_aps = [ap_per_class[c] for c in classes_with_gt]
        mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
        
        # Compute AP50 (average over classes with GT)
        valid_ap50 = [ap_at_50[c] for c in classes_with_gt]
        AP50 = sum(valid_ap50) / len(valid_ap50) if valid_ap50 else 0.0
        
        # Compute AP75 (average over classes with GT)
        valid_ap75 = [ap_at_75[c] for c in classes_with_gt]
        AP75 = sum(valid_ap75) / len(valid_ap75) if valid_ap75 else 0.0
        
        return {
            'mAP': mAP,
            'AP50': AP50,
            'AP75': AP75,
            'AP_per_class': ap_per_class,
        }
    
    def _compute_class_ap(self, class_id: int, iou_threshold: float) -> float:
        """Compute AP for a single class at a single IoU threshold.
        
        Args:
            class_id: Class index to evaluate.
            iou_threshold: IoU threshold for matching.
            
        Returns:
            Average Precision for this class and threshold.
        """
        # Collect all predictions and GT for this class
        all_pred_scores = []
        all_pred_boxes = []
        all_pred_image_ids = []
        
        for pred in self.predictions:
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                all_pred_scores.append(pred['scores'][class_mask])
                all_pred_boxes.append(pred['boxes'][class_mask])
                all_pred_image_ids.extend(
                    [pred['image_id']] * class_mask.sum().item()
                )
        
        if len(all_pred_scores) == 0:
            return 0.0
        
        all_pred_scores = torch.cat(all_pred_scores)
        all_pred_boxes = torch.cat(all_pred_boxes)
        all_pred_image_ids = torch.tensor(all_pred_image_ids)
        
        # Count total GT for this class
        total_gt = 0
        gt_per_image = {}
        for gt in self.ground_truths:
            class_mask = gt['labels'] == class_id
            num_gt = class_mask.sum().item()
            total_gt += num_gt
            if num_gt > 0:
                gt_per_image[gt['image_id']] = {
                    'boxes': gt['boxes'][class_mask],
                    'matched': torch.zeros(num_gt, dtype=torch.bool),
                }
        
        if total_gt == 0:
            return 0.0
        
        # Sort predictions by score (descending)
        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        
        # Compute TP/FP for each prediction
        tp = torch.zeros(len(sorted_indices))
        fp = torch.zeros(len(sorted_indices))
        
        for pred_rank, pred_idx in enumerate(sorted_indices):
            image_id = all_pred_image_ids[pred_idx].item()
            pred_box = all_pred_boxes[pred_idx:pred_idx + 1]
            
            if image_id not in gt_per_image:
                # No GT for this image, this is a false positive
                fp[pred_rank] = 1
                continue
            
            gt_data = gt_per_image[image_id]
            gt_boxes = gt_data['boxes']
            gt_matched = gt_data['matched']
            
            # Compute IoU with GT boxes
            ious = compute_iou_matrix(pred_box, gt_boxes).squeeze(0)  # (M,)
            
            # Find best unmatched GT
            available_mask = ~gt_matched
            masked_ious = ious.clone()
            masked_ious[~available_mask] = -1
            
            if masked_ious.numel() > 0:
                best_gt_idx = masked_ious.argmax()
                best_iou = masked_ious[best_gt_idx]
                
                if best_iou >= iou_threshold:
                    tp[pred_rank] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_rank] = 1
            else:
                fp[pred_rank] = 1
        
        # Compute cumulative TP and FP
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        
        # Compute precision and recall
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall = cum_tp / total_gt
        
        # Compute AP using 101-point interpolation
        ap = compute_ap(recall, precision)
        
        return ap


def build_evaluator(num_classes: int, **kwargs) -> DetectionEvaluator:
    """Build DetectionEvaluator with specified configuration.
    
    Args:
        num_classes: Number of object classes.
        **kwargs: Additional arguments for DetectionEvaluator.
        
    Returns:
        Configured DetectionEvaluator instance.
    """
    return DetectionEvaluator(num_classes=num_classes, **kwargs)
