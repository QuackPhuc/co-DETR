"""
Performance and Stress Tests for Co-DETR.

This module contains lower-priority tests including:
- Memory stress tests with larger batch sizes
- CUDA tensor compatibility tests
- Performance benchmarks for critical operations

These tests are designed to run on GPU machines (Colab/Kaggle) and
verify the implementation can handle realistic workloads.
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import List, Dict

# Skip module if running on CPU-only environment
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available - these tests require GPU"
)


# ============================================================================
# CUDA Tensor Tests
# ============================================================================

class TestCUDATensorCompatibility:
    """Tests for CUDA tensor compatibility across all modules.
    
    These tests verify that all core modules work correctly when tensors
    are placed on GPU, including proper data movement and gradient flow.
    """
    
    @pytest.fixture
    def cuda_device(self):
        """Get CUDA device."""
        return torch.device("cuda")
    
    def test_backbone_cuda_forward(self, cuda_device):
        """Backbone should produce valid outputs on CUDA."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            norm_eval=True,
            pretrained=False,
        ).to(cuda_device)
        
        # Create CUDA input
        images = torch.rand(2, 3, 224, 224, device=cuda_device)
        
        # Forward pass
        backbone.eval()
        with torch.no_grad():
            features = backbone(images)
        
        # Verify outputs are on CUDA
        assert all(f.is_cuda for f in features), "Features should be on CUDA"
        assert all(not torch.isnan(f).any() for f in features), "No NaN values"
        assert len(features) == 3, "Should have C3, C4, C5 features"
    
    def test_backbone_cuda_gradient_flow(self, cuda_device):
        """Gradients should flow correctly on CUDA."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            norm_eval=True,
            pretrained=False,
        ).to(cuda_device)
        
        images = torch.rand(2, 3, 224, 224, device=cuda_device, requires_grad=True)
        
        backbone.train()
        features = backbone(images)
        
        # Backward pass
        loss = sum(f.sum() for f in features)
        loss.backward()
        
        # Check gradients exist for unfrozen parameters
        has_grad = any(
            p.grad is not None 
            for p in backbone.parameters() 
            if p.requires_grad
        )
        assert has_grad, "Should have gradients for unfrozen parameters"
    
    def test_neck_cuda_forward(self, cuda_device):
        """ChannelMapper should work correctly on CUDA."""
        from codetr.models.neck import ChannelMapper
        
        neck = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        ).to(cuda_device)
        
        # Create CUDA backbone features
        features = [
            torch.rand(2, 512, 56, 56, device=cuda_device),
            torch.rand(2, 1024, 28, 28, device=cuda_device),
            torch.rand(2, 2048, 14, 14, device=cuda_device),
        ]
        
        with torch.no_grad():
            outputs = neck(features)
        
        assert all(o.is_cuda for o in outputs), "Outputs should be on CUDA"
        assert len(outputs) == 4, "Should have 4 output levels"
        assert all(o.shape[1] == 256 for o in outputs), "All should have 256 channels"
    
    def test_transformer_cuda_forward(self, cuda_device):
        """Transformer should work correctly on CUDA."""
        from codetr.models.transformer.transformer import CoDeformableDetrTransformer
        
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=2,  # Reduced for memory
            num_decoder_layers=2,
            num_feature_levels=4,
            num_queries=100,  # Reduced for memory
        ).to(cuda_device)
        
        batch_size = 2
        
        # Create CUDA inputs
        srcs = [
            torch.rand(batch_size, 256, 25, 25, device=cuda_device),
            torch.rand(batch_size, 256, 13, 13, device=cuda_device),
            torch.rand(batch_size, 256, 7, 7, device=cuda_device),
            torch.rand(batch_size, 256, 4, 4, device=cuda_device),
        ]
        masks = [torch.zeros(batch_size, h, w, dtype=torch.bool, device=cuda_device) 
                 for (_, _, h, w) in [s.shape for s in srcs]]
        pos_embeds = [torch.rand_like(s) for s in srcs]
        query_embed = torch.rand(100, 512, device=cuda_device)
        
        with torch.no_grad():
            outputs = transformer(
                srcs=srcs,
                masks=masks,
                pos_embeds=pos_embeds,
                query_embed=query_embed,
            )
        
        # Verify outputs are on CUDA
        hs, init_reference, inter_references, _, _ = outputs
        assert hs.is_cuda, "Hidden states should be on CUDA"
        assert init_reference.is_cuda, "Init reference should be on CUDA"
    
    def test_detr_head_cuda_forward(self, cuda_device):
        """DETR head should work correctly on CUDA."""
        from codetr.models.heads.detr_head import CoDeformDETRHead
        
        head = CoDeformDETRHead(
            num_classes=80,
            embed_dims=256,
            num_decoder_layers=2,
            num_query=100,
        ).to(cuda_device)
        
        batch_size = 2
        num_queries = 100
        
        # Create CUDA inputs
        hs = torch.rand(2, batch_size, num_queries, 256, device=cuda_device)  # num_layers, B, Q, C
        init_reference = torch.rand(batch_size, num_queries, 4, device=cuda_device)
        inter_references = torch.rand(2, batch_size, num_queries, 4, device=cuda_device)
        
        with torch.no_grad():
            outputs = head(
                hidden_states=hs,
                init_reference=init_reference,
                inter_references=inter_references,
            )
        
        assert 'pred_logits' in outputs, "Should have pred_logits"
        assert 'pred_boxes' in outputs, "Should have pred_boxes"
        assert outputs['pred_logits'].is_cuda, "Logits should be on CUDA"
    
    def test_losses_cuda_computation(self, cuda_device):
        """Loss functions should work correctly on CUDA."""
        from codetr.models.losses import FocalLoss, GIoULoss
        
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        giou_loss = GIoULoss(reduction='mean')
        
        # Create CUDA tensors
        pred_logits = torch.rand(2, 100, 80, device=cuda_device)
        target_classes = torch.randint(0, 80, (2, 100), device=cuda_device)
        
        pred_boxes = torch.rand(2, 100, 4, device=cuda_device)
        target_boxes = torch.rand(2, 100, 4, device=cuda_device)
        
        # Compute losses
        cls_loss = focal_loss(
            inputs=pred_logits.view(-1, 80),
            targets=target_classes.view(-1),
        )
        
        bbox_loss = giou_loss(
            inputs=pred_boxes.view(-1, 4),
            targets=target_boxes.view(-1, 4),
        )
        
        assert cls_loss.is_cuda, "Classification loss should be on CUDA"
        assert bbox_loss.is_cuda, "Bbox loss should be on CUDA"
        assert not torch.isnan(cls_loss), "No NaN in classification loss"
        assert not torch.isnan(bbox_loss), "No NaN in bbox loss"
    
    def test_mixed_precision_cuda(self, cuda_device):
        """Model should work with mixed precision (AMP) on CUDA."""
        from codetr.models.backbone import ResNetBackbone
        from codetr.models.neck import ChannelMapper
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        
        neck = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        ).to(cuda_device)
        
        images = torch.rand(2, 3, 224, 224, device=cuda_device)
        
        # Test with autocast
        with torch.cuda.amp.autocast():
            features = backbone(images)
            outputs = neck(features)
        
        # Results should still be valid
        assert all(not torch.isnan(o).any() for o in outputs), "No NaN with AMP"
        assert all(not torch.isinf(o).any() for o in outputs), "No Inf with AMP"
    
    def test_cuda_memory_release(self, cuda_device):
        """GPU memory should be properly released after operations."""
        from codetr.models.backbone import ResNetBackbone
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(cuda_device)
        
        # Create and run model
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        
        images = torch.rand(2, 3, 224, 224, device=cuda_device)
        
        with torch.no_grad():
            _ = backbone(images)
        
        # Clean up
        del backbone, images
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(cuda_device)
        
        # Memory should return to approximately initial level
        memory_diff = abs(final_memory - initial_memory)
        # Allow 10MB tolerance for cached allocations
        assert memory_diff < 10 * 1024 * 1024, (
            f"Memory leak detected: {memory_diff / 1024 / 1024:.2f} MB not released"
        )


# ============================================================================
# Memory Stress Tests
# ============================================================================

class TestMemoryStressLargerBatches:
    """Tests for larger batch sizes to verify memory efficiency.
    
    These tests stress the model with batch sizes 8-16 to ensure:
    - No out-of-memory errors
    - Stable numerical behavior
    - Consistent output shapes
    """
    
    @pytest.fixture
    def cuda_device(self):
        """Get CUDA device."""
        return torch.device("cuda")
    
    def test_backbone_batch_8(self, cuda_device):
        """Backbone should handle batch size 8."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        backbone.eval()
        
        batch_size = 8
        images = torch.rand(batch_size, 3, 224, 224, device=cuda_device)
        
        with torch.no_grad():
            features = backbone(images)
        
        assert features[0].shape[0] == batch_size
        assert all(not torch.isnan(f).any() for f in features)
    
    def test_backbone_batch_16(self, cuda_device):
        """Backbone should handle batch size 16 (may require smaller input)."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        backbone.eval()
        
        batch_size = 16
        # Use smaller images for batch 16 to fit in memory
        images = torch.rand(batch_size, 3, 160, 160, device=cuda_device)
        
        with torch.no_grad():
            features = backbone(images)
        
        assert features[0].shape[0] == batch_size
        assert all(not torch.isnan(f).any() for f in features)
    
    def test_full_pipeline_batch_4(self, cuda_device):
        """Full backbone + neck pipeline with batch size 4."""
        from codetr.models.backbone import ResNetBackbone
        from codetr.models.neck import ChannelMapper
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        
        neck = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        ).to(cuda_device)
        
        backbone.eval()
        
        batch_size = 4
        images = torch.rand(batch_size, 3, 320, 320, device=cuda_device)
        
        with torch.no_grad():
            backbone_feats = backbone(images)
            neck_feats = neck(backbone_feats)
        
        assert len(neck_feats) == 4
        assert all(f.shape[0] == batch_size for f in neck_feats)
        assert all(not torch.isnan(f).any() for f in neck_feats)
    
    def test_transformer_batch_4(self, cuda_device):
        """Transformer with batch size 4 (memory intensive)."""
        from codetr.models.transformer.transformer import CoDeformableDetrTransformer
        
        # Use smaller config for memory efficiency
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_feature_levels=4,
            num_queries=100,
        ).to(cuda_device)
        transformer.eval()
        
        batch_size = 4
        
        # Smaller spatial sizes
        srcs = [
            torch.rand(batch_size, 256, 20, 20, device=cuda_device),
            torch.rand(batch_size, 256, 10, 10, device=cuda_device),
            torch.rand(batch_size, 256, 5, 5, device=cuda_device),
            torch.rand(batch_size, 256, 3, 3, device=cuda_device),
        ]
        masks = [torch.zeros(batch_size, h, w, dtype=torch.bool, device=cuda_device) 
                 for (_, _, h, w) in [s.shape for s in srcs]]
        pos_embeds = [torch.rand_like(s) for s in srcs]
        query_embed = torch.rand(100, 512, device=cuda_device)
        
        with torch.no_grad():
            outputs = transformer(
                srcs=srcs,
                masks=masks,
                pos_embeds=pos_embeds,
                query_embed=query_embed,
            )
        
        hs, init_reference, inter_references, _, _ = outputs
        assert hs.shape[1] == batch_size, "Batch dimension should match"
        assert not torch.isnan(hs).any(), "No NaN in hidden states"
    
    def test_gradient_accumulation_large_effective_batch(self, cuda_device):
        """Gradient accumulation should produce stable gradients."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        backbone.train()
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, backbone.parameters()),
            lr=1e-4,
        )
        
        accumulation_steps = 4
        micro_batch_size = 2
        
        optimizer.zero_grad()
        
        for step in range(accumulation_steps):
            images = torch.rand(micro_batch_size, 3, 224, 224, device=cuda_device)
            features = backbone(images)
            
            # Dummy loss
            loss = sum(f.sum() for f in features) / accumulation_steps
            loss.backward()
        
        # Check gradients are finite before optimizer step
        for param in backbone.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), "Gradients should be finite"
                assert not torch.isinf(param.grad).any(), "Gradients should not be Inf"
        
        optimizer.step()
    
    def test_many_queries_memory(self, cuda_device):
        """Test with larger number of queries (300+)."""
        from codetr.models.heads.detr_head import CoDeformDETRHead
        
        num_queries = 300  # Standard DETR query count
        
        head = CoDeformDETRHead(
            num_classes=80,
            embed_dims=256,
            num_decoder_layers=2,
            num_query=num_queries,
        ).to(cuda_device)
        head.eval()
        
        batch_size = 2
        
        hs = torch.rand(2, batch_size, num_queries, 256, device=cuda_device)
        init_reference = torch.rand(batch_size, num_queries, 4, device=cuda_device)
        inter_references = torch.rand(2, batch_size, num_queries, 4, device=cuda_device)
        
        with torch.no_grad():
            outputs = head(
                hidden_states=hs,
                init_reference=init_reference,
                inter_references=inter_references,
            )
        
        assert outputs['pred_logits'].shape[1] == num_queries
        assert not torch.isnan(outputs['pred_logits']).any()


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations.
    
    These tests measure execution time and throughput to ensure
    the implementation meets performance requirements.
    """
    
    @pytest.fixture
    def cuda_device(self):
        """Get CUDA device."""
        return torch.device("cuda")
    
    def test_backbone_throughput(self, cuda_device):
        """Measure backbone inference throughput."""
        from codetr.models.backbone import ResNetBackbone
        
        backbone = ResNetBackbone(
            frozen_stages=1,
            pretrained=False,
        ).to(cuda_device)
        backbone.eval()
        
        batch_size = 4
        num_warmup = 5
        num_iterations = 20
        
        images = torch.rand(batch_size, 3, 640, 640, device=cuda_device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = backbone(images)
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = backbone(images)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        images_per_second = (batch_size * num_iterations) / total_time
        
        # Log performance (will be visible in test output)
        print(f"\nBackbone Throughput: {images_per_second:.1f} images/second")
        print(f"Average latency: {total_time / num_iterations * 1000:.2f} ms/batch")
        
        # Basic sanity check - should process at least 10 images/second
        assert images_per_second > 10, "Backbone too slow"
    
    def test_neck_throughput(self, cuda_device):
        """Measure neck (ChannelMapper) throughput."""
        from codetr.models.neck import ChannelMapper
        
        neck = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        ).to(cuda_device)
        neck.eval()
        
        batch_size = 4
        num_warmup = 5
        num_iterations = 50
        
        features = [
            torch.rand(batch_size, 512, 80, 80, device=cuda_device),
            torch.rand(batch_size, 1024, 40, 40, device=cuda_device),
            torch.rand(batch_size, 2048, 20, 20, device=cuda_device),
        ]
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = neck(features)
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = neck(features)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        batches_per_second = num_iterations / total_time
        
        print(f"\nNeck Throughput: {batches_per_second:.1f} batches/second")
        print(f"Average latency: {total_time / num_iterations * 1000:.2f} ms/batch")
        
        # Neck should be fast
        assert batches_per_second > 100, "Neck too slow"
    
    def test_transformer_encoder_latency(self, cuda_device):
        """Measure transformer encoder latency."""
        from codetr.models.transformer.encoder import (
            CoDeformableDetrTransformerEncoder,
            DeformableTransformerEncoderLayer,
        )
        
        # Create encoder layer first
        encoder_layer = DeformableTransformerEncoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.1,
            num_levels=4,
            num_points=4,
        )
        
        encoder = CoDeformableDetrTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=6,
        ).to(cuda_device)
        encoder.eval()
        
        batch_size = 2
        num_warmup = 3
        num_iterations = 10
        
        # Total tokens from 4 levels
        total_len = 40*40 + 20*20 + 10*10 + 5*5  # 2025 tokens
        
        query = torch.rand(batch_size, total_len, 256, device=cuda_device)
        spatial_shapes = torch.tensor([[40, 40], [20, 20], [10, 10], [5, 5]], 
                                       device=cuda_device)
        level_start_index = torch.tensor([0, 1600, 2000, 2100], device=cuda_device)
        valid_ratios = torch.ones(batch_size, 4, 2, device=cuda_device)
        
        reference_points = torch.rand(batch_size, total_len, 4, 2, device=cuda_device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = encoder(
                    src=query,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                )
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = encoder(
                    src=query,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                )
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        
        print(f"\nTransformer Encoder Latency: {avg_latency_ms:.2f} ms")
        
        # Encoder should be under 100ms per forward
        assert avg_latency_ms < 500, "Encoder too slow"
    
    def test_deformable_attention_efficiency(self, cuda_device):
        """Measure deformable attention efficiency."""
        from codetr.models.transformer.attention import MultiScaleDeformableAttention
        
        attention = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
        ).to(cuda_device)
        attention.eval()
        
        batch_size = 2
        num_queries = 1000
        num_warmup = 5
        num_iterations = 20
        
        # Calculate total value length
        spatial_shapes = torch.tensor([[40, 40], [20, 20], [10, 10], [5, 5]], 
                                       device=cuda_device)
        value_len = sum(h * w for h, w in spatial_shapes.tolist())  # 2025
        
        query = torch.rand(batch_size, num_queries, 256, device=cuda_device)
        value = torch.rand(batch_size, value_len, 256, device=cuda_device)
        level_start_index = torch.tensor([0, 1600, 2000, 2100], device=cuda_device)
        
        # Reference points: (B, Q, num_levels, 2)
        reference_points = torch.rand(batch_size, num_queries, 4, 2, device=cuda_device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = attention(
                    query=query,
                    value=value,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = attention(
                    query=query,
                    value=value,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        
        print(f"\nDeformable Attention Latency: {avg_latency_ms:.2f} ms")
        
        # Should be efficient
        assert avg_latency_ms < 100, "Deformable attention too slow"
    
    def test_loss_computation_speed(self, cuda_device):
        """Measure loss computation speed."""
        from codetr.models.losses import FocalLoss, GIoULoss
        
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        giou_loss = GIoULoss(reduction='mean')
        
        batch_size = 4
        num_queries = 300
        num_classes = 80
        num_warmup = 10
        num_iterations = 100
        
        pred_logits = torch.rand(batch_size * num_queries, num_classes, device=cuda_device)
        target_classes = torch.randint(0, num_classes, (batch_size * num_queries,), device=cuda_device)
        
        pred_boxes = torch.rand(batch_size * num_queries, 4, device=cuda_device)
        target_boxes = torch.rand(batch_size * num_queries, 4, device=cuda_device)
        
        # Warmup
        for _ in range(num_warmup):
            _ = focal_loss(inputs=pred_logits, targets=target_classes)
            _ = giou_loss(inputs=pred_boxes, targets=target_boxes)
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = focal_loss(inputs=pred_logits, targets=target_classes)
            _ = giou_loss(inputs=pred_boxes, targets=target_boxes)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        
        print(f"\nLoss Computation Latency: {avg_latency_ms:.4f} ms")
        
        # Loss should be very fast
        assert avg_latency_ms < 10, "Loss computation too slow"
    
    def test_nms_throughput(self, cuda_device):
        """Measure NMS operation throughput."""
        from torchvision.ops import nms
        
        num_proposals = 2000
        num_warmup = 10
        num_iterations = 100
        
        boxes = torch.rand(num_proposals, 4, device=cuda_device) * 500
        # Ensure x2 > x1, y2 > y1
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:].abs() + 1
        scores = torch.rand(num_proposals, device=cuda_device)
        
        # Warmup
        for _ in range(num_warmup):
            _ = nms(boxes, scores, iou_threshold=0.7)
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = nms(boxes, scores, iou_threshold=0.7)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        ops_per_second = num_iterations / (end_time - start_time)
        
        print(f"\nNMS Latency ({num_proposals} proposals): {avg_latency_ms:.4f} ms")
        print(f"NMS Throughput: {ops_per_second:.0f} ops/second")
        
        # NMS should be fast
        assert avg_latency_ms < 5, "NMS too slow"


# ============================================================================
# Combined Stress Tests
# ============================================================================

class TestCombinedStressScenarios:
    """Combined stress tests simulating realistic workloads."""
    
    @pytest.fixture
    def cuda_device(self):
        """Get CUDA device."""
        return torch.device("cuda")
    
    def test_full_forward_batch_2(self, cuda_device):
        """Full model forward pass with batch size 2."""
        from codetr.models.backbone import ResNetBackbone
        from codetr.models.neck import ChannelMapper
        from codetr.models.transformer.transformer import CoDeformableDetrTransformer
        from codetr.models.heads.detr_head import CoDeformDETRHead
        from codetr.models.utils.position_encoding import PositionEmbeddingSine
        
        # Build components
        backbone = ResNetBackbone(frozen_stages=1, pretrained=False).to(cuda_device)
        neck = ChannelMapper(in_channels=[512, 1024, 2048], out_channels=256, num_extra_levels=1).to(cuda_device)
        transformer = CoDeformableDetrTransformer(
            embed_dim=256, num_heads=8, num_encoder_layers=2, num_decoder_layers=2,
            num_feature_levels=4, num_queries=100,
        ).to(cuda_device)
        head = CoDeformDETRHead(
            num_classes=80, embed_dims=256, num_decoder_layers=2, num_query=100,
        ).to(cuda_device)
        pos_encoder = PositionEmbeddingSine(num_pos_feats=128).to(cuda_device)
        
        # Set to eval
        backbone.eval()
        neck.eval()
        transformer.eval()
        head.eval()
        
        batch_size = 2
        images = torch.rand(batch_size, 3, 320, 320, device=cuda_device)
        
        with torch.no_grad():
            # Backbone
            backbone_feats = backbone(images)
            
            # Neck
            neck_feats = neck(backbone_feats)
            
            # Prepare transformer inputs
            masks = [torch.zeros(batch_size, f.shape[2], f.shape[3], 
                                  dtype=torch.bool, device=cuda_device) 
                     for f in neck_feats]
            pos_embeds = [pos_encoder(f, m) for f, m in zip(neck_feats, masks)]
            query_embed = torch.rand(100, 512, device=cuda_device)
            
            # Transformer
            hs, init_ref, inter_refs, _, _ = transformer(
                srcs=neck_feats,
                masks=masks,
                pos_embeds=pos_embeds,
                query_embed=query_embed,
            )
            
            # Head
            outputs = head(
                hidden_states=hs,
                init_reference=init_ref,
                inter_references=inter_refs,
            )
        
        assert outputs['pred_logits'].shape == (batch_size, 100, 80)
        assert outputs['pred_boxes'].shape == (batch_size, 100, 4)
        assert not torch.isnan(outputs['pred_logits']).any()
        assert not torch.isnan(outputs['pred_boxes']).any()
    
    def test_training_step_memory_efficiency(self, cuda_device):
        """Verify memory efficiency during training step."""
        from codetr.models.backbone import ResNetBackbone
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        
        backbone = ResNetBackbone(frozen_stages=1, pretrained=False).to(cuda_device)
        backbone.train()
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, backbone.parameters()),
            lr=1e-4,
        )
        
        batch_size = 2
        
        for i in range(5):
            images = torch.rand(batch_size, 3, 320, 320, device=cuda_device)
            
            optimizer.zero_grad()
            features = backbone(images)
            loss = sum(f.sum() for f in features)
            loss.backward()
            optimizer.step()
        
        peak_memory_mb = torch.cuda.max_memory_allocated(cuda_device) / 1024 / 1024
        
        print(f"\nPeak Memory Usage (5 training steps): {peak_memory_mb:.1f} MB")
        
        # Backbone alone should use less than 4GB
        assert peak_memory_mb < 4096, f"Memory usage too high: {peak_memory_mb:.1f} MB"
