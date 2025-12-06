"""Weight Conversion Tool for Co-DETR.

This module provides utilities to convert MMDetection pretrained Co-DETR
checkpoints to the pure PyTorch model format.

The conversion handles:
- ResNet-50 backbone weights
- ChannelMapper neck weights
- Transformer encoder/decoder weights
- Detection head weights (DETR, RPN, RoI, ATSS)

Usage:
    python tools/convert_weights.py --input mmdet.pth --output converted.pth
    python tools/convert_weights.py --input mmdet.pth --output converted.pth --verify
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load and analyze MMDetection checkpoint.

    Args:
        checkpoint_path: Path to MMDetection checkpoint file.

    Returns:
        State dict from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # MMDetection checkpoints can have different structures
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    return state_dict


def print_checkpoint_summary(state_dict: Dict[str, torch.Tensor]) -> None:
    """Print summary of checkpoint keys grouped by component.

    Args:
        state_dict: Model state dictionary.
    """
    components: Dict[str, List[str]] = {
        'backbone': [],
        'neck': [],
        'query_head.transformer.encoder': [],
        'query_head.transformer.decoder': [],
        'query_head.transformer.other': [],
        'query_head.other': [],
        'rpn_head': [],
        'roi_head': [],
        'bbox_head': [],
        'other': [],
    }
    
    for key in sorted(state_dict.keys()):
        categorized = False
        for comp in ['backbone', 'neck', 'rpn_head', 'roi_head', 'bbox_head']:
            if key.startswith(comp):
                components[comp].append(key)
                categorized = True
                break
        
        if not categorized:
            if key.startswith('query_head.transformer.encoder'):
                components['query_head.transformer.encoder'].append(key)
            elif key.startswith('query_head.transformer.decoder'):
                components['query_head.transformer.decoder'].append(key)
            elif key.startswith('query_head.transformer'):
                components['query_head.transformer.other'].append(key)
            elif key.startswith('query_head'):
                components['query_head.other'].append(key)
            else:
                components['other'].append(key)
    
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    
    for comp, keys in components.items():
        if keys:
            print(f"\n{comp}: {len(keys)} keys")
            if len(keys) <= 5:
                for k in keys:
                    print(f"  - {k}: {state_dict[k].shape}")
            else:
                for k in keys[:3]:
                    print(f"  - {k}: {state_dict[k].shape}")
                print(f"  ... and {len(keys) - 3} more")


# =============================================================================
# Key Mapping Functions
# =============================================================================

def map_backbone_key(old_key: str) -> Optional[str]:
    """Map MMDetection backbone key to new format.

    MMDetection key format:
        backbone.conv1.weight
        backbone.bn1.weight
        backbone.layer1.0.conv1.weight
        backbone.layer1.0.bn1.weight
        ...

    New format (torchvision ResNet-50 wrapper):
        backbone.conv1.weight
        backbone.bn1.weight
        backbone.layer1.0.conv1.weight
        backbone.layer1.0.bn1.weight
        ...

    The backbone keys are mostly identical since both use torchvision ResNet.

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not a backbone key.
    """
    if not old_key.startswith('backbone.'):
        return None
    
    # Most backbone keys are direct mappings
    # Both use torchvision ResNet structure
    return old_key


def map_neck_key(old_key: str) -> Optional[str]:
    """Map MMDetection neck key to new format.

    MMDetection ChannelMapper key format:
        neck.convs.0.conv.weight       -> lateral conv for first input
        neck.convs.0.gn.weight         -> GroupNorm for first level
        neck.convs.1.conv.weight       -> lateral conv for second input
        neck.extra_convs.0.conv.weight -> extra level conv
        neck.extra_convs.0.gn.weight   -> extra level GroupNorm

    New format:
        neck.lateral_convs.0.weight
        neck.lateral_norms.0.weight
        neck.lateral_convs.1.weight
        neck.extra_convs.0.weight
        neck.extra_norms.0.weight

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not a neck key.
    """
    if not old_key.startswith('neck.'):
        return None
    
    # Handle lateral convolutions
    # neck.convs.{i}.conv.weight -> neck.lateral_convs.{i}.weight
    match = re.match(r'neck\.convs\.(\d+)\.conv\.(.+)', old_key)
    if match:
        idx = match.group(1)
        suffix = match.group(2)
        return f'neck.lateral_convs.{idx}.{suffix}'
    
    # Handle lateral GroupNorm
    # neck.convs.{i}.gn.weight -> neck.lateral_norms.{i}.weight
    match = re.match(r'neck\.convs\.(\d+)\.gn\.(.+)', old_key)
    if match:
        idx = match.group(1)
        suffix = match.group(2)
        return f'neck.lateral_norms.{idx}.{suffix}'
    
    # Handle extra convolutions
    # neck.extra_convs.{i}.conv.weight -> neck.extra_convs.{i}.weight
    match = re.match(r'neck\.extra_convs\.(\d+)\.conv\.(.+)', old_key)
    if match:
        idx = match.group(1)
        suffix = match.group(2)
        return f'neck.extra_convs.{idx}.{suffix}'
    
    # Handle extra GroupNorm
    # neck.extra_convs.{i}.gn.weight -> neck.extra_norms.{i}.weight
    match = re.match(r'neck\.extra_convs\.(\d+)\.gn\.(.+)', old_key)
    if match:
        idx = match.group(1)
        suffix = match.group(2)
        return f'neck.extra_norms.{idx}.{suffix}'
    
    return None


def map_transformer_key(old_key: str) -> Optional[str]:
    """Map MMDetection transformer key to new format.

    MMDetection transformer is inside query_head:
        query_head.transformer.encoder.layers.{i}.attentions.0.* (deformable attn)
        query_head.transformer.encoder.layers.{i}.ffns.0.*
        query_head.transformer.encoder.layers.{i}.norms.{j}.*
        query_head.transformer.decoder.layers.{i}.attentions.{j}.*
        query_head.transformer.decoder.layers.{i}.ffns.0.*
        query_head.transformer.decoder.layers.{i}.norms.{j}.*
        query_head.transformer.level_embeds
        query_head.transformer.enc_output
        query_head.transformer.enc_output_norm

    New format:
        transformer.encoder.layers.{i}.self_attn.*
        transformer.encoder.layers.{i}.ffn.0.*
        transformer.encoder.layers.{i}.norm1.*/norm2.*
        transformer.decoder.layers.{i}.self_attn.*
        transformer.decoder.layers.{i}.cross_attn.*
        transformer.decoder.layers.{i}.ffn.0.*
        transformer.decoder.layers.{i}.norm1.*/norm2.*/norm3.*
        transformer.level_embeds
        transformer.enc_output
        transformer.enc_output_norm

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not a transformer key.
    """
    if not old_key.startswith('query_head.transformer.'):
        return None
    
    # Remove query_head. prefix
    key = old_key[len('query_head.'):]
    
    # === ENCODER ===
    # Encoder self-attention
    # transformer.encoder.layers.{i}.attentions.0.* -> transformer.encoder.layers.{i}.self_attn.*
    match = re.match(r'transformer\.encoder\.layers\.(\d+)\.attentions\.0\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        suffix = match.group(2)
        return f'transformer.encoder.layers.{layer_idx}.self_attn.{suffix}'
    
    # Encoder FFN
    # transformer.encoder.layers.{i}.ffns.0.layers.{j}.* -> transformer.encoder.layers.{i}.ffn.{j}.*
    match = re.match(r'transformer\.encoder\.layers\.(\d+)\.ffns\.0\.layers\.(\d+)\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        ffn_layer = match.group(2)
        suffix = match.group(3)
        return f'transformer.encoder.layers.{layer_idx}.ffn.{ffn_layer}.{suffix}'
    
    # Encoder norms
    # transformer.encoder.layers.{i}.norms.{0|1}.* -> transformer.encoder.layers.{i}.norm{1|2}.*
    match = re.match(r'transformer\.encoder\.layers\.(\d+)\.norms\.(\d+)\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        norm_idx = int(match.group(2))
        suffix = match.group(3)
        new_norm_name = f'norm{norm_idx + 1}'
        return f'transformer.encoder.layers.{layer_idx}.{new_norm_name}.{suffix}'
    
    # === DECODER ===
    # Decoder self-attention (attentions.0)
    # transformer.decoder.layers.{i}.attentions.0.* -> transformer.decoder.layers.{i}.self_attn.*
    match = re.match(r'transformer\.decoder\.layers\.(\d+)\.attentions\.0\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        suffix = match.group(2)
        return f'transformer.decoder.layers.{layer_idx}.self_attn.{suffix}'
    
    # Decoder cross-attention (attentions.1)
    # transformer.decoder.layers.{i}.attentions.1.* -> transformer.decoder.layers.{i}.cross_attn.*
    match = re.match(r'transformer\.decoder\.layers\.(\d+)\.attentions\.1\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        suffix = match.group(2)
        return f'transformer.decoder.layers.{layer_idx}.cross_attn.{suffix}'
    
    # Decoder FFN
    # transformer.decoder.layers.{i}.ffns.0.layers.{j}.* -> transformer.decoder.layers.{i}.ffn.{j}.*
    match = re.match(r'transformer\.decoder\.layers\.(\d+)\.ffns\.0\.layers\.(\d+)\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        ffn_layer = match.group(2)
        suffix = match.group(3)
        return f'transformer.decoder.layers.{layer_idx}.ffn.{ffn_layer}.{suffix}'
    
    # Decoder norms (3 norms: self_attn, cross_attn, ffn)
    # transformer.decoder.layers.{i}.norms.{0|1|2}.* -> transformer.decoder.layers.{i}.norm{1|2|3}.*
    match = re.match(r'transformer\.decoder\.layers\.(\d+)\.norms\.(\d+)\.(.+)', key)
    if match:
        layer_idx = match.group(1)
        norm_idx = int(match.group(2))
        suffix = match.group(3)
        new_norm_name = f'norm{norm_idx + 1}'
        return f'transformer.decoder.layers.{layer_idx}.{new_norm_name}.{suffix}'
    
    # === TRANSFORMER TOP-LEVEL ===
    # Direct mappings for top-level transformer components
    direct_mappings = [
        'transformer.level_embeds',
        'transformer.enc_output.weight',
        'transformer.enc_output.bias',
        'transformer.enc_output_norm.weight',
        'transformer.enc_output_norm.bias',
        'transformer.pos_trans.weight',
        'transformer.pos_trans.bias',
        'transformer.pos_trans_norm.weight',
        'transformer.pos_trans_norm.bias',
        'transformer.reference_points.weight',
        'transformer.reference_points.bias',
    ]
    
    for mapping in direct_mappings:
        if key == mapping:
            return key
    
    # Handle level_embeds
    if key == 'transformer.level_embeds':
        return key
    
    return None


def map_query_head_key(old_key: str) -> Optional[str]:
    """Map MMDetection query head key to new format.

    MMDetection query head:
        query_head.cls_branches.{i}.* -> query_head.cls_branches.{i}.*
        query_head.reg_branches.{i}.{j}.* -> query_head.reg_branches.{i}.{j}.*
        query_head.query_embedding.weight -> query_head.query_embedding.weight

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not a query head key.
    """
    if not old_key.startswith('query_head.'):
        return None
    
    # Skip transformer keys (handled separately)
    if old_key.startswith('query_head.transformer.'):
        return None
    
    # Most query head keys are direct mappings
    # cls_branches, reg_branches, query_embedding
    return old_key


def map_rpn_head_key(old_key: str) -> Optional[str]:
    """Map MMDetection RPN head key to new format.

    Most keys are direct mappings.

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not an RPN head key.
    """
    if not old_key.startswith('rpn_head.'):
        return None
    
    # Most RPN keys are direct mappings
    return old_key


def map_roi_head_key(old_key: str) -> Optional[str]:
    """Map MMDetection RoI head key to new format.

    MMDetection uses list: roi_head.0.* -> roi_head.*

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not a RoI head key.
    """
    if not old_key.startswith('roi_head.'):
        return None
    
    # Remove list index: roi_head.0.* -> roi_head.*
    match = re.match(r'roi_head\.0\.(.+)', old_key)
    if match:
        return f'roi_head.{match.group(1)}'
    
    return old_key


def map_atss_head_key(old_key: str) -> Optional[str]:
    """Map MMDetection ATSS head key to new format.

    MMDetection uses bbox_head.0.* for ATSS -> atss_head.*

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if not an ATSS head key.
    """
    if not old_key.startswith('bbox_head.'):
        return None
    
    # Remove list index and rename: bbox_head.0.* -> atss_head.*
    match = re.match(r'bbox_head\.0\.(.+)', old_key)
    if match:
        return f'atss_head.{match.group(1)}'
    
    # Handle case without list index
    match = re.match(r'bbox_head\.(.+)', old_key)
    if match:
        return f'atss_head.{match.group(1)}'
    
    return None


def convert_key(old_key: str) -> Optional[str]:
    """Convert MMDetection key to new model format.

    Args:
        old_key: MMDetection checkpoint key.

    Returns:
        New model key or None if no mapping exists.
    """
    mappers = [
        map_backbone_key,
        map_neck_key,
        map_transformer_key,
        map_query_head_key,
        map_rpn_head_key,
        map_roi_head_key,
        map_atss_head_key,
    ]
    
    for mapper in mappers:
        new_key = mapper(old_key)
        if new_key is not None:
            return new_key
    
    return None


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_checkpoint(
    source_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Convert MMDetection checkpoint to new model format.

    Args:
        source_state_dict: MMDetection model state dict.
        strict: If True, raise error for unmapped keys.
        verbose: If True, print conversion progress.

    Returns:
        Tuple of:
            - Converted state dict
            - List of successfully converted keys
            - List of skipped keys (no mapping)
    """
    converted_state_dict = {}
    converted_keys = []
    skipped_keys = []
    
    for old_key, value in source_state_dict.items():
        new_key = convert_key(old_key)
        
        if new_key is not None:
            converted_state_dict[new_key] = value
            converted_keys.append(old_key)
            if verbose:
                print(f"  [OK] {old_key} -> {new_key}")
        else:
            skipped_keys.append(old_key)
            if verbose:
                print(f"  [SKIP] {old_key}")
    
    if strict and skipped_keys:
        raise ValueError(
            f"Strict mode: {len(skipped_keys)} keys could not be mapped:\n"
            + "\n".join(f"  - {k}" for k in skipped_keys[:10])
        )
    
    return converted_state_dict, converted_keys, skipped_keys


def verify_conversion(
    converted_state_dict: Dict[str, torch.Tensor],
    num_classes: int = 80,
    verbose: bool = True,
) -> bool:
    """Verify converted weights by loading into new model.

    Args:
        converted_state_dict: Converted model state dict.
        num_classes: Number of object classes.
        verbose: If True, print verification details.

    Returns:
        True if verification passed, False otherwise.
    """
    try:
        # Import model
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from codetr.models.detector import CoDETR
        
        # Create model with matching config
        if verbose:
            print("\nCreating model for verification...")
        
        model = CoDETR(
            num_classes=num_classes,
            pretrained_backbone=False,  # Don't load pretrained weights
            use_rpn=True,
            use_roi=True,
            use_atss=True,
            use_dn=True,
        )
        
        # Get model state dict for comparison
        model_state_dict = model.state_dict()
        
        if verbose:
            print(f"Model has {len(model_state_dict)} parameters")
            print(f"Converted checkpoint has {len(converted_state_dict)} parameters")
        
        # Check for matching keys
        model_keys = set(model_state_dict.keys())
        converted_keys = set(converted_state_dict.keys())
        
        matched_keys = model_keys & converted_keys
        missing_in_checkpoint = model_keys - converted_keys
        extra_in_checkpoint = converted_keys - model_keys
        
        if verbose:
            print(f"\nMatched keys: {len(matched_keys)}")
            print(f"Missing in checkpoint: {len(missing_in_checkpoint)}")
            print(f"Extra in checkpoint: {len(extra_in_checkpoint)}")
            
            if missing_in_checkpoint:
                print("\nKeys missing in checkpoint (first 10):")
                for k in sorted(missing_in_checkpoint)[:10]:
                    print(f"  - {k}")
            
            if extra_in_checkpoint:
                print("\nExtra keys in checkpoint (first 10):")
                for k in sorted(extra_in_checkpoint)[:10]:
                    print(f"  - {k}")
        
        # Check shapes for matched keys
        shape_mismatches = []
        for key in matched_keys:
            model_shape = model_state_dict[key].shape
            converted_shape = converted_state_dict[key].shape
            if model_shape != converted_shape:
                shape_mismatches.append((key, model_shape, converted_shape))
        
        if shape_mismatches:
            if verbose:
                print(f"\nShape mismatches: {len(shape_mismatches)}")
                for key, model_shape, converted_shape in shape_mismatches[:10]:
                    print(f"  - {key}: model={model_shape}, checkpoint={converted_shape}")
            return False
        
        # Try loading weights (non-strict)
        if verbose:
            print("\nLoading converted weights into model...")
        
        missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
        
        if verbose:
            print(f"Missing keys after load: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
        
        # Try a forward pass
        if verbose:
            print("\nRunning test forward pass...")
        
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)  # Small input for quick test
            try:
                output = model(dummy_input)
                if verbose:
                    print("Forward pass successful!")
                    if isinstance(output, list):
                        print(f"Output: {len(output)} detections")
                    else:
                        print(f"Output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")
                return True
            except Exception as e:
                if verbose:
                    print(f"Forward pass failed: {e}")
                return False
                
    except Exception as e:
        if verbose:
            print(f"Verification failed: {e}")
        return False


def main():
    """Main entry point for weight conversion."""
    parser = argparse.ArgumentParser(
        description="Convert MMDetection Co-DETR weights to pure PyTorch format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to MMDetection checkpoint file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save converted checkpoint"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of object classes (default: 80 for COCO)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted weights by loading into model"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any keys cannot be mapped"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed conversion progress"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze checkpoint, don't convert"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print("=" * 60)
    print("Co-DETR Weight Conversion Tool")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    source_state_dict = analyze_checkpoint(args.input)
    print(f"Loaded {len(source_state_dict)} keys")
    
    # Print summary
    print_checkpoint_summary(source_state_dict)
    
    if args.analyze_only:
        print("\n[Analyze-only mode, exiting]")
        return 0
    
    # Convert
    print("\n" + "=" * 60)
    print("CONVERTING WEIGHTS")
    print("=" * 60)
    
    converted_state_dict, converted_keys, skipped_keys = convert_checkpoint(
        source_state_dict,
        strict=args.strict,
        verbose=args.verbose,
    )
    
    print(f"\nConversion complete:")
    print(f"  - Converted: {len(converted_keys)} keys")
    print(f"  - Skipped:   {len(skipped_keys)} keys")
    
    # Verify if requested
    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        
        success = verify_conversion(
            converted_state_dict,
            num_classes=args.num_classes,
            verbose=args.verbose,
        )
        
        if not success:
            print("\n[WARNING] Verification failed!")
    
    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'state_dict': converted_state_dict,
        'meta': {
            'source': str(args.input),
            'num_classes': args.num_classes,
            'conversion_script': 'tools/convert_weights.py',
        }
    }, args.output)
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
