#!/usr/bin/env python3
"""Inference script for Co-DETR object detection.

This script provides a command-line interface for running inference
on single images or batches of images using a trained Co-DETR model.

Usage:
    Single image:
        python tools/inference.py --checkpoint model.pth --image test.jpg --output output/
    
    Batch inference:
        python tools/inference.py --checkpoint model.pth --image_dir images/ --output output/
    
    With visualization:
        python tools/inference.py --checkpoint model.pth --image test.jpg --output output/ --visualize

Example:
    >>> python tools/inference.py \\
    ...     --checkpoint checkpoints/co_detr_r50.pth \\
    ...     --image data/test.jpg \\
    ...     --output output/ \\
    ...     --score_threshold 0.3 \\
    ...     --device cuda \\
    ...     --visualize
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codetr.models.detector import CoDETR, build_codetr
from codetr.data.transforms import Compose, ToTensor, Normalize, Resize
from codetr.data.transforms.transforms import IMAGENET_MEAN, IMAGENET_STD


def build_inference_transforms(
    min_size: int = 800,
    max_size: int = 1333,
) -> Compose:
    """Build transformation pipeline for inference.
    
    Args:
        min_size: Minimum size of shorter side.
        max_size: Maximum size of longer side.
        
    Returns:
        Composed transform pipeline.
    """
    return Compose([
        Resize(min_size=min_size, max_size=max_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_path: str) -> Image.Image:
    """Load image from file.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        PIL Image in RGB mode.
        
    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path).convert("RGB")


def load_model(
    checkpoint_path: str,
    num_classes: int = 80,
    device: torch.device = torch.device("cpu"),
    use_aux_heads: bool = False,
) -> CoDETR:
    """Load Co-DETR model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        num_classes: Number of object classes.
        device: Device to load model on.
        use_aux_heads: Whether to use auxiliary heads.
        
    Returns:
        Loaded CoDETR model in eval mode.
    """
    # Build model
    model = build_codetr(
        num_classes=num_classes,
        pretrained_backbone=False,  # Will load from checkpoint
        use_aux_heads=use_aux_heads,
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict (allow partial loading)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(
    image: Image.Image,
    transforms: Compose,
    device: torch.device,
) -> Tuple[Tensor, Tuple[int, int]]:
    """Preprocess image for model input.
    
    Args:
        image: PIL Image to preprocess.
        transforms: Transform pipeline.
        device: Device to place tensor on.
        
    Returns:
        Tuple of (preprocessed tensor, original size).
    """
    original_size = (image.height, image.width)
    
    # Apply transforms (returns tensor and optional target)
    tensor, _ = transforms(image, None)
    
    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor, original_size


def postprocess_predictions(
    predictions: List[Dict[str, Tensor]],
    original_sizes: List[Tuple[int, int]],
    input_sizes: List[Tuple[int, int]],
) -> List[Dict[str, Tensor]]:
    """Convert predictions from normalized to absolute coordinates.
    
    Args:
        predictions: List of prediction dicts with normalized boxes.
        original_sizes: Original (H, W) sizes of images.
        input_sizes: Input (H, W) sizes after preprocessing.
        
    Returns:
        Predictions with boxes in absolute coordinates.
    """
    processed = []
    
    for pred, orig_size, input_size in zip(predictions, original_sizes, input_sizes):
        boxes = pred["boxes"].clone()
        
        # Model outputs normalized xyxy boxes [0, 1]
        # Scale to original image size
        if boxes.numel() > 0:
            orig_h, orig_w = orig_size
            boxes[:, 0] *= orig_w  # x1
            boxes[:, 1] *= orig_h  # y1
            boxes[:, 2] *= orig_w  # x2
            boxes[:, 3] *= orig_h  # y2
            
            # Clamp to image bounds
            boxes[:, 0].clamp_(0, orig_w)
            boxes[:, 1].clamp_(0, orig_h)
            boxes[:, 2].clamp_(0, orig_w)
            boxes[:, 3].clamp_(0, orig_h)
        
        processed.append({
            "boxes": boxes,
            "scores": pred["scores"],
            "labels": pred["labels"],
        })
    
    return processed


@torch.no_grad()
def run_inference(
    model: CoDETR,
    image_paths: List[str],
    transforms: Compose,
    device: torch.device,
    score_threshold: float = 0.3,
) -> Tuple[List[Dict[str, Tensor]], List[float]]:
    """Run inference on a list of images.
    
    Args:
        model: CoDETR model in eval mode.
        image_paths: List of image paths.
        transforms: Preprocessing transforms.
        device: Device to run inference on.
        score_threshold: Minimum score threshold.
        
    Returns:
        Tuple of (predictions, inference times per image).
    """
    all_predictions = []
    inference_times = []
    
    for image_path in image_paths:
        # Load and preprocess
        image = load_image(image_path)
        tensor, orig_size = preprocess_image(image, transforms, device)
        input_size = (tensor.shape[2], tensor.shape[3])
        
        # Run inference
        start_time = time.time()
        predictions = model(tensor)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Postprocess
        predictions = postprocess_predictions(predictions, [orig_size], [input_size])
        
        # Filter by score threshold
        pred = predictions[0]
        if pred["scores"].numel() > 0:
            mask = pred["scores"] >= score_threshold
            pred = {
                "boxes": pred["boxes"][mask],
                "scores": pred["scores"][mask],
                "labels": pred["labels"][mask],
            }
        
        all_predictions.append(pred)
    
    return all_predictions, inference_times


def save_predictions_json(
    predictions: List[Dict[str, Tensor]],
    image_paths: List[str],
    output_path: str,
) -> None:
    """Save predictions to JSON file.
    
    Args:
        predictions: List of prediction dictionaries.
        image_paths: List of image paths.
        output_path: Path to save JSON file.
    """
    results = []
    
    for pred, image_path in zip(predictions, image_paths):
        result = {
            "image": os.path.basename(image_path),
            "detections": []
        }
        
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            result["detections"].append({
                "bbox": box.tolist(),
                "score": float(score),
                "class": int(label),
            })
        
        results.append(result)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved predictions to {output_path}")


def get_default_class_names(num_classes: int = 80) -> List[str]:
    """Get default COCO class names.
    
    Args:
        num_classes: Number of classes.
        
    Returns:
        List of class names.
    """
    # COCO class names
    coco_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    if num_classes <= len(coco_names):
        return coco_names[:num_classes]
    else:
        return coco_names + [f"class_{i}" for i in range(len(coco_names), num_classes)]


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Run Co-DETR inference on images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to single image",
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images for batch inference",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results",
    )
    
    # Model options
    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of object classes",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum score threshold for detections",
    )
    
    # Processing options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=800,
        help="Minimum image size",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1333,
        help="Maximum image size",
    )
    
    # Output options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with bounding boxes",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save predictions as JSON",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        help="Path to file with class names (one per line)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Build transforms
    transforms = build_inference_transforms(
        min_size=args.min_size,
        max_size=args.max_size,
    )
    
    # Load model
    print("Loading model...")
    model = load_model(
        args.checkpoint,
        num_classes=args.num_classes,
        device=device,
    )
    
    # Load class names
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, "r") as f:
            class_names = [line.strip() for line in f]
    else:
        class_names = get_default_class_names(args.num_classes)
    
    # Get input images
    if args.image:
        image_paths = [args.image]
    else:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        image_paths.sort()
    
    if not image_paths:
        print("No images found")
        return
    
    print(f"Processing {len(image_paths)} image(s)...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run inference
    predictions, inference_times = run_inference(
        model,
        image_paths,
        transforms,
        device,
        args.score_threshold,
    )
    
    # Print timing info
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    
    # Print detection counts
    for image_path, pred in zip(image_paths, predictions):
        num_dets = len(pred["boxes"])
        print(f"  {os.path.basename(image_path)}: {num_dets} detections")
    
    # Save visualizations
    if args.visualize:
        try:
            from visualize import visualize_predictions, tensor_to_numpy
            
            for image_path, pred in zip(image_paths, predictions):
                basename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(args.output, f"{basename}_vis.jpg")
                
                pred_np = {
                    "boxes": tensor_to_numpy(pred["boxes"]),
                    "scores": tensor_to_numpy(pred["scores"]),
                    "labels": tensor_to_numpy(pred["labels"]),
                }
                
                visualize_predictions(
                    image_path,
                    pred_np,
                    class_names,
                    output_path,
                )
            
            print(f"Saved visualizations to {args.output}")
        except ImportError:
            print("Warning: Could not import visualize module, skipping visualization")
    
    # Save JSON
    if args.save_json:
        json_path = os.path.join(args.output, "predictions.json")
        save_predictions_json(predictions, image_paths, json_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
