#!/usr/bin/env python3
"""Generate synthetic sample dataset for overfitting tests.

This script creates a small synthetic dataset in YOLOv5 format for validating
that the full training pipeline works correctly. The dataset consists of
simple colored shapes (rectangles) on random backgrounds.

Usage:
    python tools/create_sample_dataset.py --output data/sample --num-train 20 --num-val 5

The generated dataset structure:
    data/sample/
    ├── train/
    │   ├── images/
    │   │   ├── img_0000.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img_0000.txt
    │       └── ...
    └── val/
        ├── images/
        └── labels/

Label format (YOLOv5): class_id center_x center_y width height (normalized 0-1)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# Default shape colors by class
CLASS_COLORS = [
    (255, 0, 0),      # Class 0: Red
    (0, 255, 0),      # Class 1: Green
    (0, 0, 255),      # Class 2: Blue
    (255, 255, 0),    # Class 3: Yellow
    (255, 0, 255),    # Class 4: Magenta
]

# Class names for reference
CLASS_NAMES = [
    "red_box",
    "green_box",
    "blue_box",
    "yellow_box",
    "magenta_box",
]


def generate_random_background(width: int, height: int) -> Image.Image:
    """Generate a random colored background.
    
    Args:
        width: Image width.
        height: Image height.
        
    Returns:
        PIL Image with random background.
    """
    # Random solid color or gradient
    if random.random() < 0.5:
        # Solid color (muted colors to contrast with shapes)
        r = random.randint(100, 200)
        g = random.randint(100, 200)
        b = random.randint(100, 200)
        img = Image.new("RGB", (width, height), (r, g, b))
    else:
        # Simple gradient
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        r1, g1, b1 = random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)
        r2, g2, b2 = random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)
        for y in range(height):
            ratio = y / height
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            for x in range(width):
                pixels[x, y] = (r, g, b)
    
    return img


def add_random_noise(img: Image.Image, noise_level: float = 0.02) -> Image.Image:
    """Add slight noise to image for more realistic appearance.
    
    Args:
        img: Input PIL Image.
        noise_level: Standard deviation of noise (as fraction of 255).
        
    Returns:
        Image with added noise.
    """
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, noise_level * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_boxes(
    num_boxes: int,
    num_classes: int,
    img_width: int,
    img_height: int,
    min_size: float = 0.1,
    max_size: float = 0.3,
) -> List[Tuple[int, float, float, float, float]]:
    """Generate random bounding boxes.
    
    Args:
        num_boxes: Number of boxes to generate.
        num_classes: Number of classes available.
        img_width: Image width.
        img_height: Image height.
        min_size: Minimum box size as fraction of image.
        max_size: Maximum box size as fraction of image.
        
    Returns:
        List of (class_id, cx, cy, w, h) tuples, all normalized to [0, 1].
    """
    boxes = []
    
    for _ in range(num_boxes):
        # Random class
        class_id = random.randint(0, num_classes - 1)
        
        # Random size (normalized)
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)
        
        # Random center position (ensure box stays within image)
        cx = random.uniform(w / 2, 1 - w / 2)
        cy = random.uniform(h / 2, 1 - h / 2)
        
        boxes.append((class_id, cx, cy, w, h))
    
    return boxes


def draw_boxes(
    img: Image.Image,
    boxes: List[Tuple[int, float, float, float, float]],
) -> Image.Image:
    """Draw colored rectangles on image.
    
    Args:
        img: PIL Image to draw on.
        boxes: List of (class_id, cx, cy, w, h) tuples (normalized coordinates).
        
    Returns:
        Image with drawn boxes.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    
    for class_id, cx, cy, w, h in boxes:
        # Convert normalized to pixel coordinates
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        
        # Get color for this class
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        
        # Draw filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
    
    return img


def save_yolo_labels(
    boxes: List[Tuple[int, float, float, float, float]],
    filepath: Path,
) -> None:
    """Save boxes in YOLOv5 label format.
    
    Args:
        boxes: List of (class_id, cx, cy, w, h) tuples.
        filepath: Path to save label file.
    """
    with open(filepath, "w") as f:
        for class_id, cx, cy, w, h in boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def generate_dataset(
    output_dir: Path,
    num_train: int = 20,
    num_val: int = 5,
    num_classes: int = 2,
    img_width: int = 640,
    img_height: int = 640,
    min_boxes: int = 1,
    max_boxes: int = 3,
) -> None:
    """Generate complete sample dataset.
    
    Args:
        output_dir: Output directory.
        num_train: Number of training images.
        num_val: Number of validation images.
        num_classes: Number of object classes.
        img_width: Image width.
        img_height: Image height.
        min_boxes: Minimum number of boxes per image.
        max_boxes: Maximum number of boxes per image.
    """
    print(f"Generating sample dataset in: {output_dir}")
    print(f"  Training images: {num_train}")
    print(f"  Validation images: {num_val}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Image size: {img_width}x{img_height}")
    
    # Create directory structure
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Generate training and validation images
    for split, num_images in [("train", num_train), ("val", num_val)]:
        print(f"\nGenerating {split} split...")
        
        for i in range(num_images):
            # Generate background
            img = generate_random_background(img_width, img_height)
            
            # Generate random boxes
            num_boxes = random.randint(min_boxes, max_boxes)
            boxes = generate_boxes(
                num_boxes=num_boxes,
                num_classes=num_classes,
                img_width=img_width,
                img_height=img_height,
            )
            
            # Draw boxes on image
            img = draw_boxes(img, boxes)
            
            # Add slight noise
            img = add_random_noise(img, noise_level=0.01)
            
            # Save image
            img_filename = f"img_{i:04d}.jpg"
            img_path = output_dir / split / "images" / img_filename
            img.save(img_path, quality=95)
            
            # Save labels
            label_filename = f"img_{i:04d}.txt"
            label_path = output_dir / split / "labels" / label_filename
            save_yolo_labels(boxes, label_path)
            
            print(f"  [{i+1}/{num_images}] {img_filename} - {num_boxes} boxes")
    
    # Save class names file
    names_path = output_dir / "classes.txt"
    with open(names_path, "w") as f:
        for i in range(num_classes):
            f.write(f"{CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'}\n")
    
    print(f"\nDataset generation complete!")
    print(f"Class names saved to: {names_path}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({num_train} images)")
    print(f"  │   └── labels/ ({num_train} labels)")
    print(f"  ├── val/")
    print(f"  │   ├── images/ ({num_val} images)")
    print(f"  │   └── labels/ ({num_val} labels)")
    print(f"  └── classes.txt")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample dataset for overfitting tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample",
        help="Output directory for the generated dataset.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=20,
        help="Number of training images to generate.",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=5,
        help="Number of validation images to generate.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of object classes.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size (width and height).",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=1,
        help="Minimum number of boxes per image.",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=3,
        help="Maximum number of boxes per image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate dataset
    generate_dataset(
        output_dir=Path(args.output),
        num_train=args.num_train,
        num_val=args.num_val,
        num_classes=args.num_classes,
        img_width=args.img_size,
        img_height=args.img_size,
        min_boxes=args.min_boxes,
        max_boxes=args.max_boxes,
    )


if __name__ == "__main__":
    main()
