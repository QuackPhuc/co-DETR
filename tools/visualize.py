"""Visualization tools for object detection.

This module provides functions for drawing bounding boxes, labels, and scores
on images for detection visualization and result presentation.

Key features:
    - Color-coded bounding boxes by class
    - Score and label text overlay
    - Support for PIL and numpy images
    - Automatic color palette generation

Example:
    >>> from tools.visualize import draw_boxes, visualize_predictions
    >>> image = visualize_predictions(
    ...     image_path="test.jpg",
    ...     predictions=predictions,
    ...     class_names=["person", "car", "dog"],
    ...     output_path="output.jpg"
    ... )
"""

import colorsys
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_color_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for each class.
    
    Uses HSV color space with evenly distributed hues for maximum distinction.
    
    Args:
        num_classes: Number of classes to generate colors for.
        
    Returns:
        List of RGB color tuples (R, G, B) with values in [0, 255].
        
    Example:
        >>> palette = create_color_palette(10)
        >>> print(palette[0])  # First class color
    """
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.75
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_boxes(
    image: Union[Image.Image, np.ndarray],
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color_palette: Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """Draw bounding boxes with labels and scores on image.
    
    Args:
        image: Input image (PIL Image or numpy array).
        boxes: Bounding boxes in xyxy format. Shape: (N, 4).
        labels: Class labels for each box. Shape: (N,).
        scores: Optional confidence scores. Shape: (N,).
        class_names: Optional list of class names for label text.
        color_palette: Optional color palette for classes.
        line_width: Width of bounding box lines.
        font_size: Size of label text font.
        
    Returns:
        PIL Image with drawn boxes.
        
    Example:
        >>> image = Image.open("test.jpg")
        >>> boxes = np.array([[100, 100, 200, 200], [150, 150, 300, 300]])
        >>> labels = np.array([0, 1])
        >>> scores = np.array([0.95, 0.87])
        >>> result = draw_boxes(image, boxes, labels, scores, ["cat", "dog"])
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Make a copy to avoid modifying original
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Generate color palette if not provided
    num_classes = int(labels.max()) + 1 if len(labels) > 0 else 1
    if color_palette is None:
        color_palette = create_color_palette(max(num_classes, 80))
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # Draw each box
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        label = int(label)
        
        # Get color for this class
        color = color_palette[label % len(color_palette)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Create label text
        if class_names is not None and label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = f"class_{label}"
        
        if scores is not None and i < len(scores):
            label_text = f"{label_text}: {scores[i]:.2f}"
        
        # Calculate text size
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text background
        padding = 2
        text_bg = [
            x1,
            y1 - text_height - 2 * padding,
            x1 + text_width + 2 * padding,
            y1,
        ]
        
        # Ensure text is within image bounds
        if text_bg[1] < 0:
            text_bg[1] = y1
            text_bg[3] = y1 + text_height + 2 * padding
        
        draw.rectangle(text_bg, fill=color)
        
        # Draw text
        text_position = (text_bg[0] + padding, text_bg[1] + padding)
        draw.text(text_position, label_text, fill=(255, 255, 255), font=font)
    
    return image


def save_visualization(
    image: Union[Image.Image, np.ndarray],
    save_path: str,
) -> None:
    """Save visualization image to file.
    
    Args:
        image: Image to save (PIL Image or numpy array).
        save_path: Path to save the image.
        
    Note:
        Creates parent directories if they don't exist.
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create output directory if needed
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save image
    image.save(save_path)


def visualize_predictions(
    image_path: str,
    predictions: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    score_threshold: float = 0.0,
) -> Image.Image:
    """High-level function to visualize detection predictions on an image.
    
    Args:
        image_path: Path to input image.
        predictions: Dictionary with keys:
            - 'boxes': Bounding boxes in xyxy format. Shape: (N, 4).
            - 'labels': Class labels. Shape: (N,).
            - 'scores': Confidence scores. Shape: (N,).
        class_names: Optional list of class names.
        output_path: Optional path to save visualization.
        score_threshold: Minimum score to display.
        
    Returns:
        PIL Image with visualized predictions.
        
    Example:
        >>> predictions = {
        ...     'boxes': np.array([[100, 100, 200, 200]]),
        ...     'labels': np.array([0]),
        ...     'scores': np.array([0.95]),
        ... }
        >>> image = visualize_predictions(
        ...     "test.jpg", predictions, ["person"], "output.jpg"
        ... )
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Extract predictions
    boxes = predictions.get('boxes', np.array([]))
    labels = predictions.get('labels', np.array([]))
    scores = predictions.get('scores', None)
    
    # Filter by score threshold
    if scores is not None and len(scores) > 0:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    
    # Draw boxes
    if len(boxes) > 0:
        image = draw_boxes(image, boxes, labels, scores, class_names)
    
    # Save if output path provided
    if output_path is not None:
        save_visualization(image, output_path)
    
    return image


def tensor_to_numpy(tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor (can be on GPU).
        
    Returns:
        Numpy array on CPU.
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return np.array(tensor)


def visualize_batch_predictions(
    images: List[str],
    predictions: List[Dict],
    class_names: Optional[List[str]] = None,
    output_dir: str = "output",
    score_threshold: float = 0.3,
) -> List[Image.Image]:
    """Visualize predictions for a batch of images.
    
    Args:
        images: List of image paths.
        predictions: List of prediction dictionaries.
        class_names: Optional list of class names.
        output_dir: Directory to save visualizations.
        score_threshold: Minimum score to display.
        
    Returns:
        List of PIL Images with visualized predictions.
    """
    results = []
    
    for image_path, pred in zip(images, predictions):
        # Convert tensors to numpy if needed
        pred_np = {
            'boxes': tensor_to_numpy(pred.get('boxes', np.array([]))),
            'labels': tensor_to_numpy(pred.get('labels', np.array([]))),
            'scores': tensor_to_numpy(pred.get('scores', None)) if pred.get('scores') is not None else None,
        }
        
        # Generate output path
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_pred.jpg")
        
        # Visualize
        result = visualize_predictions(
            image_path,
            pred_np,
            class_names,
            output_path,
            score_threshold,
        )
        results.append(result)
    
    return results
