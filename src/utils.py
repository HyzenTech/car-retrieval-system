"""
Utility functions for the Car Retrieval System.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Car type classes
CAR_TYPES = [
    'crossover',
    'hatchback',
    'mpv',
    'offroad',
    'pickup',
    'sedan',
    'truck',
    'van'
]

NUM_CLASSES = len(CAR_TYPES)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_classification_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for classification.
    
    Args:
        train: Whether to use training augmentations
        img_size: Target image size
    
    Returns:
        torchvision transforms composition
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_image(image_path: str) -> np.ndarray:
    """Load an image from path as BGR numpy array."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image from path as RGB numpy array."""
    image = load_image(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_image_pil(image_path: str) -> Image.Image:
    """Load an image as PIL Image."""
    return Image.open(image_path).convert('RGB')


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.1
) -> np.ndarray:
    """
    Crop a region from an image with optional padding.
    
    Args:
        image: Input image (H, W, C)
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Relative padding to add around the crop
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return image[y1:y2, x1:x2]


def draw_detection(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a detection box with label on an image.
    
    Args:
        image: Input image (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        label: Class label
        confidence: Detection confidence
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with drawn detection
    """
    image = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    text = f"{label}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    
    # Draw label background
    cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    
    # Draw text
    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), text_thickness)
    
    return image


def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Get a consistent color for each car class."""
    colors = {
        'crossover': (255, 128, 0),    # Orange
        'hatchback': (0, 255, 128),    # Cyan-green
        'mpv': (128, 0, 255),          # Purple
        'offroad': (0, 128, 255),      # Light blue
        'pickup': (255, 255, 0),       # Yellow
        'sedan': (0, 255, 0),          # Green
        'truck': (0, 0, 255),          # Red
        'van': (255, 0, 255)           # Magenta
    }
    return colors.get(class_name.lower(), (255, 255, 255))


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create a video writer.
    
    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: (width, height)
        codec: Video codec
    
    Returns:
        cv2.VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
