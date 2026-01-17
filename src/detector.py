"""
Car Object Detector using YOLOv8.

This module provides car detection functionality using the Ultralytics YOLOv8 model
pre-trained on the COCO dataset.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    raise


class CarDetector:
    """
    Car detector using YOLOv8 pre-trained on COCO dataset.
    
    The COCO dataset class index for 'car' is 2.
    """
    
    COCO_CAR_CLASS_ID = 2  # Car class in COCO dataset
    COCO_VEHICLE_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}  # Vehicle classes
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        detect_all_vehicles: bool = False
    ):
        """
        Initialize the car detector.
        
        Args:
            model_name: YOLOv8 model variant ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            detect_all_vehicles: If True, detect cars, buses, and trucks
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.detect_all_vehicles = detect_all_vehicles
        
        # Set target classes
        if detect_all_vehicles:
            self.target_classes = list(self.COCO_VEHICLE_CLASSES.keys())
        else:
            self.target_classes = [self.COCO_CAR_CLASS_ID]
    
    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = False
    ) -> Dict[str, Any]:
        """
        Detect cars in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            return_crops: Whether to return cropped car images
        
        Returns:
            Dictionary containing:
                - boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
                - scores: List of confidence scores
                - labels: List of class labels
                - crops: List of cropped images (if return_crops=True)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.target_classes,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = []
        scores = []
        labels = []
        crops = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                scores.append(confidence)
                labels.append(self.COCO_VEHICLE_CLASSES.get(class_id, 'vehicle'))
                
                # Crop if requested
                if return_crops:
                    crop = self._crop_box(image, (int(x1), int(y1), int(x2), int(y2)))
                    crops.append(crop)
        
        result = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'num_detections': len(boxes)
        }
        
        if return_crops:
            result['crops'] = crops
        
        return result
    
    def _crop_box(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.05
    ) -> np.ndarray:
        """
        Crop a bounding box from an image with padding.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Relative padding to add
        
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add small padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return image[y1:y2, x1:x2].copy()
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Detect cars in a batch of images.
        
        Args:
            images: List of input images
        
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
    
    def detect_from_path(self, image_path: str, return_crops: bool = False) -> Dict[str, Any]:
        """
        Detect cars from an image file path.
        
        Args:
            image_path: Path to the image file
            return_crops: Whether to return cropped car images
        
        Returns:
            Detection results dictionary
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.detect(image, return_crops=return_crops)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'target_classes': self.target_classes,
            'detect_all_vehicles': self.detect_all_vehicles
        }


def create_detector(
    model_size: str = 'n',
    confidence: float = 0.5,
    detect_all_vehicles: bool = False
) -> CarDetector:
    """
    Factory function to create a car detector.
    
    Args:
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        confidence: Confidence threshold
        detect_all_vehicles: Whether to detect all vehicle types
    
    Returns:
        CarDetector instance
    """
    model_name = f'yolov8{model_size}.pt'
    return CarDetector(
        model_name=model_name,
        confidence_threshold=confidence,
        detect_all_vehicles=detect_all_vehicles
    )


if __name__ == '__main__':
    # Quick test
    import sys
    
    detector = create_detector(model_size='n', confidence=0.5)
    print(f"Detector created: {detector.get_model_info()}")
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        results = detector.detect_from_path(image_path)
        print(f"Found {results['num_detections']} cars")
        for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
            print(f"  {i+1}. {label}: {score:.3f} at {box}")
