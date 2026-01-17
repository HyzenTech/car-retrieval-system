"""
Integrated Car Retrieval Pipeline.

Combines object detection (YOLOv8) and car type classification (ResNet50)
into a unified pipeline for end-to-end car retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

import numpy as np
import cv2
from PIL import Image

import torch

# Import our modules
from detector import CarDetector, create_detector
from classifier import CarClassifierInference, CAR_TYPES, create_classifier
from utils import draw_detection, get_color_for_class


class CarRetrievalPipeline:
    """
    End-to-end car retrieval pipeline.
    
    Workflow:
        1. Detect cars in the input image using YOLOv8
        2. Crop each detection
        3. Classify each crop using ResNet50
        4. Return combined detection + classification results
    """
    
    def __init__(
        self,
        detector: Optional[CarDetector] = None,
        classifier: Optional[CarClassifierInference] = None,
        classifier_model_path: Optional[str] = None,
        detector_confidence: float = 0.5,
        classifier_confidence: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            detector: Pre-initialized CarDetector (optional)
            classifier: Pre-initialized CarClassifierInference (optional)
            classifier_model_path: Path to classifier weights
            detector_confidence: Detection confidence threshold
            classifier_confidence: Classification confidence threshold
            device: Device for inference
        """
        # Initialize detector
        if detector is not None:
            self.detector = detector
        else:
            self.detector = create_detector(
                model_size='n',
                confidence=detector_confidence,
                detect_all_vehicles=True
            )
        
        # Initialize classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = create_classifier(
                model_path=classifier_model_path,
                device=device
            )
        
        self.classifier_confidence = classifier_confidence
        
        print(f"Pipeline initialized:")
        print(f"  Detector: YOLOv8 (conf={detector_confidence})")
        print(f"  Classifier: ResNet50 (model={'loaded' if classifier_model_path else 'pretrained only'})")
    
    def process(
        self,
        image: np.ndarray,
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            image: Input image (BGR format)
            return_visualization: Whether to return annotated image
        
        Returns:
            Dictionary containing:
                - detections: List of detection results
                - num_cars: Total number of cars detected
                - processing_time: Time taken for processing
                - visualization: Annotated image (if requested)
        """
        start_time = time.time()
        
        # Step 1: Detect cars
        detection_results = self.detector.detect(image, return_crops=True)
        
        # Step 2: Classify each detection
        detections = []
        
        for i, (box, score, label, crop) in enumerate(zip(
            detection_results['boxes'],
            detection_results['scores'],
            detection_results['labels'],
            detection_results.get('crops', [])
        )):
            detection = {
                'id': i,
                'box': box,
                'detection_score': score,
                'detection_label': label
            }
            
            # Classify if we have a crop
            if crop is not None and crop.size > 0:
                try:
                    classification = self.classifier.predict(crop)
                    detection['car_type'] = classification['class']
                    detection['car_type_confidence'] = classification['confidence']
                    detection['car_type_probabilities'] = classification['probabilities']
                except Exception as e:
                    print(f"Classification error for detection {i}: {e}")
                    detection['car_type'] = 'unknown'
                    detection['car_type_confidence'] = 0.0
            else:
                detection['car_type'] = 'unknown'
                detection['car_type_confidence'] = 0.0
            
            detections.append(detection)
        
        processing_time = time.time() - start_time
        
        result = {
            'detections': detections,
            'num_cars': len(detections),
            'processing_time': processing_time
        }
        
        # Create visualization if requested
        if return_visualization:
            vis_image = self._visualize(image, detections)
            result['visualization'] = vis_image
        
        return result
    
    def _visualize(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create visualization with detection boxes and labels."""
        vis_image = image.copy()
        
        for det in detections:
            box = det['box']
            car_type = det.get('car_type', 'unknown')
            confidence = det.get('car_type_confidence', 0.0)
            
            # Get color for this car type
            color = get_color_for_class(car_type)
            
            # Draw detection
            label = f"{car_type}"
            vis_image = draw_detection(
                vis_image, box, label, confidence, color=color, thickness=2
            )
        
        return vis_image
    
    def process_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Process multiple images."""
        return [self.process(img) for img in images]


def create_pipeline(
    classifier_model_path: Optional[str] = None,
    detector_confidence: float = 0.5,
    device: Optional[str] = None
) -> CarRetrievalPipeline:
    """
    Factory function to create the pipeline.
    
    Args:
        classifier_model_path: Path to trained classifier weights
        detector_confidence: Detection confidence threshold
        device: Device for inference
    
    Returns:
        CarRetrievalPipeline instance
    """
    return CarRetrievalPipeline(
        classifier_model_path=classifier_model_path,
        detector_confidence=detector_confidence,
        device=device
    )


if __name__ == '__main__':
    import sys
    
    # Quick test
    print("Testing Car Retrieval Pipeline...")
    
    # Create pipeline (no classifier weights - just testing)
    pipeline = create_pipeline()
    
    # Test with a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            result = pipeline.process(image, return_visualization=True)
            
            print(f"\nResults:")
            print(f"  Cars detected: {result['num_cars']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            
            for det in result['detections']:
                print(f"  - {det['car_type']} (conf={det['car_type_confidence']:.2f}) at {det['box']}")
            
            # Save visualization
            if 'visualization' in result:
                output_path = 'test_output.jpg'
                cv2.imwrite(output_path, result['visualization'])
                print(f"\nVisualization saved to: {output_path}")
        else:
            print(f"Could not load image: {image_path}")
    else:
        print("Pipeline created successfully!")
        print("Usage: python pipeline.py <image_path>")
