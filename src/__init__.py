"""
Source modules for Car Retrieval System.
"""

from .utils import (
    CAR_TYPES, NUM_CLASSES,
    get_device, get_classification_transforms,
    load_image, load_image_rgb, load_image_pil,
    crop_image, draw_detection, get_color_for_class,
    create_video_writer, ensure_dir, AverageMeter
)

from .detector import CarDetector, create_detector
from .classifier import CarTypeClassifier, CarClassifierInference, create_classifier
from .pipeline import CarRetrievalPipeline, create_pipeline

__all__ = [
    # Utils
    'CAR_TYPES', 'NUM_CLASSES',
    'get_device', 'get_classification_transforms',
    'load_image', 'load_image_rgb', 'load_image_pil',
    'crop_image', 'draw_detection', 'get_color_for_class',
    'create_video_writer', 'ensure_dir', 'AverageMeter',
    # Detector
    'CarDetector', 'create_detector',
    # Classifier
    'CarTypeClassifier', 'CarClassifierInference', 'create_classifier',
    # Pipeline
    'CarRetrievalPipeline', 'create_pipeline'
]
