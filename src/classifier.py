"""
Car Type Classifier using ResNet50.

This module provides Indonesian car type classification using a ResNet50 backbone
with a custom classification head.
"""

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Car type classes for Indonesian cars
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


class CarTypeClassifier(nn.Module):
    """
    Car type classifier using ResNet50 backbone with custom classification head.
    
    Architecture:
        - ResNet50 backbone (pretrained on ImageNet)
        - Global Average Pooling
        - Dropout
        - FC layer (2048 -> 512)
        - ReLU + Dropout
        - FC layer (512 -> num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = CAR_TYPES
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        
        backbone = models.resnet50(weights=weights)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize custom layers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of custom layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        return 2048


class CarClassifierInference:
    """
    Wrapper class for inference with the car type classifier.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        img_size: int = 224
    ):
        """
        Initialize the inference wrapper.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            img_size: Input image size
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.img_size = img_size
        
        # Initialize model
        self.model = CarTypeClassifier(num_classes=NUM_CLASSES, pretrained=True)
        
        # Load weights if provided
        if model_path is not None:
            self.load_weights(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_weights(self, model_path: str):
        """Load trained weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded weights from {model_path}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image as numpy array (BGR or RGB)
        
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict car type for a single image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        tensor = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get top prediction
            max_prob, pred_idx = torch.max(probs, dim=1)
            pred_class = CAR_TYPES[pred_idx.item()]
            confidence = max_prob.item()
            
            # Get all class probabilities
            all_probs = {
                CAR_TYPES[i]: probs[0, i].item()
                for i in range(NUM_CLASSES)
            }
        
        return {
            'class': pred_class,
            'class_idx': pred_idx.item(),
            'confidence': confidence,
            'probabilities': all_probs
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Predict car types for a batch of images.
        
        Args:
            images: List of input images
        
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Predict car type from an image file path.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Prediction results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.predict(image)


def create_classifier(
    model_path: Optional[str] = None,
    device: Optional[str] = None
) -> CarClassifierInference:
    """
    Factory function to create a classifier inference wrapper.
    
    Args:
        model_path: Path to trained model weights
        device: Device for inference
    
    Returns:
        CarClassifierInference instance
    """
    return CarClassifierInference(model_path=model_path, device=device)


if __name__ == '__main__':
    # Quick test
    import sys
    
    # Create model (no weights loaded - will use random initialization)
    model = CarTypeClassifier(num_classes=NUM_CLASSES, pretrained=True)
    print(f"Model created with {NUM_CLASSES} classes: {CAR_TYPES}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
