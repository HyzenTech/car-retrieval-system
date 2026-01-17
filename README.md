# 🚗 Car Retrieval System

An end-to-end deep learning system for **detecting and classifying Indonesian cars** by type. This project combines YOLOv8 object detection with a ResNet50-based classifier to identify car types including crossover, hatchback, MPV, offroad, pickup, sedan, truck, and van.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Video Inference](#video-inference)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## 🎯 Overview

The Car Retrieval System is designed to:
1. **Detect** multiple car instances in images and videos using YOLOv8
2. **Classify** each detected car into one of 8 Indonesian car types
3. Process real-time video feeds with visualization

### Key Features
- YOLOv8-based vehicle detection (pre-trained on COCO)
- ResNet50-based car type classification (custom trained)
- Integrated pipeline for end-to-end inference
- Video processing with real-time statistics overlay
- Comprehensive evaluation metrics and visualizations

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Input Image/   │────▶│   YOLOv8     │────▶│  Car Detections │
│     Video       │     │  Detector    │     │  (Bounding Box) │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Annotated      │◀────│   ResNet50   │◀────│   Crop Regions  │
│    Output       │     │  Classifier  │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/car-retrieval-system.git
cd car-retrieval-system

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pillow>=9.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

## 📊 Dataset

The dataset contains Indonesian car images categorized into 8 classes:

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Crossover | 2,125 | 265 | 266 |
| Hatchback | 2,090 | 309 | 280 |
| MPV | 2,469 | 342 | 313 |
| Offroad | 1,919 | 240 | 240 |
| Pickup | 744 | 153 | 123 |
| Sedan | 1,961 | 284 | 255 |
| Truck | 659 | 83 | 82 |
| Van | 1,575 | 197 | 197 |
| **Total** | **13,542** | **1,873** | **1,756** |

### Dataset Structure

```
dataset/dataset/
├── train/
│   ├── crossover/
│   ├── hatchback/
│   ├── mpv/
│   ├── offroad/
│   ├── pickup/
│   ├── sedan/
│   ├── truck/
│   └── van/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

## 🏋️ Training

### Train the Classifier

```bash
python scripts/train_classifier.py \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --exp_name resnet50_v1
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--dropout` | 0.5 | Dropout rate |
| `--pretrained` | True | Use ImageNet pretrained weights |
| `--exp_name` | None | Experiment name |

### Training Features
- Data augmentation (RandomCrop, ColorJitter, Rotation, etc.)
- Cosine annealing learning rate schedule
- Label smoothing (0.1)
- Early stopping with best model checkpointing
- TensorBoard logging

## 📈 Evaluation

### Run Evaluation

```bash
python scripts/evaluate.py \
    --model models/classifier/resnet50_v1/best_model.pth \
    --output_dir outputs/evaluation
```

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.33% |
| **Weighted Precision** | 86.53% |
| **Weighted Recall** | 86.33% |
| **Weighted F1-Score** | 86.34% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Crossover | 0.85 | 0.81 | 0.83 |
| Hatchback | 0.88 | 0.83 | 0.85 |
| MPV | 0.80 | 0.90 | 0.85 |
| Offroad | 0.86 | 0.85 | 0.86 |
| Pickup | 0.93 | 0.89 | 0.91 |
| Sedan | 0.87 | 0.87 | 0.87 |
| Truck | 0.92 | 0.88 | 0.90 |
| Van | 0.90 | 0.90 | 0.90 |

## 🎬 Video Inference

### Process Video

```bash
python video_inference.py \
    --input traffic_test.mp4 \
    --output outputs/output_video.mp4 \
    --classifier_model models/classifier/resnet50_v1/best_model.pth
```

### Video Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | traffic_test.mp4 | Input video path |
| `--output` | outputs/output_video.mp4 | Output video path |
| `--classifier_model` | None | Path to classifier weights |
| `--detector_confidence` | 0.5 | Detection confidence threshold |
| `--skip_frames` | 1 | Process every N frames |

### Video Processing Results

| Metric | Value |
|--------|-------|
| Frames Processed | 2,960 |
| Average FPS | 13.6 |
| Inference Time | 73.7ms |
| Total Detections | 14,107 |

## 📁 Project Structure

```
car-retrieval-system/
├── dataset/dataset/          # Training dataset
├── models/
│   ├── detector/            # YOLOv8 weights
│   └── classifier/          # ResNet50 weights
├── outputs/
│   ├── evaluation/          # Metrics, plots
│   └── output_video.mp4     # Annotated video
├── scripts/
│   ├── train_classifier.py  # Training script
│   └── evaluate.py          # Evaluation script
├── src/
│   ├── __init__.py
│   ├── classifier.py        # ResNet50 classifier
│   ├── detector.py          # YOLOv8 detector
│   ├── pipeline.py          # Integrated pipeline
│   └── utils.py             # Utility functions
├── video_inference.py       # Video processing
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 💻 Usage

### Python API

```python
from src.pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline(
    classifier_model_path='models/classifier/resnet50_v1/best_model.pth',
    detector_confidence=0.5
)

# Process image
import cv2
image = cv2.imread('test_image.jpg')
result = pipeline.process(image, return_visualization=True)

# Access results
print(f"Detected {result['num_cars']} cars")
for det in result['detections']:
    print(f"  - {det['car_type']}: {det['car_type_confidence']:.2f}")

# Save visualization
cv2.imwrite('output.jpg', result['visualization'])
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) for deep learning framework
- Indonesian car image dataset providers

## 📧 Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Made with ❤️ for Indonesian Car Detection**
