# Computer Vision Guide

## Introduction

Computer Vision enables machines to interpret and understand visual information from the world, similar to human vision.

## Core Tasks

### 1. Image Classification

Assigning a label to an entire image.

**Popular Models:**
- ResNet
- VGG
- EfficientNet
- Vision Transformers (ViT)

**Applications:**
- Medical image diagnosis
- Product categorization
- Scene recognition

### 2. Object Detection

Locating and classifying objects in images.

**Popular Models:**
- YOLO (You Only Look Once)
- Faster R-CNN
- SSD (Single Shot Detector)
- RetinaNet

**Output:**
- Bounding boxes
- Class labels
- Confidence scores

### 3. Image Segmentation

Pixel-level classification of images.

**Types:**
- **Semantic Segmentation**: Label each pixel
- **Instance Segmentation**: Separate object instances
- **Panoptic Segmentation**: Combine semantic and instance

**Popular Models:**
- U-Net
- Mask R-CNN
- DeepLab
- SegFormer

### 4. Face Recognition

Identifying or verifying faces in images.

**Steps:**
1. Face detection
2. Face alignment
3. Feature extraction
4. Face matching

**Applications:**
- Security systems
- Photo organization
- Access control

## Image Processing Fundamentals

### Basic Operations

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Convert color space
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize
resized = cv2.resize(image, (224, 224))

# Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)
```

### Data Augmentation

Techniques to increase dataset diversity:
- Random crop
- Horizontal/vertical flip
- Rotation
- Color jittering
- Cutout/Mixup

## Pre-trained Models

### Using Transfer Learning

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

## Datasets

### Common Datasets

- **MNIST**: Handwritten digits (60K images)
- **CIFAR-10/100**: Small images (60K images)
- **ImageNet**: Large-scale dataset (1.4M images)
- **COCO**: Object detection/segmentation (330K images)
- **Pascal VOC**: Object detection (20K images)

## Evaluation Metrics

### Classification
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

### Object Detection
- mAP (mean Average Precision)
- IoU (Intersection over Union)

### Segmentation
- Pixel Accuracy
- IoU/Dice Coefficient
- Mean IoU

## Best Practices

1. **Preprocessing**
   - Normalize images (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Resize to consistent dimensions
   - Apply augmentation during training

2. **Model Selection**
   - Start with pre-trained models
   - Consider speed vs. accuracy tradeoff
   - Use appropriate architecture for task

3. **Training**
   - Use learning rate scheduling
   - Monitor validation metrics
   - Save best model checkpoints

4. **Inference**
   - Batch processing for efficiency
   - Use TensorRT or ONNX for optimization
   - Consider model quantization

## Tools and Libraries

### OpenCV
```python
import cv2
```

### PIL/Pillow
```python
from PIL import Image
```

### torchvision
```python
import torchvision
from torchvision import transforms
```

## Advanced Topics

- **3D Computer Vision**: Point clouds, depth estimation
- **Video Analysis**: Action recognition, tracking
- **GANs for Vision**: Image generation, style transfer
- **Self-Supervised Learning**: SimCLR, BYOL
- **Attention Mechanisms**: CBAM, Squeeze-and-Excitation

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Vision](https://pytorch.org/vision/)
- [Papers with Code - Computer Vision](https://paperswithcode.com/area/computer-vision)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
