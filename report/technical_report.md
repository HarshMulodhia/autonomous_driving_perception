# Perception for Autonomous Driving — Technical Report

## 1. Problem Statement

Object detection for autonomous driving using KITTI/BDD100K datasets.
The goal is to detect and localise road objects (cars, pedestrians, cyclists)
in real-time from camera images — a safety-critical task for self-driving
vehicles.

## 2. Dataset Overview

| Dataset | Training Images | Classes | Annotation Format |
|---------|----------------|---------|-------------------|
| KITTI   | 7,481          | Car, Pedestrian, Cyclist (+ Van, Truck, Tram, …) | Per-image `.txt` with 2-D bounding boxes |
| BDD100K | 70,000         | 10 classes (car, bus, pedestrian, rider, truck, motor, bicycle, traffic sign, traffic light, train) | JSON with 2-D bounding boxes |

## 3. Model Architecture

### Primary: YOLOv8-medium (Single-stage, Anchor-free)
- Backbone: CSPDarknet
- Neck: PANet with C2f modules
- Head: Decoupled head for classification and regression
- Input size: 640×640
- Parameters: ~25.9 M

### Baseline: Faster R-CNN ResNet50-FPN (Two-stage)
- Backbone: ResNet-50 with Feature Pyramid Network
- Region Proposal Network (RPN) → RoI Pooling → Classification
- Parameters: ~41.8 M

## 4. Results

| Model       | mAP@0.5 | Precision | Recall | FPS  |
|-------------|---------|-----------|--------|------|
| YOLOv8-m    | 0.XX    | 0.XX      | 0.XX   | XX   |
| Faster RCNN | 0.XX    | 0.XX      | 0.XX   | XX   |

### Per-class AP (KITTI)

| Class      | YOLOv8-m AP@0.5 | Faster RCNN AP@0.5 |
|------------|-----------------|---------------------|
| Car        | 0.XX            | 0.XX                |
| Pedestrian | 0.XX            | 0.XX                |
| Cyclist    | 0.XX            | 0.XX                |

## 5. Failure Mode Analysis

- **Small objects (<32×32 pixels):** Degraded recall (~0.XX) — pooling/downsampling loses fine detail.
- **Night / low-light images:** False positive rate increases by XX% due to low contrast.
- **Occluded pedestrians:** Recall drops to XX% when >50% of the body is occluded.
- **Distant cyclists:** Confusion with pedestrians at long range (IoU-based matching fails).

## 6. Hyperparameter Choices

| Hyperparameter | KITTI | BDD100K |
|----------------|-------|---------|
| Batch size     | 16    | 8–12   |
| Learning rate  | 0.01  | 0.005  |
| Epochs         | 50    | 30–50  |
| Image size     | 640   | 640    |
| Optimizer      | SGD + momentum | AdamW |

## 7. Future Improvements

1. **Multi-sensor fusion** — Combine camera RGB with LiDAR point clouds for depth-aware detection.
2. **Vision Transformer backbone** — Evaluate DETR / RT-DETR for end-to-end detection without NMS.
3. **TensorRT / ONNX quantization** — INT8 inference for edge deployment on NVIDIA Jetson.
4. **Temporal modelling** — Track objects across frames to stabilize detections and reduce flickering.
5. **Domain adaptation** — Fine-tune on target driving conditions (rain, snow, night) with minimal labels.
