# Autonomous Driving Perception

An object detection pipeline for autonomous driving perception, trained on KITTI and BDD100K datasets.

## Overview

This project implements a complete object detection pipeline including:
- **Dataset handling**: KITTI and BDD100K loaders with data augmentation
- **Models**: Faster R-CNN and YOLOv8 implementations
- **Training**: Full training pipeline with hyperparameter optimization
- **Evaluation**: mAP, precision, recall metrics with failure analysis
- **Inference**: Video sequence inference pipeline

## Project Structure

```
autonomous_driving_perception/
├── src/
│   ├── datasets/        # Dataset loaders (KITTI, BDD100K) and augmentation
│   ├── models/          # Object detection models (Faster R-CNN, YOLOv8)
│   ├── training/        # Training pipeline
│   ├── evaluation/      # Metrics and visualization
│   └── inference/       # Inference pipeline
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for analysis
├── scripts/             # Training and evaluation scripts
├── configs/             # Configuration files
├── environment.yml      # Conda environment specification (primary)
└── requirements.txt     # Pip-compatible mirror of dependencies (for pip-only workflows)
```

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-enabled GPU (recommended)

### Installation

```bash
git clone https://github.com/HarshMulodhia/autonomous_driving_perception.git
cd autonomous_driving_perception

# Create and activate the conda environment
conda env create -f environment.yml
conda activate autonomous_driving_perception

# Install the package in editable mode
pip install -e .
```

To update the environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

To deactivate or remove the environment:

```bash
conda deactivate
conda env remove -n autonomous_driving_perception
```

## Datasets

### KITTI

Download the KITTI object detection dataset from [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/):

```
data/kitti/
├── training/
│   ├── image_2/         # Left color images
│   └── label_2/         # Annotations
└── testing/
    └── image_2/
```

### BDD100K

Download BDD100K from [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/):

```
data/bdd100k/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── det_20/
    │   ├── det_train.json
    │   └── det_val.json
```

## Usage

### Training

Train Faster R-CNN on KITTI:
```bash
python scripts/train.py \
    --dataset kitti \
    --data-root data/kitti \
    --model faster_rcnn \
    --epochs 20 \
    --batch-size 4 \
    --output-dir outputs/faster_rcnn_kitti
```

Train YOLOv8 on BDD100K:
```bash
python scripts/train.py \
    --dataset bdd100k \
    --data-root data/bdd100k \
    --model yolo \
    --epochs 50 \
    --batch-size 16 \
    --output-dir outputs/yolo_bdd100k
```

### Evaluation

```bash
python scripts/evaluate.py \
    --dataset kitti \
    --data-root data/kitti \
    --model faster_rcnn \
    --checkpoint outputs/faster_rcnn_kitti/best_model.pth \
    --output-dir outputs/eval_results
```

### Inference

Run inference on a video:
```bash
python scripts/infer.py \
    --input path/to/video.mp4 \
    --model faster_rcnn \
    --checkpoint outputs/faster_rcnn_kitti/best_model.pth \
    --output path/to/output.mp4
```

Run inference on an image directory:
```bash
python scripts/infer.py \
    --input path/to/images/ \
    --model faster_rcnn \
    --checkpoint outputs/faster_rcnn_kitti/best_model.pth \
    --output path/to/output/
```

## Results

### KITTI Dataset (Faster R-CNN ResNet-50)

| Class       | AP@0.5 |
|-------------|--------|
| Car         | 0.72   |
| Pedestrian  | 0.58   |
| Cyclist     | 0.61   |
| **mAP@0.5** | **0.64** |

### BDD100K Dataset (YOLOv8m)

| Class       | AP@0.5 |
|-------------|--------|
| Car         | 0.68   |
| Truck       | 0.55   |
| Bus         | 0.52   |
| Pedestrian  | 0.47   |
| **mAP@0.5** | **0.56** |

## Evaluation Metrics

- **mAP (mean Average Precision)**: Primary detection metric at IoU threshold 0.5
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Configuration

Configuration files are in `configs/`. Example config for Faster R-CNN:

```yaml
model:
  name: faster_rcnn
  backbone: resnet50
  num_classes: 9
  pretrained: true

training:
  epochs: 20
  batch_size: 4
  learning_rate: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  lr_scheduler:
    type: step
    step_size: 7
    gamma: 0.1

augmentation:
  horizontal_flip: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  random_crop: false
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src
```

## Notebooks

- `notebooks/week1_dataset_exploration.ipynb`: Dataset statistics and visualization
- `notebooks/week3_evaluation_analysis.ipynb`: Model evaluation and failure analysis

## Key Technologies

- **PyTorch** & **torchvision**: Deep learning framework and model zoo
- **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Image and video processing
- **KITTI / BDD100K**: Autonomous driving datasets
- **Jupyter Notebooks**: Interactive analysis

## License

MIT License