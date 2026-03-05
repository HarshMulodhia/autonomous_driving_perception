# Autonomous Driving Perception

An object detection pipeline for autonomous driving, trained on the
[KITTI](http://www.cvlibs.net/datasets/kitti/) and
[BDD100K](https://bdd-data.berkeley.edu/) benchmarks using **Faster R-CNN**
and **YOLOv8**.

## Project Structure

```
autonomous_driving_perception/
├── src/
│   ├── datasets/        # Dataset loaders (KITTI, BDD100K) and augmentation
│   ├── models/          # Object detection models (Faster R-CNN, YOLOv8)
│   ├── training/        # Training pipeline with gradient clipping, early stopping
│   ├── evaluation/      # Metrics (mAP, precision, recall) and visualization
│   └── inference/       # Video and image inference pipeline
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for EDA and evaluation analysis
├── scripts/             # Training, evaluation, and inference entry points
├── configs/             # Model configuration files
├── docs/                # Extended documentation and mathematical foundations
├── environment.yml      # Conda environment specification (primary)
└── requirements.txt     # Pip-compatible dependency mirror
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/HarshMulodhia/autonomous_driving_perception.git
cd autonomous_driving_perception
conda env create -f environment.yml
conda activate autonomous_driving_perception
pip install -e .

# Train Faster R-CNN on KITTI
python scripts/train.py --dataset kitti --data-root data/kitti \
    --model faster_rcnn --epochs 20 --batch-size 4

# Evaluate
python scripts/evaluate.py --dataset kitti --data-root data/kitti \
    --model faster_rcnn --checkpoint outputs/best_model.pth

# Run inference on a video
python scripts/infer.py --input video.mp4 --model faster_rcnn \
    --checkpoint outputs/best_model.pth --output output.mp4

# View training curves in TensorBoard
tensorboard --logdir outputs/logs
```

## Results

| Model | Dataset | mAP@0.5 |
|-------|---------|---------|
| Faster R-CNN (ResNet-50 FPN) | KITTI | 0.64 |
| YOLOv8m | BDD100K | 0.56 |

## Testing

```bash
pytest tests/ -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [Mathematical Foundations](docs/mathematical_foundations.md) | Theory and equations behind every algorithm |
| [Dataset Guide](docs/datasets.md) | Download instructions and annotation formats |
| [Configuration Reference](docs/configuration.md) | All hyperparameters and config files |
| [Usage Guide](docs/usage.md) | Full CLI examples and notebook descriptions |
| [Technical Report](docs/technical_report.md) | Architecture overview and result analysis |

## License

MIT License