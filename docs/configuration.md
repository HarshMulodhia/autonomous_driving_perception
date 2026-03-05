# Configuration Reference

## YOLOv8 KITTI Config (`configs/yolov8_kitti.yaml`)

```yaml
path: data/kitti_yolo
train: images/train
val: images/val

nc: 3
names: ['Car', 'Pedestrian', 'Cyclist']
```

| Field | Description |
|-------|-------------|
| `path` | Root directory of the YOLO-format dataset |
| `train` | Relative path to training images |
| `val` | Relative path to validation images |
| `nc` | Number of classes |
| `names` | Ordered list of class names |

---

## Faster R-CNN Config (via CLI)

Faster R-CNN is configured through command-line arguments and the
`TrainingConfig` dataclass in `src/training/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Number of training epochs |
| `batch_size` | 16 | Mini-batch size |
| `learning_rate` | 0.01 | Initial SGD learning rate |
| `momentum` | 0.9 | SGD momentum |
| `weight_decay` | 0.0005 | L2 regularisation coefficient |
| `lr_step_size` | 7 | Epochs between StepLR decays |
| `lr_gamma` | 0.1 | StepLR multiplicative factor |
| `lr_scheduler_type` | `step` | `step` or `cosine` |
| `grad_clip_norm` | 10.0 | Max gradient L2 norm (0 to disable) |
| `early_stopping_patience` | 0 | Epochs without improvement before stopping (0 to disable) |
| `device` | `cuda` | `cuda` or `cpu` |
| `output_dir` | `outputs` | Directory for checkpoints |
| `log_dir` | `None` | TensorBoard log directory (defaults to `<output_dir>/logs`) |
| `amp` | `True` | Enable automatic mixed precision |
| `num_workers` | 2 | DataLoader worker processes |

### Example YAML (for reference)

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
  grad_clip_norm: 10.0
  early_stopping_patience: 5
  log_dir: outputs/faster_rcnn_kitti/logs

augmentation:
  horizontal_flip: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  random_crop: false
```
