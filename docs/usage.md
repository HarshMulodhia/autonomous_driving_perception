# Usage Guide

Detailed command-line examples for training, evaluation, and inference.

---

## Training

### Faster R-CNN on KITTI

```bash
python scripts/train.py --dataset kitti --data-root data/kitti --model faster_rcnn --epochs 20 --batch-size 4 --device cpu --output-dir outputs/faster_rcnn_kitti
```

### YOLOv8 on KITTI

```bash
python scripts/train.py --dataset kitti --data-root data/kitti --model yolo --epochs 50 --batch-size 16 --device cpu --output-dir outputs/yolo_kitti
```

### YOLOv8 on BDD100K

```bash
python scripts/train.py --dataset bdd100k --data-root data/bdd100k --model yolo --epochs 50 --batch-size 16 --device cpu --output-dir outputs/yolo_bdd100k
```

### Key CLI flags

| Flag | Description |
|------|-------------|
| `--dataset` | `kitti` or `bdd100k` |
| `--data-root` | Root directory of the dataset |
| `--model` | `faster_rcnn` or `yolo` |
| `--epochs` | Number of training epochs |
| `--batch-size` | Mini-batch size |
| `--lr` | Initial learning rate |
| `--imgsz` | Input image size (YOLOv8) |
| `--device` | `cuda` or `cpu` |
| `--output-dir` | Where to save checkpoints |
| `--log-dir` | TensorBoard log directory (default: `<output-dir>/logs`) |

---

## Evaluation

```bash
python scripts/evaluate.py --dataset kitti --data-root data/kitti --model faster_rcnn --checkpoint outputs/faster_rcnn_kitti/best_model.pth --output-dir outputs/eval_results
```

Results are printed to stdout and saved as `metrics.json` in the output
directory.  Evaluation metrics are also written to TensorBoard event files
in `<output-dir>/logs` (override with `--log-dir`).

---

## Inference

### Video

```bash
python scripts/infer.py --input path/to/video.mp4 --model faster_rcnn --checkpoint outputs/faster_rcnn_kitti/best_model.pth --output path/to/output.mp4
```

### Image directory

```bash
python scripts/infer.py --input path/to/images/ --model faster_rcnn --checkpoint outputs/faster_rcnn_kitti/best_model.pth --output path/to/output/
```

---

## TensorBoard

Training and evaluation metrics are automatically logged to TensorBoard.
Launch the TensorBoard UI to visualise loss curves and evaluation results:

```bash
tensorboard --logdir outputs/faster_rcnn_kitti/logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

**Logged scalars during training:**

| Tag | Description |
|-----|-------------|
| `Loss/train` | Average training loss per epoch |
| `Loss/val` | Average validation loss per epoch |
| `LearningRate` | Current learning rate per epoch |

**Logged scalars during evaluation:**

| Tag | Description |
|-----|-------------|
| `eval/mAP` | Mean Average Precision |
| `eval/AP_class_<id>` | Per-class Average Precision |

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_eda.ipynb` | Dataset statistics and sample visualisations |
| `notebooks/02_evaluation.ipynb` | Model evaluation, per-class analysis, and failure modes |
