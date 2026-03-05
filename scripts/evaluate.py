#!/usr/bin/env python
"""Evaluation entry-point script.

Usage::

    python scripts/evaluate.py \\
        --dataset kitti --data-root data/kitti \\
        --model faster_rcnn \\
        --checkpoint outputs/faster_rcnn_kitti/best_model.pth \\
        --output-dir outputs/eval_results
"""

import argparse
import json
import logging
import os
import sys

# Ensure the project root is on the path so ``src.*`` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an object detector")
    parser.add_argument("--dataset", choices=["kitti", "bdd100k"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", choices=["faster_rcnn", "yolo"], default="yolo")
    parser.add_argument("--checkpoint", required=True, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="outputs/eval_results")
    parser.add_argument("--log-dir", default=None, help="TensorBoard log directory for evaluation metrics")
    return parser.parse_args()


def _log_to_tensorboard(log_dir: str, model_type: str, metrics_dict: dict) -> None:
    """Write evaluation metrics to TensorBoard if available."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard is not installed; skipping TensorBoard logging.")
        return

    writer = SummaryWriter(log_dir=log_dir)
    if model_type == "yolo":
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"eval/{key}", value, 0)
    elif model_type == "faster_rcnn":
        writer.add_scalar("eval/mAP", metrics_dict["mAP"], 0)
        for cls_id, ap in metrics_dict.get("AP_per_class", {}).items():
            writer.add_scalar(f"eval/AP_class_{cls_id}", ap, 0)
    writer.close()
    logger.info("Evaluation metrics logged to TensorBoard at %s", log_dir)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "Starting evaluation: model=%s, dataset=%s, checkpoint=%s",
        args.model, args.dataset, args.checkpoint,
    )

    if args.model == "yolo":
        from ultralytics import YOLO

        model = YOLO(args.checkpoint)
        data_cfg = "configs/yolov8_kitti.yaml" if args.dataset == "kitti" else None
        if data_cfg is None:
            raise ValueError("YOLO eval config for BDD100K not provided.")

        metrics = model.val(data=data_cfg, imgsz=args.imgsz, conf=args.conf)
        results = {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "precision": float(metrics.box.p.mean()),
            "recall": float(metrics.box.r.mean()),
        }
        print(json.dumps(results, indent=2))
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.model == "faster_rcnn":
        import numpy as np
        from src.datasets.augmentation import build_transforms
        from src.models.faster_rcnn import FasterRCNNDetector
        from src.evaluation.metrics import DetectionEvaluator

        if args.dataset == "kitti":
            from src.datasets.kitti import KITTIDataset, KITTI_CLASS_NAMES
            val_ds = KITTIDataset(args.data_root, split="val", transforms=build_transforms(augment=False))
            num_classes = KITTIDataset.NUM_CLASSES
            class_names = KITTI_CLASS_NAMES
        else:
            from src.datasets.bdd100k import BDD100KDataset, BDD100K_CLASS_NAMES
            val_ds = BDD100KDataset(args.data_root, split="val", transforms=build_transforms(augment=False))
            num_classes = BDD100KDataset.NUM_CLASSES
            class_names = BDD100K_CLASS_NAMES

        detector = FasterRCNNDetector(
            num_classes=num_classes + 1, device_name="cpu",
            class_names=class_names,
        )
        detector.load_checkpoint(args.checkpoint)

        evaluator = DetectionEvaluator(num_classes=num_classes, class_names=class_names)

        for i in range(len(val_ds)):
            image, target = val_ds[i]
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            result = detector.predict(img_np, score_threshold=args.conf)
            evaluator.update(
                result.boxes, result.scores, result.labels,
                target["boxes"].numpy(), target["labels"].numpy(),
            )

        metrics = evaluator.evaluate()
        print(json.dumps(metrics, indent=2, default=str))
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    # Log evaluation metrics to TensorBoard
    tb_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    if args.model == "yolo":
        _log_to_tensorboard(tb_dir, "yolo", results)
    elif args.model == "faster_rcnn":
        _log_to_tensorboard(tb_dir, "faster_rcnn", metrics)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
