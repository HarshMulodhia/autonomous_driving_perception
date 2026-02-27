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
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an object detector")
    parser.add_argument("--dataset", choices=["kitti", "bdd100k"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", choices=["faster_rcnn", "yolo"], default="yolo")
    parser.add_argument("--checkpoint", required=True, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="outputs/eval_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
