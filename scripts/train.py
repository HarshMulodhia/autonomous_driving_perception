#!/usr/bin/env python
"""Training entry-point script.

Usage examples::

    # Faster R-CNN on KITTI
    python scripts/train.py \\
        --dataset kitti --data-root data/kitti \\
        --model faster_rcnn --epochs 20 --batch-size 4 \\
        --output-dir outputs/faster_rcnn_kitti

    # YOLOv8 on KITTI (uses ultralytics)
    python scripts/train.py \\
        --dataset kitti --data-root data/kitti \\
        --model yolo --epochs 50 --batch-size 16 \\
        --output-dir outputs/yolo_kitti
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--dataset", choices=["kitti", "bdd100k"], required=True)
    parser.add_argument("--data-root", required=True, help="Root directory of the dataset")
    parser.add_argument("--model", choices=["faster_rcnn", "yolo"], default="yolo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics is required for YOLOv8. pip install ultralytics")

        model = YOLO("yolov8m.pt")
        data_cfg = "configs/yolov8_kitti.yaml" if args.dataset == "kitti" else None
        if data_cfg is None:
            sys.exit("YOLO training config for BDD100K not yet provided.")

        model.train(
            data=data_cfg,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            amp=True,
            lr0=args.lr,
            augment=True,
            project=args.output_dir,
            name=f"yolov8_{args.dataset}",
        )

    elif args.model == "faster_rcnn":
        from src.datasets.augmentation import build_transforms
        from src.models.faster_rcnn import FasterRCNNDetector
        from src.training.train import Trainer, TrainingConfig

        if args.dataset == "kitti":
            from src.datasets.kitti import KITTIDataset
            train_ds = KITTIDataset(args.data_root, split="train", transforms=build_transforms(augment=True))
            val_ds = KITTIDataset(args.data_root, split="val", transforms=build_transforms(augment=False))
            num_classes = KITTIDataset.NUM_CLASSES + 1  # +background
        else:
            from src.datasets.bdd100k import BDD100KDataset
            train_ds = BDD100KDataset(args.data_root, split="train", transforms=build_transforms(augment=True))
            val_ds = BDD100KDataset(args.data_root, split="val", transforms=build_transforms(augment=False))
            num_classes = BDD100KDataset.NUM_CLASSES + 1

        detector = FasterRCNNDetector(num_classes=num_classes, device_name=args.device)
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            output_dir=args.output_dir,
        )
        trainer = Trainer(detector.model, train_ds, val_ds, config)
        trainer.train()

    print("Training complete.")


if __name__ == "__main__":
    main()
