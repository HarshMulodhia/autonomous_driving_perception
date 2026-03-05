#!/usr/bin/env python
"""Inference entry-point script for video and image directories.

Usage::

    python scripts/infer.py \\
        --input path/to/video.mp4 \\
        --model faster_rcnn \\
        --checkpoint outputs/faster_rcnn_kitti/best_model.pth \\
        --output path/to/output.mp4

    python scripts/infer.py \\
        --input path/to/images/ \\
        --model yolo \\
        --checkpoint results/yolov8_kitti/weights/best.pt \\
        --output path/to/output/
"""

import argparse
import logging
import os
import sys

# Ensure the project root is on the path so ``src.*`` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on video or images")
    parser.add_argument("--input", required=True, help="Video file or image directory")
    parser.add_argument("--model", choices=["faster_rcnn", "yolo"], default="yolo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output video path or directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "Starting inference: model=%s, input=%s, device=%s",
        args.model, args.input, args.device,
    )

    if args.model == "yolo":
        from src.models.yolo import YOLODetector
        detector = YOLODetector(model_path=args.checkpoint, device_name=args.device)
    else:
        from src.models.faster_rcnn import FasterRCNNDetector
        detector = FasterRCNNDetector(device_name=args.device)
        detector.load_checkpoint(args.checkpoint)

    from src.inference.pipeline import InferencePipeline
    pipeline = InferencePipeline(detector, score_threshold=args.conf)

    if os.path.isdir(args.input):
        results = pipeline.run_on_directory(args.input, output_dir=args.output)
        logger.info("Processed %d images -> %s", len(results), args.output)
    else:
        results = pipeline.run_on_video(args.input, output_path=args.output)
        logger.info("Processed %d frames -> %s", len(results), args.output)


if __name__ == "__main__":
    main()
