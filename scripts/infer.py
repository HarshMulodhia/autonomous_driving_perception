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
import os


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
        print(f"Processed {len(results)} images → {args.output}")
    else:
        results = pipeline.run_on_video(args.input, output_path=args.output)
        print(f"Processed {len(results)} frames → {args.output}")


if __name__ == "__main__":
    main()
