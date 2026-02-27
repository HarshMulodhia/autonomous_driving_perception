"""Object detection models for autonomous driving."""

from .detector import BaseDetector, DetectionResult
from .faster_rcnn import FasterRCNNDetector
from .yolo import YOLODetector

__all__ = ["BaseDetector", "DetectionResult", "FasterRCNNDetector", "YOLODetector"]
