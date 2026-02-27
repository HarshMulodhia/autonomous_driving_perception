"""Evaluation metrics and visualization for object detection."""

from .metrics import (
    compute_iou,
    compute_ap,
    compute_map,
    compute_precision_recall,
    DetectionEvaluator,
)
from .visualization import (
    draw_boxes,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    visualize_predictions,
)

__all__ = [
    "compute_iou",
    "compute_ap",
    "compute_map",
    "compute_precision_recall",
    "DetectionEvaluator",
    "draw_boxes",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
    "visualize_predictions",
]
