"""Visualization utilities for object detection results."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False


# Default colour palette (BGR for OpenCV, but stored as RGB tuples)
_DEFAULT_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # red
    (0, 0, 255),    # blue
    (255, 255, 0),  # yellow
    (0, 255, 255),  # cyan
    (255, 0, 255),  # magenta
    (128, 255, 0),  # lime
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
    (0, 128, 255),  # sky blue
]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]`` boxes.
        labels: ``(N,)`` integer class labels.
        scores: Optional ``(N,)`` confidence scores.
        class_names: Mapping from label id to name.
        colors: Per-class colour list.
        thickness: Line thickness.

    Returns:
        Annotated image copy.
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required for draw_boxes")

    img = image.copy()
    palette = colors or _DEFAULT_COLORS

    for i, (box, label) in enumerate(zip(boxes, labels)):
        colour = palette[int(label) % len(palette)]
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

        text = class_names.get(int(label), str(int(label))) if class_names else str(int(label))
        if scores is not None:
            text = f"{text} {scores[i]:.2f}"
        cv2.putText(
            img, text, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )

    return img


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    class_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot a precision-recall curve.

    Args:
        precision: Precision values.
        recall: Recall values.
        class_name: Name for the legend.
        save_path: If provided, save the figure to this path.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=class_name or "PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        cm: ``(C, C)`` confusion matrix.
        class_names: List of class names.
        save_path: If provided, save the figure.
        title: Plot title.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib/seaborn is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cmap="Blues",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_predictions(
    image: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: Optional[np.ndarray] = None,
    gt_labels: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    score_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize predictions (and optionally ground truths) on an image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        pred_boxes: Predicted boxes.
        pred_labels: Predicted labels.
        pred_scores: Confidence scores.
        gt_boxes: Optional ground-truth boxes (drawn in white).
        gt_labels: Optional ground-truth labels.
        class_names: Mapping from label id to name.
        score_threshold: Only draw predictions above this confidence.
        save_path: If provided, save the annotated image.

    Returns:
        Annotated image.
    """
    keep = pred_scores >= score_threshold
    vis = draw_boxes(
        image,
        pred_boxes[keep],
        pred_labels[keep],
        pred_scores[keep],
        class_names=class_names,
    )

    if gt_boxes is not None and gt_labels is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)

    if save_path and HAS_CV2:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return vis
