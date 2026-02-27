"""Detection evaluation metrics: IoU, AP, mAP, precision/recall."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]`` array.
        box_b: ``[x1, y1, x2, y2]`` array.

    Returns:
        IoU value in ``[0, 1]``.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: ``(N, 4)`` array.
        boxes_b: ``(M, 4)`` array.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def compute_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision-recall curve for a single class.

    Args:
        pred_boxes: Per-image predicted boxes list.
        pred_scores: Per-image confidence scores list.
        pred_labels: Per-image predicted labels list.
        gt_boxes: Per-image ground-truth boxes list.
        gt_labels: Per-image ground-truth labels list.
        class_id: Class to evaluate.
        iou_threshold: IoU threshold for a true positive.

    Returns:
        ``(precision, recall)`` arrays sorted by decreasing confidence.
    """
    all_scores: List[float] = []
    all_tp: List[int] = []
    total_gt = 0

    for i in range(len(pred_boxes)):
        gt_mask = gt_labels[i] == class_id
        gt_cls = gt_boxes[i][gt_mask]
        total_gt += len(gt_cls)

        pred_mask = pred_labels[i] == class_id
        pb = pred_boxes[i][pred_mask]
        ps = pred_scores[i][pred_mask]

        if len(pb) == 0:
            continue

        matched = np.zeros(len(gt_cls), dtype=bool)

        order = np.argsort(-ps)
        for idx in order:
            all_scores.append(float(ps[idx]))
            if len(gt_cls) == 0:
                all_tp.append(0)
                continue

            ious = np.array([compute_iou(pb[idx], g) for g in gt_cls])
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not matched[best]:
                all_tp.append(1)
                matched[best] = True
            else:
                all_tp.append(0)

    if total_gt == 0 or len(all_scores) == 0:
        return np.array([1.0]), np.array([0.0])

    order = np.argsort(-np.array(all_scores))
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(1 - np.array(all_tp)[order])

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / total_gt

    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using the all-point interpolation method.

    Args:
        precision: Precision values.
        recall: Recall values.

    Returns:
        AP value.
    """
    # Prepend sentinel values
    rec = np.concatenate(([0.0], recall, [1.0]))
    prec = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # Find points where recall changes
    idx = np.where(rec[1:] != rec[:-1])[0]
    ap = float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))
    return ap


def compute_map(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mean Average Precision across all classes.

    Args:
        pred_boxes: Per-image predicted boxes.
        pred_scores: Per-image confidence scores.
        pred_labels: Per-image predicted labels.
        gt_boxes: Per-image ground-truth boxes.
        gt_labels: Per-image ground-truth labels.
        num_classes: Number of classes (excluding background at index 0).
        iou_threshold: IoU threshold.

    Returns:
        Dictionary with ``'mAP'``, ``'AP_per_class'``, ``'precision'``,
        ``'recall'`` keys.
    """
    aps = {}
    precisions = {}
    recalls = {}

    for cls_id in range(1, num_classes + 1):
        prec, rec = compute_precision_recall(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, cls_id, iou_threshold,
        )
        ap = compute_ap(prec, rec)
        aps[cls_id] = ap
        precisions[cls_id] = float(prec[-1]) if len(prec) > 0 else 0.0
        recalls[cls_id] = float(rec[-1]) if len(rec) > 0 else 0.0

    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0

    return {
        "mAP": mean_ap,
        "AP_per_class": aps,
        "precision": precisions,
        "recall": recalls,
    }


class DetectionEvaluator:
    """Accumulates predictions and ground truths, then computes metrics.

    Args:
        num_classes: Number of object classes (excluding background).
        iou_threshold: IoU threshold for matching.
        class_names: Optional mapping from class id to name.
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}
        self.reset()

    def reset(self) -> None:
        """Clear accumulated data."""
        self._pred_boxes: List[np.ndarray] = []
        self._pred_scores: List[np.ndarray] = []
        self._pred_labels: List[np.ndarray] = []
        self._gt_boxes: List[np.ndarray] = []
        self._gt_labels: List[np.ndarray] = []

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> None:
        """Add a single image's predictions and ground truths."""
        self._pred_boxes.append(np.asarray(pred_boxes))
        self._pred_scores.append(np.asarray(pred_scores))
        self._pred_labels.append(np.asarray(pred_labels))
        self._gt_boxes.append(np.asarray(gt_boxes))
        self._gt_labels.append(np.asarray(gt_labels))

    def evaluate(self) -> Dict[str, float]:
        """Compute mAP and per-class metrics."""
        return compute_map(
            self._pred_boxes,
            self._pred_scores,
            self._pred_labels,
            self._gt_boxes,
            self._gt_labels,
            self.num_classes,
            self.iou_threshold,
        )
