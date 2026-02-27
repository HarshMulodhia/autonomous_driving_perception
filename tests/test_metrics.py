"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_ap,
    compute_iou,
    compute_iou_matrix,
    compute_map,
    compute_precision_recall,
    DetectionEvaluator,
)


class TestComputeIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 5.0, 15.0, 15.0])
        inter = 5 * 5
        union = 100 + 100 - inter
        assert compute_iou(a, b) == pytest.approx(inter / union)

    def test_zero_area_box(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([0, 0, 10, 10])
        assert compute_iou(a, b) == pytest.approx(0.0)


class TestComputeIoUMatrix:
    def test_empty(self):
        a = np.zeros((0, 4))
        b = np.array([[0, 0, 10, 10]])
        result = compute_iou_matrix(a, b)
        assert result.shape == (0, 1)

    def test_identity(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        mat = compute_iou_matrix(boxes, boxes)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)


class TestComputeAP:
    def test_perfect_detection(self):
        prec = np.array([1.0, 1.0, 1.0])
        rec = np.array([0.33, 0.66, 1.0])
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_no_detection(self):
        prec = np.array([1.0])
        rec = np.array([0.0])
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(0.0)


class TestComputePrecisionRecall:
    def test_single_image_perfect(self):
        pred_boxes = [np.array([[0, 0, 10, 10]])]
        pred_scores = [np.array([0.9])]
        pred_labels = [np.array([1])]
        gt_boxes = [np.array([[0, 0, 10, 10]])]
        gt_labels = [np.array([1])]

        prec, rec = compute_precision_recall(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, class_id=1,
        )
        assert prec[-1] == pytest.approx(1.0)
        assert rec[-1] == pytest.approx(1.0)

    def test_no_predictions(self):
        pred_boxes = [np.zeros((0, 4))]
        pred_scores = [np.array([])]
        pred_labels = [np.array([])]
        gt_boxes = [np.array([[0, 0, 10, 10]])]
        gt_labels = [np.array([1])]

        prec, rec = compute_precision_recall(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, class_id=1,
        )
        # With no predictions, recall should be 0
        assert rec[-1] == pytest.approx(0.0)


class TestComputeMap:
    def test_single_class(self):
        pred_boxes = [np.array([[0, 0, 10, 10]])]
        pred_scores = [np.array([0.9])]
        pred_labels = [np.array([1])]
        gt_boxes = [np.array([[0, 0, 10, 10]])]
        gt_labels = [np.array([1])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=1,
        )
        assert "mAP" in result
        assert result["mAP"] >= 0.0


class TestDetectionEvaluator:
    def test_basic_workflow(self):
        evaluator = DetectionEvaluator(num_classes=1)
        evaluator.update(
            pred_boxes=np.array([[0, 0, 10, 10]]),
            pred_scores=np.array([0.9]),
            pred_labels=np.array([1]),
            gt_boxes=np.array([[0, 0, 10, 10]]),
            gt_labels=np.array([1]),
        )
        metrics = evaluator.evaluate()
        assert "mAP" in metrics

    def test_reset(self):
        evaluator = DetectionEvaluator(num_classes=1)
        evaluator.update(
            np.array([[0, 0, 10, 10]]),
            np.array([0.9]), np.array([1]),
            np.array([[0, 0, 10, 10]]), np.array([1]),
        )
        evaluator.reset()
        assert len(evaluator._pred_boxes) == 0
