"""Tests for the detector base classes."""

import numpy as np

from src.models.detector import DetectionResult


class TestDetectionResult:
    def test_filter_by_score(self):
        result = DetectionResult(
            boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            scores=np.array([0.9, 0.3]),
            labels=np.array([1, 2]),
        )
        filtered = result.filter_by_score(0.5)
        assert len(filtered) == 1
        assert filtered.labels[0] == 1

    def test_len(self):
        result = DetectionResult(
            boxes=np.array([[0, 0, 10, 10]]),
            scores=np.array([0.9]),
            labels=np.array([1]),
        )
        assert len(result) == 1

    def test_empty_result(self):
        result = DetectionResult(
            boxes=np.zeros((0, 4)),
            scores=np.array([]),
            labels=np.array([]),
        )
        assert len(result) == 0
        filtered = result.filter_by_score(0.5)
        assert len(filtered) == 0
