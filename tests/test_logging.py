"""Tests for logging configuration in scripts and source modules."""

import logging

import numpy as np

from src.evaluation.metrics import DetectionEvaluator


class TestModuleLoggers:
    """Verify that key modules define a logger."""

    def test_training_logger(self):
        logger = logging.getLogger("src.training.train")
        assert logger is not None
        assert logger.name == "src.training.train"

    def test_faster_rcnn_logger(self):
        logger = logging.getLogger("src.models.faster_rcnn")
        assert logger is not None
        assert logger.name == "src.models.faster_rcnn"

    def test_kitti_logger(self):
        logger = logging.getLogger("src.datasets.kitti")
        assert logger is not None
        assert logger.name == "src.datasets.kitti"

    def test_bdd100k_logger(self):
        logger = logging.getLogger("src.datasets.bdd100k")
        assert logger is not None
        assert logger.name == "src.datasets.bdd100k"

    def test_metrics_logger(self):
        logger = logging.getLogger("src.evaluation.metrics")
        assert logger is not None
        assert logger.name == "src.evaluation.metrics"

    def test_pipeline_logger(self):
        logger = logging.getLogger("src.inference.pipeline")
        assert logger is not None
        assert logger.name == "src.inference.pipeline"


class TestEvaluatorLogging:
    """Verify that evaluation logging emits expected messages."""

    def test_evaluate_logs_info(self, caplog):
        evaluator = DetectionEvaluator(num_classes=1)
        evaluator.update(
            pred_boxes=np.array([[0, 0, 10, 10]]),
            pred_scores=np.array([0.9]),
            pred_labels=np.array([1]),
            gt_boxes=np.array([[0, 0, 10, 10]]),
            gt_labels=np.array([1]),
        )
        with caplog.at_level(logging.INFO, logger="src.evaluation.metrics"):
            evaluator.evaluate()
        assert any("Evaluating" in msg for msg in caplog.messages)
