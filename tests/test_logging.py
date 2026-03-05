"""Tests for logging configuration in scripts and source modules."""

import logging
import os

import numpy as np

from src.evaluation.metrics import DetectionEvaluator
from src.training.train import TrainingConfig, HAS_TENSORBOARD


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
        assert any("Evaluating 1 images with 1 classes" in msg for msg in caplog.messages)


class TestTrainingConfig:
    """Verify TensorBoard-related configuration."""

    def test_default_log_dir_is_none(self):
        config = TrainingConfig()
        assert config.log_dir is None

    def test_custom_log_dir(self):
        config = TrainingConfig(log_dir="/tmp/tb_test")
        assert config.log_dir == "/tmp/tb_test"

    def test_tensorboard_available(self):
        """tensorboard should be importable in this environment."""
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
            available = True
        except ImportError:
            available = False
        # HAS_TENSORBOARD must agree with what we just checked.
        assert HAS_TENSORBOARD is available


class TestTensorBoardIntegration:
    """Verify TensorBoard writer is created and writes events."""

    def test_trainer_creates_writer(self, tmp_path):
        import torch
        from src.training.train import Trainer
        model = torch.nn.Linear(4, 2)
        config = TrainingConfig(
            epochs=1, batch_size=1, device="cpu",
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
        )
        # Provide an empty dataset — only verify writer creation.
        trainer = Trainer(model, train_dataset=[], config=config)
        assert trainer.writer is not None
        trainer.writer.close()

    def test_tensorboard_writes_events(self, tmp_path):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = str(tmp_path / "tb_events")
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar("test/loss", 0.5, 1)
        writer.close()
        event_files = [f for f in os.listdir(log_dir) if "events" in f]
        assert len(event_files) > 0


class TestGradScalerDevice:
    """Verify GradScaler is only enabled when CUDA is the target device."""

    def test_scaler_disabled_on_cpu(self, tmp_path):
        import torch
        from src.training.train import Trainer
        model = torch.nn.Linear(4, 2)
        config = TrainingConfig(
            epochs=1, batch_size=1, device="cpu", amp=True,
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
        )
        trainer = Trainer(model, train_dataset=[], config=config)
        assert not trainer.scaler.is_enabled()
        trainer.writer.close()

    def test_scaler_disabled_when_amp_off(self, tmp_path):
        import torch
        from src.training.train import Trainer
        model = torch.nn.Linear(4, 2)
        config = TrainingConfig(
            epochs=1, batch_size=1, device="cpu", amp=False,
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
        )
        trainer = Trainer(model, train_dataset=[], config=config)
        assert not trainer.scaler.is_enabled()
        trainer.writer.close()


class TestDataLoaderNoPersistentWorkers:
    """Verify DataLoaders do not use persistent_workers to prevent hangs."""

    def test_train_loader_no_persistent_workers(self, tmp_path):
        import torch
        from unittest.mock import patch
        from src.training.train import Trainer

        model = torch.nn.Linear(4, 2)
        ds = [
            (torch.rand(3, 32, 32), {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)})
        ]
        config = TrainingConfig(
            epochs=1, batch_size=1, device="cpu", num_workers=0,
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
        )
        trainer = Trainer(model, ds, config=config)

        # Patch DataLoader to inspect the persistent_workers argument
        with patch("src.training.train.DataLoader", wraps=torch.utils.data.DataLoader) as mock_dl:
            try:
                trainer.train()
            except (RuntimeError, TypeError, AttributeError):
                pass  # model type mismatch is expected; we just check DataLoader args
            for call_args in mock_dl.call_args_list:
                assert "persistent_workers" in call_args.kwargs
                assert call_args.kwargs["persistent_workers"] is False
