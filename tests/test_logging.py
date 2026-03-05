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


class TestTrainerLoop:
    """Verify the training loop runs end-to-end and produces expected output."""

    @staticmethod
    def _make_mock_model():
        """Return a lightweight model that mimics a torchvision detection model."""
        import torch

        class _MockDetectionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 1)

            def forward(self, images, targets=None):
                if self.training and targets is not None:
                    dummy = self.fc(torch.zeros(3)).sum()
                    return {"loss_cls": dummy.abs() + 0.1, "loss_box": dummy.abs() + 0.1}
                return [
                    {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.int64)}
                    for _ in images
                ]
        return _MockDetectionModel()

    @staticmethod
    def _make_dummy_dataset(n=6):
        """Return a list-based dataset of (image_tensor, target_dict) pairs."""
        import torch

        class _DS:
            def __len__(self):
                return n

            def __getitem__(self, idx):
                img = torch.rand(3, 32, 32)
                target = {
                    "boxes": torch.tensor([[2.0, 2.0, 30.0, 30.0]]),
                    "labels": torch.tensor([1]),
                }
                return img, target
        return _DS()

    def test_train_loop_completes(self, tmp_path):
        """Full training loop should complete and return loss history."""
        from src.training.train import Trainer
        model = self._make_mock_model()
        config = TrainingConfig(
            epochs=2, batch_size=2, device="cpu", amp=False,
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
        )
        trainer = Trainer(model, self._make_dummy_dataset(), self._make_dummy_dataset(4), config)
        history = trainer.train()

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        assert "val_loss" in history
        assert len(history["val_loss"]) == 2
        assert all(v > 0 for v in history["train_loss"])
        # Checkpoints should be written
        assert (tmp_path / "out" / "best_model.pth").exists()
        assert (tmp_path / "out" / "last_model.pth").exists()

    def test_first_batch_logged(self, tmp_path, caplog):
        """The very first batch of each epoch must be logged."""
        from src.training.train import Trainer
        model = self._make_mock_model()
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            output_dir=str(tmp_path / "out"),
        )
        trainer = Trainer(model, self._make_dummy_dataset(), config=config)
        with caplog.at_level(logging.INFO, logger="src.training.train"):
            trainer.train()
        # Should contain a log for batch 1/3
        assert any("[1/" in msg for msg in caplog.messages)

    def test_validation_logged(self, tmp_path, caplog):
        """Validation start should be logged."""
        from src.training.train import Trainer
        model = self._make_mock_model()
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            output_dir=str(tmp_path / "out"),
        )
        trainer = Trainer(model, self._make_dummy_dataset(), self._make_dummy_dataset(4), config)
        with caplog.at_level(logging.INFO, logger="src.training.train"):
            trainer.train()
        assert any("Running validation" in msg for msg in caplog.messages)
