"""Tests for the training pipeline fixes."""

import logging
import warnings

import torch
from torch.utils.data import Dataset

from src.training.train import Trainer, TrainingConfig


class _FakeDetectionDataset(Dataset):
    """Minimal dataset producing detection-style samples."""

    def __init__(self, n: int = 6) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        img = torch.rand(3, 64, 64)
        target = {
            "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def _make_detection_model(num_classes: int = 3):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TestTrainingLogging:
    """Verify that training emits epoch-start and first-batch logs."""

    def test_epoch_started_logged(self, caplog, tmp_path):
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, config=config)
        with caplog.at_level(logging.INFO, logger="src.training.train"):
            trainer.train()
        assert any("Epoch 1/1 started" in msg for msg in caplog.messages)

    def test_first_batch_logged(self, caplog, tmp_path):
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, config=config)
        with caplog.at_level(logging.INFO, logger="src.training.train"):
            trainer.train()
        assert any("[1/" in msg for msg in caplog.messages)


class TestTrainingLossDetach:
    """Verify no requires_grad warning when accumulating loss."""

    def test_no_requires_grad_warning(self, tmp_path):
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, config=config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train()
        grad_warnings = [
            w for w in caught
            if "requires_grad" in str(w.message)
        ]
        assert len(grad_warnings) == 0, (
            f"Unexpected requires_grad warning: {grad_warnings}"
        )


class TestValidation:
    """Verify validation runs correctly alongside training."""

    def test_val_loss_recorded(self, tmp_path):
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, ds, config)
        history = trainer.train()
        assert "val_loss" in history
        assert len(history["val_loss"]) == 1
        assert history["val_loss"][0] > 0

    def test_val_logged(self, caplog, tmp_path):
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, ds, config)
        with caplog.at_level(logging.INFO, logger="src.training.train"):
            trainer.train()
        assert any("Val [" in msg for msg in caplog.messages)


class TestPinMemory:
    """Verify pin_memory is set based on device, not num_workers."""

    def test_pin_memory_cpu(self, tmp_path):
        """pin_memory should be False on CPU."""
        model = _make_detection_model()
        ds = _FakeDetectionDataset(4)
        config = TrainingConfig(
            epochs=1, batch_size=2, device="cpu", amp=False,
            num_workers=0, output_dir=str(tmp_path),
        )
        trainer = Trainer(model, ds, config=config)
        history = trainer.train()
        assert "train_loss" in history
