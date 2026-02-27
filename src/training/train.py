"""Training pipeline for object detection models."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for the data loader.
        learning_rate: Initial learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularisation weight.
        lr_step_size: Epoch interval for LR decay.
        lr_gamma: Multiplicative LR decay factor.
        device: Device string (``'cuda'`` or ``'cpu'``).
        output_dir: Directory to save checkpoints and logs.
        amp: Whether to use automatic mixed precision.
        num_workers: DataLoader worker count.
    """

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 7
    lr_gamma: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"
    amp: bool = True
    num_workers: int = 2


def _collate_fn(batch: list) -> tuple:
    """Custom collate that keeps images and targets as lists."""
    return tuple(zip(*batch))


class Trainer:
    """Generic Faster R-CNN / torchvision-style trainer.

    Args:
        model: A torchvision detection model.
        train_dataset: Training :class:`Dataset`.
        val_dataset: Validation :class:`Dataset` (optional).
        config: :class:`TrainingConfig`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )
        self.scaler = torch.amp.GradScaler(
            enabled=self.config.amp,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            Dictionary mapping metric names to per-epoch values.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate_fn,
        )

        history: Dict[str, List[float]] = {"train_loss": []}
        best_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            loss = self._train_one_epoch(train_loader, epoch)
            history["train_loss"].append(loss)
            self.lr_scheduler.step()

            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint(
                    os.path.join(self.config.output_dir, "best_model.pth")
                )

            self._save_checkpoint(
                os.path.join(self.config.output_dir, "last_model.pth")
            )

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int
    ) -> float:
        self.model.train()
        total_loss = 0.0
        device = self.config.device

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.config.amp and "cuda" in self.config.device,
            ):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(losses)

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss

    def _save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
