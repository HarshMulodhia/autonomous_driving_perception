"""Training pipeline for object detection models."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    HAS_TENSORBOARD = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for the data loader.
        learning_rate: Initial learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularisation weight (λ for the penalty term).
        lr_step_size: Epoch interval for StepLR decay.
        lr_gamma: Multiplicative LR decay factor (γ).
        lr_scheduler_type: ``'step'`` for StepLR or ``'cosine'`` for
            CosineAnnealingLR.
        grad_clip_norm: Max L2 norm for gradient clipping (0 to disable).
        early_stopping_patience: Stop after this many epochs without
            validation improvement (0 to disable).
        device: Device string (``'cuda'`` or ``'cpu'``).
        output_dir: Directory to save checkpoints and logs.
        log_dir: Directory for TensorBoard event files. When *None*,
            defaults to ``<output_dir>/logs``.
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
    lr_scheduler_type: str = "step"
    grad_clip_norm: float = 10.0
    early_stopping_patience: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"
    log_dir: Optional[str] = None
    amp: bool = True
    num_workers: int = 2


def _collate_fn(batch: list) -> tuple:
    """Custom collate that keeps images and targets as lists."""
    return tuple(zip(*batch))


class Trainer:
    """Generic Faster R-CNN / torchvision-style trainer.

    Supports gradient clipping, validation loss tracking, early stopping,
    and cosine-annealing LR scheduling in addition to the default StepLR.

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

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        if self.config.lr_scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs,
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )

        self.scaler = torch.amp.GradScaler(
            enabled=self.config.amp,
        )

        # TensorBoard writer
        self.writer: Optional["SummaryWriter"] = None
        if HAS_TENSORBOARD:
            tb_dir = self.config.log_dir or os.path.join(
                self.config.output_dir, "logs",
            )
            self.writer = SummaryWriter(log_dir=tb_dir)
            logger.info("TensorBoard logging to %s", tb_dir)
        else:
            logger.warning(
                "tensorboard is not installed; skipping TensorBoard logging."
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

        val_loader: Optional[DataLoader] = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=_collate_fn,
            )

        history: Dict[str, List[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            self.lr_scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            log_msg = f"Epoch {epoch}/{self.config.epochs}  train_loss={train_loss:.4f}  lr={lr:.6f}"

            # Validation
            monitor_loss = train_loss
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                monitor_loss = val_loss
                log_msg += f"  val_loss={val_loss:.4f}"

            logger.info(log_msg)

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("LearningRate", lr, epoch)
                if val_loader is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)

            # Checkpoint best model
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
                self._save_checkpoint(
                    os.path.join(self.config.output_dir, "best_model.pth")
                )
            else:
                patience_counter += 1

            self._save_checkpoint(
                os.path.join(self.config.output_dir, "last_model.pth")
            )

            # Early stopping
            if (
                self.config.early_stopping_patience > 0
                and patience_counter >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after %d consecutive epochs of no improvement.",
                    patience_counter,
                )
                break

        if self.writer is not None:
            self.writer.close()

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

            # Gradient clipping for training stability
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(losses)

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Compute average loss on the validation set."""
        self.model.train()  # detection models need train mode for loss computation
        total_loss = 0.0
        device = self.config.device

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += float(losses)

        return total_loss / max(len(loader), 1)

    def _save_checkpoint(self, path: str) -> None:
        logger.debug("Saving checkpoint to %s", path)
        torch.save(self.model.state_dict(), path)
