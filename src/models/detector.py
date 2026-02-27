"""Base detector interface and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DetectionResult:
    """Container for model predictions on a single image.

    Attributes:
        boxes: Bounding boxes in ``[x1, y1, x2, y2]`` format, shape ``(N, 4)``.
        scores: Confidence scores per box, shape ``(N,)``.
        labels: Integer class labels per box, shape ``(N,)``.
        class_names: Optional mapping from label id to class name.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: Optional[Dict[int, str]] = field(default=None)

    def filter_by_score(self, threshold: float) -> "DetectionResult":
        """Return a new result keeping only boxes above *threshold*."""
        keep = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[keep],
            scores=self.scores[keep],
            labels=self.labels[keep],
            class_names=self.class_names,
        )

    def __len__(self) -> int:
        return len(self.scores)


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def predict(self, image: np.ndarray, score_threshold: float = 0.5) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image: RGB image as a ``(H, W, 3)`` uint8 numpy array.
            score_threshold: Minimum confidence to keep a detection.

        Returns:
            :class:`DetectionResult` for the image.
        """

    @abstractmethod
    def predict_batch(
        self, images: List[np.ndarray], score_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """Run inference on a batch of images."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model weights from *path*."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model weights to *path*."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device the model is on."""
