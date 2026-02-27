"""YOLOv8 detector wrapper using Ultralytics."""

from typing import Dict, List, Optional

import numpy as np
import torch

from .detector import BaseDetector, DetectionResult

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False


class YOLODetector(BaseDetector):
    """YOLOv8 detector via the Ultralytics library.

    Args:
        model_path: Path to a ``.pt`` weights file or a model name
            (e.g. ``'yolov8m.pt'``).
        device_name: ``'cuda'`` or ``'cpu'``.
        class_names: Optional mapping from label id to name.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        device_name: str = "cpu",
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if not HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install it with: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self._device = torch.device(device_name)
        self._class_names = class_names

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(
        self, image: np.ndarray, score_threshold: float = 0.5
    ) -> DetectionResult:
        return self.predict_batch([image], score_threshold)[0]

    def predict_batch(
        self, images: List[np.ndarray], score_threshold: float = 0.5
    ) -> List[DetectionResult]:
        results_list: List[DetectionResult] = []
        for image in images:
            outputs = self.model(
                image, conf=score_threshold,
                device=str(self._device), verbose=False,
            )
            result = outputs[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            results_list.append(DetectionResult(
                boxes=boxes,
                scores=scores,
                labels=labels,
                class_names=self._class_names,
            ))
        return results_list

    def load_checkpoint(self, path: str) -> None:
        self.model = YOLO(path)

    def save_checkpoint(self, path: str) -> None:
        # Ultralytics manages its own checkpoint format
        if hasattr(self.model, "save"):
            self.model.save(path)
