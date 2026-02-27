"""Faster R-CNN detector wrapper using torchvision."""

from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .detector import BaseDetector, DetectionResult


class FasterRCNNDetector(BaseDetector):
    """Faster R-CNN ResNet-50 FPN detector.

    Args:
        num_classes: Number of classes **including** background.
        pretrained: Whether to load ImageNet-pretrained backbone weights.
        device_name: ``'cuda'`` or ``'cpu'``.
        class_names: Optional mapping from label id to name.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        device_name: str = "cpu",
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        weights = "DEFAULT" if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes,
        )
        self._device = torch.device(device_name)
        self.model.to(self._device)
        self.model.eval()
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

    @torch.no_grad()
    def predict_batch(
        self, images: List[np.ndarray], score_threshold: float = 0.5
    ) -> List[DetectionResult]:
        self.model.eval()
        tensors = [
            torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(self._device)
            for img in images
        ]
        outputs = self.model(tensors)

        results: List[DetectionResult] = []
        for out in outputs:
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            keep = scores >= score_threshold
            results.append(DetectionResult(
                boxes=boxes[keep],
                scores=scores[keep],
                labels=labels[keep],
                class_names=self._class_names,
            ))
        return results

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
