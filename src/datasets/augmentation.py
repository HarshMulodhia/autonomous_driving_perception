"""Data augmentation transforms for object detection."""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class DetectionAugmentation:
    """Augmentation pipeline for object detection tasks.

    Applies transforms consistently to both the image and bounding boxes.

    Args:
        horizontal_flip_prob: Probability of random horizontal flip.
        color_jitter_prob: Probability of applying colour jitter.
        brightness: Max brightness adjustment factor.
        contrast: Max contrast adjustment factor.
        saturation: Max saturation adjustment factor.
        hue: Max hue adjustment factor.
        min_size: Minimum dimension for resizing (None = no resize).
        max_size: Maximum dimension for resizing (None = no cap).
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        """Apply augmentation to an image and its targets.

        Args:
            image: PIL Image.
            target: Dict containing ``'boxes'`` (N,4) and ``'labels'`` (N,) tensors.

        Returns:
            Augmented ``(image, target)`` pair.
        """
        image, target = self._maybe_horizontal_flip(image, target)
        image = self._maybe_color_jitter(image)
        if self.min_size is not None:
            image, target = self._resize(image, target)
        return image, target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_horizontal_flip(
        self, image: Image.Image, target: Dict
    ) -> Tuple[Image.Image, Dict]:
        if random.random() < self.horizontal_flip_prob:
            w, _ = image.size
            image = TF.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target = {**target, "boxes": boxes}
        return image, target

    def _maybe_color_jitter(self, image: Image.Image) -> Image.Image:
        if random.random() < self.color_jitter_prob:
            factors = [
                (self.brightness, TF.adjust_brightness),
                (self.contrast, TF.adjust_contrast),
                (self.saturation, TF.adjust_saturation),
            ]
            # Shuffle order for more variety
            random.shuffle(factors)
            for factor, fn in factors:
                if factor > 0:
                    f = random.uniform(max(0.0, 1.0 - factor), 1.0 + factor)
                    image = fn(image, f)
            if self.hue > 0:
                h = random.uniform(-self.hue, self.hue)
                image = TF.adjust_hue(image, h)
        return image

    def _resize(
        self, image: Image.Image, target: Dict
    ) -> Tuple[Image.Image, Dict]:
        w, h = image.size
        scale = self.min_size / min(h, w)  # type: ignore[operator]
        if self.max_size is not None:
            scale = min(scale, self.max_size / max(h, w))

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = TF.resize(image, [new_h, new_w])

        if target["boxes"].numel() > 0:
            boxes = target["boxes"] * scale
            target = {**target, "boxes": boxes}

        return image, target


class ToTensor:
    """Convert a PIL Image to a float32 torch Tensor in ``[0, 1]``."""

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return TF.to_tensor(image), target


class Compose:
    """Compose multiple detection transforms together."""

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def build_transforms(
    augment: bool = True,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.5,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> Compose:
    """Build a standard transform pipeline.

    Args:
        augment: Whether to include augmentation (False for validation/test).
        horizontal_flip_prob: Probability of horizontal flip (ignored if not augmenting).
        color_jitter_prob: Probability of colour jitter (ignored if not augmenting).
        min_size: Optional minimum image dimension.
        max_size: Optional maximum image dimension.

    Returns:
        A :class:`Compose` transform.
    """
    transforms: List = []
    if augment:
        transforms.append(
            DetectionAugmentation(
                horizontal_flip_prob=horizontal_flip_prob,
                color_jitter_prob=color_jitter_prob,
                min_size=min_size,
                max_size=max_size,
            )
        )
    transforms.append(ToTensor())
    return Compose(transforms)
