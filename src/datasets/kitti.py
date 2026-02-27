"""KITTI dataset loader for object detection."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# KITTI object classes mapped to contiguous IDs
KITTI_CLASSES = {
    "Car": 1,
    "Van": 2,
    "Truck": 3,
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Cyclist": 6,
    "Tram": 7,
    "Misc": 8,
    "DontCare": 0,
}

KITTI_CLASS_NAMES = {v: k for k, v in KITTI_CLASSES.items() if v > 0}


def parse_kitti_label(label_path: str) -> Dict:
    """Parse a KITTI label file into boxes and class labels.

    Args:
        label_path: Path to the KITTI label .txt file.

    Returns:
        Dictionary with 'boxes' (N,4) array and 'labels' (N,) array.
    """
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros(0, dtype=np.int64)}

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            class_name = parts[0]
            if class_name not in KITTI_CLASSES or KITTI_CLASSES[class_name] == 0:
                continue
            # KITTI format: left, top, right, bottom (x1, y1, x2, y2)
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(KITTI_CLASSES[class_name])

    if len(boxes) == 0:
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros(0, dtype=np.int64)}

    return {
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


class KITTIDataset(Dataset):
    """PyTorch Dataset for the KITTI object detection benchmark.

    Expected directory structure::

        root/
          training/
            image_2/   *.png
            label_2/   *.txt
          testing/
            image_2/   *.png

    Args:
        root: Root directory of the KITTI dataset.
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        transforms: Optional callable applied to ``(image, target)`` pairs.
        val_fraction: Fraction of training data to use for validation.
    """

    NUM_CLASSES = 8

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        val_fraction: float = 0.1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        if split in ("train", "val"):
            img_dir = self.root / "training" / "image_2"
            label_dir = self.root / "training" / "label_2"
        else:
            img_dir = self.root / "testing" / "image_2"
            label_dir = None

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        all_ids = sorted(p.stem for p in img_dir.glob("*.png"))
        if not all_ids:
            all_ids = sorted(p.stem for p in img_dir.glob("*.jpg"))

        n_val = max(1, int(len(all_ids) * val_fraction))
        if split == "train":
            self.ids = all_ids[n_val:]
        elif split == "val":
            self.ids = all_ids[:n_val]
        else:
            self.ids = all_ids

        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, idx: int) -> Image.Image:
        img_id = self.ids[idx]
        for ext in (".png", ".jpg"):
            path = self.img_dir / (img_id + ext)
            if path.exists():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"Image not found for id: {img_id}")

    def _load_target(self, idx: int) -> Dict:
        img_id = self.ids[idx]
        if self.label_dir is None:
            return {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}

        label_path = self.label_dir / (img_id + ".txt")
        ann = parse_kitti_label(str(label_path))

        target = {
            "boxes": torch.as_tensor(ann["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(ann["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return target

    def __getitem__(self, idx: int) -> Tuple:
        image = self._load_image(idx)
        target = self._load_target(idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def class_names(self) -> Dict[int, str]:
        return KITTI_CLASS_NAMES
