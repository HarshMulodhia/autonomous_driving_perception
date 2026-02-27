"""BDD100K dataset loader for object detection."""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# BDD100K detection categories
BDD100K_CLASSES = {
    "pedestrian": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "train": 6,
    "motorcycle": 7,
    "bicycle": 8,
    "traffic light": 9,
    "traffic sign": 10,
}

BDD100K_CLASS_NAMES = {v: k for k, v in BDD100K_CLASSES.items()}


def parse_bdd100k_annotations(ann_path: str) -> Dict[str, Dict]:
    """Parse a BDD100K detection annotation JSON file.

    Args:
        ann_path: Path to the BDD100K JSON annotation file.

    Returns:
        Dictionary mapping image filename to annotation data.
    """
    with open(ann_path, "r") as f:
        data = json.load(f)

    annotations: Dict[str, Dict] = {}
    for item in data:
        filename = item["name"]
        boxes = []
        labels = []
        for obj in item.get("labels", []) or []:
            category = obj.get("category", "")
            if category not in BDD100K_CLASSES:
                continue
            box2d = obj.get("box2d")
            if box2d is None:
                continue
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(BDD100K_CLASSES[category])

        annotations[filename] = {
            "boxes": (np.array(boxes, dtype=np.float32)
                      if boxes else np.zeros((0, 4), dtype=np.float32)),
            "labels": (np.array(labels, dtype=np.int64)
                       if labels else np.zeros(0, dtype=np.int64)),
        }

    return annotations


class BDD100KDataset(Dataset):
    """PyTorch Dataset for the BDD100K object detection benchmark.

    Expected directory structure::

        root/
          images/
            train/  *.jpg
            val/    *.jpg
          labels/
            det_20/
              det_train.json
              det_val.json

    Args:
        root: Root directory of the BDD100K dataset.
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        transforms: Optional callable applied to ``(image, target)`` pairs.
    """

    NUM_CLASSES = 10

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        img_dir = self.root / "images" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.img_dir = img_dir
        self.annotations: Dict[str, Dict] = {}

        if split in ("train", "val"):
            ann_file = self.root / "labels" / "det_20" / f"det_{split}.json"
            if not ann_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            self.annotations = parse_bdd100k_annotations(str(ann_file))

        # Build list of image files present on disk
        all_files = sorted(img_dir.glob("*.jpg"))
        self.filenames = [f.name for f in all_files]

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_image(self, idx: int) -> Image.Image:
        path = self.img_dir / self.filenames[idx]
        return Image.open(path).convert("RGB")

    def _load_target(self, idx: int) -> Dict:
        filename = self.filenames[idx]
        default_ann = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
        }
        ann = self.annotations.get(filename, default_ann)
        return {
            "boxes": torch.as_tensor(ann["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(ann["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

    def __getitem__(self, idx: int) -> Tuple:
        image = self._load_image(idx)
        target = self._load_target(idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def class_names(self) -> Dict[int, str]:
        return BDD100K_CLASS_NAMES
