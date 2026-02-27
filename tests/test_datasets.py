"""Tests for dataset modules."""

import numpy as np
import torch

from src.datasets.kitti import parse_kitti_label
from src.datasets.bdd100k import parse_bdd100k_annotations
from src.datasets.augmentation import (
    DetectionAugmentation,
    ToTensor,
    Compose,
    build_transforms,
)


class TestParseKittiLabel:
    def test_valid_label(self, tmp_path):
        label_file = tmp_path / "000001.txt"
        label_file.write_text(
            "Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 "
            "1.57 1.73 4.15 1.00 1.49 13.89 -1.69\n"
            "DontCare -1 -1 -10 0.00 0.00 0.00 0.00 "
            "-1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        ann = parse_kitti_label(str(label_file))
        assert ann["boxes"].shape == (1, 4)
        assert ann["labels"][0] == 1  # Car

    def test_missing_file(self):
        ann = parse_kitti_label("/nonexistent/path.txt")
        assert ann["boxes"].shape == (0, 4)
        assert len(ann["labels"]) == 0


class TestParseBdd100kAnnotations:
    def test_valid_json(self, tmp_path):
        import json
        data = [
            {
                "name": "test.jpg",
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
                    }
                ],
            }
        ]
        ann_file = tmp_path / "det_val.json"
        ann_file.write_text(json.dumps(data))
        annotations = parse_bdd100k_annotations(str(ann_file))
        assert "test.jpg" in annotations
        assert annotations["test.jpg"]["boxes"].shape == (1, 4)


class TestDetectionAugmentation:
    def test_augmentation_call(self):
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
        aug = DetectionAugmentation(horizontal_flip_prob=0.0, color_jitter_prob=0.0)
        out_img, out_target = aug(img, target)
        assert out_target["boxes"].shape == (1, 4)

    def test_empty_boxes(self):
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        target = {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        aug = DetectionAugmentation(horizontal_flip_prob=1.0)
        out_img, out_target = aug(img, target)
        assert out_target["boxes"].shape == (0, 4)


class TestBuildTransforms:
    def test_train_transforms(self):
        t = build_transforms(augment=True)
        assert isinstance(t, Compose)
        assert len(t.transforms) == 2  # augment + to_tensor

    def test_val_transforms(self):
        t = build_transforms(augment=False)
        assert len(t.transforms) == 1  # only to_tensor


class TestToTensor:
    def test_conversion(self):
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0)}
        tensor_img, _ = ToTensor()(img, target)
        assert isinstance(tensor_img, torch.Tensor)
        assert tensor_img.shape[0] == 3
