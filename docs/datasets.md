# Dataset Guide

Detailed instructions for downloading and preparing datasets used in this
project.

---

## KITTI Object Detection

Download the KITTI object detection dataset from
[http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/).

### Required files

* **Left colour images** of the object dataset (`image_2`)
* **Training labels** of the object dataset (`label_2`)

### Expected directory layout

```
data/kitti/
├── training/
│   ├── image_2/         # Left colour images (*.png)
│   └── label_2/         # Per-image annotations (*.txt)
└── testing/
    └── image_2/         # Test images (no labels)
```

### Annotation format

Each line in a KITTI label file contains 15 space-separated fields:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | type | Object class (`Car`, `Pedestrian`, `Cyclist`, …) |
| 1 | truncated | Float 0 – 1 indicating truncation |
| 2 | occluded | Integer 0 – 3 for occlusion level |
| 3 | alpha | Observation angle |
| 4–7 | bbox | 2-D bounding box: left, top, right, bottom |
| 8–10 | dimensions | 3-D object dimensions (height, width, length) |
| 11–13 | location | 3-D location (x, y, z) in camera coords |
| 14 | rotation_y | Rotation around Y-axis |

### Classes (8 foreground)

`Car` · `Van` · `Truck` · `Pedestrian` · `Person_sitting` · `Cyclist` ·
`Tram` · `Misc`

`DontCare` annotations are ignored during training and evaluation.

### Automatic train / val split

By default, `KITTIDataset` reserves the first 10 % of sorted image IDs for
validation. Override via the `val_fraction` parameter.

---

## BDD100K

Download BDD100K from
[https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/).

### Required files

* 100K images split into `train/`, `val/`, `test/`
* Detection labels (`det_20/` JSON files)

### Expected directory layout

```
data/bdd100k/
├── images/
│   ├── train/           # Training images (*.jpg)
│   ├── val/             # Validation images (*.jpg)
│   └── test/            # Test images (*.jpg)
└── labels/
    └── det_20/
        ├── det_train.json
        └── det_val.json
```

### Annotation format

JSON array where each element corresponds to one image and contains a
`labels` list. Each label has a `category` string and a `box2d` object with
`x1`, `y1`, `x2`, `y2` fields.

### Classes (10 foreground)

`pedestrian` · `rider` · `car` · `truck` · `bus` · `train` · `motorcycle` ·
`bicycle` · `traffic light` · `traffic sign`
