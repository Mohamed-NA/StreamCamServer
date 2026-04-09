from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def require_child_text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"Missing <{tag}> in XML element <{element.tag}>")
    return child.text.strip()


def parse_xml_det(xml_path: Path, class_to_idx: dict[str, int]) -> tuple[list[list[float]], list[int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = require_child_text(obj, "name")
        label_idx = class_to_idx.get(name)
        if label_idx is None:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"Missing <bndbox> in {xml_path}")
        x1 = float(require_child_text(bnd, "xmin"))
        y1 = float(require_child_text(bnd, "ymin"))
        x2 = float(require_child_text(bnd, "xmax"))
        y2 = float(require_child_text(bnd, "ymax"))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(label_idx)
    return boxes, labels


class FaceMaskDataset(Dataset):
    def __init__(self, image_paths: list[Path], dataset_root: Path, classes: list[str], augment: bool = False):
        self.image_paths = image_paths
        self.dataset_root = dataset_root
        self.classes = classes
        self.augment = augment
        self.annotations_dir = dataset_root / "annotations"
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, tuple[list[list[float]], list[int]]]:
        cache: dict[str, tuple[list[list[float]], list[int]]] = {}
        for img_path in self.image_paths:
            xml_path = self.annotations_dir / f"{img_path.stem}.xml"
            cache[img_path.stem] = parse_xml_det(xml_path, self.class_to_idx)
        return cache

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        boxes, labels = self.annotations[img_path.stem]
        boxes = [box.copy() for box in boxes]
        labels = list(labels)

        if self.augment and random.random() > 0.5:
            img_rgb = img_rgb[:, ::-1, :].copy()
            width = img_rgb.shape[1]
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]

        img_tensor = TF.to_tensor(img_rgb)
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))

