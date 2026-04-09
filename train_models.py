from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, Image as HFImage
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, ViTForImageClassification
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from faster_rcnn_data import FaceMaskDataset, collate_fn


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "models.json").exists():
            return candidate
    raise RuntimeError("Could not locate the StreamCamServer project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXPORT_DIR = NOTEBOOK_DIR / "exports"
CACHE_DIR = NOTEBOOK_DIR / "cache"
CHECKPOINT_DIR = NOTEBOOK_DIR / "checkpoints"
for path in (DATA_DIR, EXPORT_DIR, CACHE_DIR, CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)


def require_child_text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"Missing <{tag}> in XML element <{element.tag}>")
    return child.text.strip()


def ensure_dataset() -> Path:
    dataset_path = DATA_DIR / "face-mask-detection"
    if dataset_path.exists():
        return dataset_path

    downloaded_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    if not isinstance(downloaded_path, str) or not downloaded_path:
        raise RuntimeError("kagglehub.dataset_download returned an empty path")

    source_path = Path(downloaded_path).resolve()
    shutil.copytree(source_path, dataset_path)
    shutil.rmtree(source_path)
    return dataset_path


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_xml_clf(xml_file: Path) -> tuple[str, list[str]]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = require_child_text(root, "filename")
    labels = [require_child_text(obj, "name") for obj in root.findall("object")]
    return filename, labels


def build_classifier_dataframe(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "annotations"

    rows: list[dict[str, Any]] = []
    for xml_file in annotations_path.iterdir():
        filename, labels = parse_xml_clf(xml_file)
        if "with_mask" in labels and "without_mask" in labels:
            final_label = "mask_weared_incorrectly"
        elif "without_mask" in labels:
            final_label = "without_mask"
        else:
            final_label = "with_mask"
        rows.append({"image_path": str(images_path / filename), "label": final_label})

    df = pd.DataFrame(rows)
    label2id = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id).astype("int64")
    return df, label2id, id2label


VIT_IMAGE_SIZE = 224
VIT_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
VIT_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def vit_preprocess_images(images: list[Image.Image]) -> torch.Tensor:
    batch = []
    for image in images:
        resized = image.convert("RGB").resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - VIT_MEAN) / VIT_STD
        batch.append(arr.transpose(2, 0, 1))
    return torch.tensor(np.stack(batch), dtype=torch.float32)


def train_vit(args: argparse.Namespace) -> None:
    dataset_path = ensure_dataset()
    df, label2id, id2label = build_classifier_dataframe(dataset_path)

    hf_df = pd.DataFrame(df[["image_path", "label_id"]].copy())
    hf_df.columns = ["image_path", "label"]
    split = Dataset.from_pandas(hf_df, preserve_index=False).train_test_split(
        test_size=0.2,
        seed=42,
    )

    train_ds = split["train"].cast_column("image_path", HFImage())
    eval_ds = split["test"].cast_column("image_path", HFImage())

    cache_prefix = CACHE_DIR / "vit"

    def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        images = [img.convert("RGB") for img in batch["image_path"]]
        pixel_values = vit_preprocess_images(images).numpy()
        return {
            "pixel_values": [arr for arr in pixel_values],
            "labels": batch["label"],
        }

    prepared_train_ds = train_ds.map(
        preprocess_batch,
        batched=True,
        batch_size=32,
        remove_columns=["image_path"],
        load_from_cache_file=True,
        cache_file_name=str(cache_prefix.with_name("vit_train.arrow")),
        desc="Preprocessing train images",
    )
    prepared_eval_ds = eval_ds.map(
        preprocess_batch,
        batched=True,
        batch_size=32,
        remove_columns=["image_path"],
        load_from_cache_file=True,
        cache_file_name=str(cache_prefix.with_name("vit_eval.arrow")),
        desc="Preprocessing eval images",
    )

    prepared_train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    prepared_eval_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    torch.set_float32_matmul_precision("high")
    device = detect_device()
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    output_dir = NOTEBOOK_DIR / "vit-face-mask"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.vit_batch_size,
        per_device_eval_batch_size=args.vit_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.vit_epochs,
        fp16=False,
        bf16=False,
        logging_steps=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        disable_tqdm=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=args.workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_train_ds,
        eval_dataset=prepared_eval_ds,
        compute_metrics=compute_metrics,
    )

    print(f"ViT device: {device}")
    print(f"ViT workers: {args.workers}")
    print(f"Train: {len(prepared_train_ds)} | Eval: {len(prepared_eval_ds)}")
    trainer.train()
    results = trainer.evaluate()
    print("ViT evaluation:", results)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    box_predictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    in_features = box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def split_detection_dataset(
    dataset_path: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    images_dir = dataset_path / "images"
    annotations_dir = dataset_path / "annotations"
    all_images = sorted(images_dir.glob("*.png"))
    all_images = [p for p in all_images if (annotations_dir / f"{p.stem}.xml").exists()]
    if not all_images:
        raise RuntimeError(f"No annotated images found under {dataset_path}")

    rng = random.Random(seed)
    rng.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_paths = all_images[:n_train]
    val_paths = all_images[n_train:n_train + n_val]
    test_paths = all_images[n_train + n_val:]
    return train_paths, val_paths, test_paths


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = cast(dict[str, torch.Tensor], model(images, targets))
        losses = torch.stack(list(loss_dict.values())).sum()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
            print(
                f"  Epoch {epoch} [{i + 1}/{len(data_loader)}] "
                f"loss={losses.item():.4f} "
                f"(cls={loss_dict['loss_classifier'].item():.3f} "
                f"box={loss_dict['loss_box_reg'].item():.3f})"
            )
    return total_loss / len(data_loader)


@torch.no_grad()
def validate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = cast(dict[str, torch.Tensor], model(images, targets))
        total_loss += torch.stack(list(loss_dict.values())).sum().item()
    return total_loss / len(data_loader)


def plot_loss(history: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="Train Loss", color="#2563eb", lw=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#dc2626", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_rcnn(args: argparse.Namespace) -> None:
    dataset_path = ensure_dataset()
    device = detect_device()
    classes = ["__background__", "with_mask", "without_mask", "mask_weared_incorrect"]
    train_paths, val_paths, _ = split_detection_dataset(dataset_path)

    train_ds = FaceMaskDataset(train_paths, dataset_path, classes, augment=True)
    val_ds = FaceMaskDataset(val_paths, dataset_path, classes, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.rcnn_batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.rcnn_batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"Faster R-CNN device: {device}")
    print(f"Faster R-CNN workers: {args.workers}")
    print(f"Faster R-CNN epochs: {args.rcnn_epochs}")

    for epoch in range(1, args.rcnn_epochs + 1):
        started_at = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - started_at
        print(
            f"Epoch {epoch:02d}/{args.rcnn_epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.0f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "num_classes": len(classes),
                    "classes": classes,
                },
                CHECKPOINT_DIR / "best_model.pth",
            )

        if epoch % 5 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth",
            )

    plot_loss(history, CHECKPOINT_DIR / "loss_curve.png")
    print(f"Best Faster R-CNN val loss: {best_val_loss:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train StreamCamServer models outside Jupyter.")
    parser.add_argument(
        "target",
        choices=["vit", "rcnn", "all"],
        help="Which training flow to run.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker count.")
    parser.add_argument("--vit-epochs", type=int, default=3, help="ViT training epochs.")
    parser.add_argument("--vit-batch-size", type=int, default=16, help="ViT batch size.")
    parser.add_argument("--rcnn-epochs", type=int, default=10, help="Faster R-CNN training epochs.")
    parser.add_argument("--rcnn-batch-size", type=int, default=4, help="Faster R-CNN batch size.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers must be >= 0")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if args.target in {"vit", "all"}:
        train_vit(args)
    if args.target in {"rcnn", "all"}:
        train_rcnn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
