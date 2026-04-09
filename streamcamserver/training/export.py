from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import torchvision
from transformers import ViTForImageClassification

from streamcamserver.paths import CHECKPOINT_DIR, EXPORT_DIR, MODEL_DIR, NOTEBOOK_DIR
from streamcamserver.training.train import build_model


class _ViTONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class _RCNNONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, images: torch.Tensor):
        preds = self.model(list(images))[0]
        return preds["boxes"], preds["labels"].to(torch.float32), preds["scores"]


def _copy_export_file(output_path: Path) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / output_path.name
    destination.write_bytes(output_path.read_bytes())
    sidecar = output_path.with_suffix(output_path.suffix + ".data")
    if sidecar.exists():
        (MODEL_DIR / sidecar.name).write_bytes(sidecar.read_bytes())
    return destination


def find_vit_checkpoint() -> Path | None:
    base = NOTEBOOK_DIR / "vit-face-mask"
    checkpoints = sorted(base.glob("checkpoint-*/")) if base.exists() else []
    return checkpoints[-1] if checkpoints else None


def export_vit() -> Path | None:
    checkpoint_dir = find_vit_checkpoint()
    if checkpoint_dir is None:
        print("Skipping ViT export: no trained checkpoint found in notebooks/vit-face-mask/")
        return None

    model = ViTForImageClassification.from_pretrained(str(checkpoint_dir)).cpu().eval()
    wrapper = _ViTONNXWrapper(model)
    output_path = EXPORT_DIR / "vit_model.onnx"
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output_path),
            opset_version=14,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        )

    onnx.checker.check_model(str(output_path))
    destination = _copy_export_file(output_path)
    print(f"Exported ViT ONNX to {destination}")
    return destination


def export_rcnn() -> Path:
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing RCNN checkpoint: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model = build_model(num_classes=int(checkpoint.get("num_classes", 4)), pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    wrapper = _RCNNONNXWrapper(model)
    output_path = EXPORT_DIR / "faster_rcnn.onnx"
    dummy = torch.zeros(1, 3, 640, 640)
    orig_tracing = getattr(torchvision, "_is_tracing", None)
    if orig_tracing is not None:
        torchvision._is_tracing = lambda: True

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(output_path),
                opset_version=12,
                dynamo=False,
                input_names=["images"],
                output_names=["boxes", "labels", "scores"],
                dynamic_axes={
                    "images": {0: "batch", 2: "H", 3: "W"},
                    "boxes": {0: "N"},
                    "labels": {0: "N"},
                    "scores": {0: "N"},
                },
                training=torch.onnx.TrainingMode.EVAL,
            )
    finally:
        if orig_tracing is not None:
            torchvision._is_tracing = orig_tracing

    onnx.checker.check_model(str(output_path))
    destination = _copy_export_file(output_path)
    print(f"Exported Faster R-CNN ONNX to {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained checkpoints to ONNX runtime files.")
    parser.add_argument("target", choices=["vit", "rcnn", "all"], help="Which export to run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.target in {"vit", "all"}:
        export_vit()
    if args.target in {"rcnn", "all"}:
        export_rcnn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
