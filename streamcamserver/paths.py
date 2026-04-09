from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "config" / "models.json").exists():
            return candidate
    raise RuntimeError("Could not locate the StreamCamServer project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXPORT_DIR = NOTEBOOK_DIR / "exports"
CACHE_DIR = NOTEBOOK_DIR / "cache"
CHECKPOINT_DIR = NOTEBOOK_DIR / "checkpoints"
MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_DIR = PROJECT_ROOT / "config"
CERTIFICATES_DIR = PROJECT_ROOT / "certificates"

for path in (DATA_DIR, NOTEBOOK_DIR, MODEL_DIR, CONFIG_DIR):
    path.mkdir(parents=True, exist_ok=True)
