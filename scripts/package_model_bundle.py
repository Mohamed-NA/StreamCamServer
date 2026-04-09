from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamcamserver.paths import CONFIG_DIR, MODEL_DIR, PROJECT_ROOT


def _iter_bundle_files() -> list[Path]:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model directory: {MODEL_DIR}")

    files = sorted(
        path
        for path in MODEL_DIR.iterdir()
        if path.is_file() and path.name not in {".gitkeep", "README.md"}
    )
    if not files:
        raise FileNotFoundError("No runtime model files found in model/.")
    return files


def build_manifest(files: list[Path]) -> dict[str, object]:
    return {
        "project": "streamcamserver",
        "config": "config/models.json",
        "files": [
            {
                "path": f"model/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }


def package_bundle(output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _iter_bundle_files()
    manifest = build_manifest(files)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    with tarfile.open(output_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"model/{path.name}")
        archive.add(CONFIG_DIR / "models.json", arcname="config/models.json")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, fileobj=io.BytesIO(manifest_bytes))

    sha_path = output_path.with_suffix(output_path.suffix + ".sha256")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    return output_path, sha_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package runtime models into a release-ready archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "model-bundle.tar.gz",
        help="Where to write the tar.gz bundle.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle_path, sha_path = package_bundle(args.output)
    print(f"Wrote model bundle: {bundle_path}")
    print(f"Wrote checksum: {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
