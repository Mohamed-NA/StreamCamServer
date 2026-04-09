from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamcamserver.paths import MODEL_DIR, PROJECT_ROOT


def install_bundle(bundle_path: Path, destination: Path) -> None:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.startswith("model/") and member.isfile()
        ]
        if not members:
            raise RuntimeError("Bundle does not contain any model/ files.")

        for member in members:
            source = archive.extractfile(member)
            if source is None:
                continue
            (destination / Path(member.name).name).write_bytes(source.read())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install a packaged model bundle into model/.")
    parser.add_argument("bundle", type=Path, help="Path to model-bundle.tar.gz")
    parser.add_argument(
        "--destination",
        type=Path,
        default=MODEL_DIR,
        help="Where to extract the model files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    install_bundle(args.bundle, args.destination)
    try:
        shown_path = args.destination.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = args.destination
    print(f"Installed model bundle into {shown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
