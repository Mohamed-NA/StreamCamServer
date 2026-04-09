# StreamCamServer

Real-time browser camera streaming over HTTPS/WSS with server-side mask detection and switchable runtime models.

The app serves a web UI, receives webcam frames over Socket.IO, runs inference on the server, and returns an annotated frame plus structured detections.

## Quick Start

If you just want to run the app on a new device, use Docker first.

### Docker First

This repo uses a split image layout:

- `streamcamserver-app`: code, UI, config, certificates
- `streamcamserver-models`: runtime model bundle only

If the images are already published, outside users do not need this repo at all.

Create a `compose.yaml` in any folder with:

```yaml
services:
  models:
    image: iroyal9/streamcamserver-models:latest
    restart: "no"
    volumes:
      - models-data:/shared-models

  app:
    image: iroyal9/streamcamserver-app:latest
    depends_on:
      models:
        condition: service_completed_successfully
    ports:
      - "8080:8080"
    volumes:
      - models-data:/app/model:ro

volumes:
  models-data:
```

Then run:

```bash
docker compose up
```

The stack serves the app on:

```text
https://localhost:8080
```

If you already cloned the repo, you can also run:

```bash
APP_IMAGE=iroyal9/streamcamserver-app:latest \
MODELS_IMAGE=iroyal9/streamcamserver-models:latest \
docker compose -f infra/docker/compose.yaml up
```

### New Device Setup

For a new machine, choose one of these paths:

1. Docker only
   - use the published app image and models image
   - run the compose stack
2. Repo + model bundle
   - clone the repo
   - install dependencies
   - download `model-bundle.tar.gz` from GitHub Releases
   - install it into `model/`
   - run the server locally

Repo + bundle commands:

```bash
uv sync --group notebook
uv run scripts/install_model_bundle.py dist/model-bundle.tar.gz
uv run scripts/run_server.py
```

## Project Layout

```text
.
├── streamcamserver/               # runtime package + training/export code
│   ├── app.py
│   ├── inference.py
│   ├── paths.py
│   └── training/
├── scripts/
│   ├── run_server.py
│   ├── train_models.py
│   ├── export_models.py
│   ├── package_model_bundle.py
│   └── install_model_bundle.py
├── config/
│   └── models.json                # model registry and active default
├── infra/
│   └── docker/
│       ├── Dockerfile.app
│       ├── Dockerfile.models
│       ├── compose.yaml
│       └── copy_models.sh
├── model/                         # local unpacked runtime artifacts
├── notebooks/
│   ├── mask_detector_4_DEBI.ipynb
│   └── train_in_colab.ipynb
├── templates/
├── static/
└── certificates/
```

## Supported Runtime Models

- `MobileNetV2` classifier via TensorFlow Lite
- `ViT Base` classifier via ONNX Runtime
- `Faster R-CNN` detector via ONNX Runtime

The active/default registry lives in [config/models.json](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/config/models.json).

## Requirements

- Python `3.12`
- `uv`
- local TLS certs at:
  - `certificates/certificate.crt`
  - `certificates/private.key`

For training and export, install the notebook dependency group.

## Local Development Setup

```bash
uv sync --group notebook
```

## Run The Server

```bash
uv run scripts/run_server.py
```

The app listens on `https://localhost:8080`.

## Training

Run training outside the notebook with:

```bash
uv run scripts/train_models.py vit
uv run scripts/train_models.py rcnn
uv run scripts/train_models.py all
```

The training script defaults to `--workers 2`, which is a safer laptop baseline than aggressive notebook multiprocessing.

## Export Runtime Models

After training checkpoints exist:

```bash
uv run scripts/export_models.py vit
uv run scripts/export_models.py rcnn
uv run scripts/export_models.py all
```

This writes deployable runtime files into `model/`.

## Model Bundle Workflow

Runtime model binaries are meant to live outside normal git history.

Local workflow:

1. Export runtime files into `model/`
2. Package them into a release archive
3. Upload the archive to GitHub Releases
4. Trigger the GitHub Actions models-image build with that release tag

Create the bundle:

```bash
uv run scripts/package_model_bundle.py
```

That writes:

- `dist/model-bundle.tar.gz`
- `dist/model-bundle.tar.gz.sha256`

Install a downloaded bundle into `model/`:

```bash
uv run scripts/install_model_bundle.py dist/model-bundle.tar.gz
```

The bundle includes:

- runtime files from `model/`
- `config/models.json`
- a `manifest.json` with per-file SHA-256 hashes

Example upload with GitHub CLI:

```bash
gh release create v1.0.0 --title "v1.0.0" --notes "Runtime model bundle release"
gh release upload v1.0.0 dist/model-bundle.tar.gz dist/model-bundle.tar.gz.sha256
```

## Docker

The models image now builds from `dist/model-bundle.tar.gz`, not directly from tracked files in `model/`.

Build locally:

```bash
uv run scripts/package_model_bundle.py
docker build -f infra/docker/Dockerfile.app -t streamcamserver-app:local .
docker build -f infra/docker/Dockerfile.models -t streamcamserver-models:local .
```

Run with Compose:

```bash
docker compose -f infra/docker/compose.yaml up
```

The `models` container copies the unpacked bundle into a shared Docker volume mounted at `/app/model` in the app container.

## GitHub Releases Recommendation

Recommended artifact flow:

1. Train/export models locally or in Colab/Kaggle
2. Run `uv run scripts/package_model_bundle.py`
3. Upload `dist/model-bundle.tar.gz` and its checksum to a GitHub Release
4. Run the `Test and Publish Images` workflow manually with `release_tag=<your-tag>`
5. The workflow downloads the release asset and builds the models image from it

This keeps runtime binaries out of normal source history while still allowing reproducible image builds.

## Notebooks

- Local notebook: [notebooks/mask_detector_4_DEBI.ipynb](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/notebooks/mask_detector_4_DEBI.ipynb)
- Colab notebook: [notebooks/train_in_colab.ipynb](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/notebooks/train_in_colab.ipynb)

The notebook workflow is for training and export. The deployable runtime artifacts belong in `model/` or in a packaged release bundle.

## Common Commands

Sync dependencies:

```bash
uv sync --group notebook
```

Run the app:

```bash
uv run scripts/run_server.py
```

Open Jupyter:

```bash
uv run --group notebook jupyter lab
```

Package the runtime bundle:

```bash
uv run scripts/package_model_bundle.py
```

Install a bundle into `model/`:

```bash
uv run scripts/install_model_bundle.py dist/model-bundle.tar.gz
```
