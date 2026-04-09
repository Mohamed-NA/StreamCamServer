# StreamCamServer Documentation

## Overview

StreamCamServer is a Flask + Socket.IO application for browser-based webcam streaming with server-side mask inference. The browser captures frames, sends them to the server over WSS, and receives back an annotated frame plus structured detections.

The project supports three runtime models:

- `mobilenet`: TensorFlow Lite classifier
- `vit`: ONNX Runtime classifier
- `faster_rcnn`: ONNX Runtime detector

Model definitions and the active default are configured in [config/models.json](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/config/models.json).

## Current Structure

```text
.
├── streamcamserver/
│   ├── app.py                      # Flask app + Socket.IO handlers
│   ├── inference.py                # model loading and inference backends
│   ├── paths.py                    # shared project paths
│   └── training/
│       ├── data.py                 # Faster R-CNN dataset helpers
│       ├── train.py                # scriptable training entrypoint
│       └── export.py               # ONNX export logic
├── scripts/
│   ├── run_server.py               # launcher
│   ├── train_models.py             # training CLI
│   ├── export_models.py            # export CLI
│   ├── package_model_bundle.py     # release bundle packager
│   └── install_model_bundle.py     # local bundle installer
├── config/
│   └── models.json
├── model/                          # unpacked local runtime artifacts
├── infra/docker/
│   ├── Dockerfile.app
│   ├── Dockerfile.models
│   ├── compose.yaml
│   └── copy_models.sh
├── notebooks/
│   ├── mask_detector_4_DEBI.ipynb
│   └── train_in_colab.ipynb
├── templates/
├── static/
└── certificates/
```

## Runtime Architecture

The app entrypoint is [scripts/run_server.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/scripts/run_server.py), which calls into [streamcamserver/app.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/streamcamserver/app.py).

At startup the server:

1. Creates the Flask app and Socket.IO server
2. Loads the model registry from [config/models.json](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/config/models.json)
3. Instantiates the default model through [streamcamserver/inference.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/streamcamserver/inference.py)
4. Serves the UI from [templates/index.html](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/templates/index.html) and [static/script.js](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/static/script.js)

Frame flow:

1. Browser captures a frame from the webcam
2. Browser sends the frame to the server with Socket.IO
3. Server decodes the frame and runs the currently active backend
4. Server emits the annotated frame and detection payload back to the client

## Model Backends

Runtime loading lives in [streamcamserver/inference.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/streamcamserver/inference.py).

Supported backends:

- `tflite`
  - used for the MobileNetV2 classifier
- `onnx`
  - used for the ViT classifier
  - used for the Faster R-CNN detector

The config file declares:

- model id
- backend
- type (`classifier` or `detector`)
- runtime path
- class labels
- preprocessing mode
- optional confidence threshold

## Training And Export

Training and export code lives under [streamcamserver/training](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/streamcamserver/training).

Command entrypoints:

- [scripts/train_models.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/scripts/train_models.py)
- [scripts/export_models.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/scripts/export_models.py)

Training outputs:

- ViT checkpoints under `notebooks/vit-face-mask/`
- Faster R-CNN checkpoints under `notebooks/checkpoints/`

Export outputs:

- ONNX runtime files copied into `model/`

The local notebook and Colab notebook are wrappers around the same training/export logic, not a separate pipeline.

## Model Artifact Strategy

Runtime model binaries should not be kept in normal git history.

Instead:

1. Export runtime files into `model/`
2. Package them with [scripts/package_model_bundle.py](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/scripts/package_model_bundle.py)
3. Upload the archive to GitHub Releases
4. Trigger the GitHub Actions models-image workflow with the release tag
5. Download that archive when preparing a local runtime environment

Bundle commands:

```bash
uv run scripts/package_model_bundle.py
uv run scripts/install_model_bundle.py dist/model-bundle.tar.gz
```

The release bundle contains:

- runtime files from `model/`
- [config/models.json](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/config/models.json)
- `manifest.json`

The manifest records the SHA-256 checksum of every bundled runtime file.

Example release commands:

```bash
gh release create v1.0.0 --title "v1.0.0" --notes "Runtime model bundle release"
gh release upload v1.0.0 dist/model-bundle.tar.gz dist/model-bundle.tar.gz.sha256
```

## Docker Layout

Docker files live under [infra/docker](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/infra/docker).

### App Image

[infra/docker/Dockerfile.app](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/infra/docker/Dockerfile.app):

- installs project dependencies with `uv`
- copies application code, config, templates, static assets, and certificates
- starts the server via `python -m streamcamserver.app`

The app image does not embed runtime models.

### Models Image

[infra/docker/Dockerfile.models](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/infra/docker/Dockerfile.models):

- expects `dist/model-bundle.tar.gz` in the build context
- extracts the bundle into `/models`
- uses [infra/docker/copy_models.sh](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/infra/docker/copy_models.sh) to copy those files into a shared Docker volume

This keeps the model bundle separate from the app image and from the source repository history.

### Compose

[infra/docker/compose.yaml](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/infra/docker/compose.yaml) starts:

- `models`
  - one-shot init container
  - copies unpacked runtime artifacts into the shared `models-data` volume
- `app`
  - serves the UI on `https://localhost:8080`
  - mounts the shared volume at `/app/model`

## Local Development

Install dependencies:

```bash
uv sync --group notebook
```

Run the server:

```bash
uv run scripts/run_server.py
```

Run notebook tooling:

```bash
uv run --group notebook jupyter lab
```

Run training:

```bash
uv run scripts/train_models.py vit
uv run scripts/train_models.py rcnn
uv run scripts/train_models.py all
```

Run export:

```bash
uv run scripts/export_models.py all
```

Package the release bundle:

```bash
uv run scripts/package_model_bundle.py
```

## GitHub Releases Workflow

Recommended flow:

1. Train models locally or in Colab/Kaggle
2. Export runtime files into `model/`
3. Run `uv run scripts/package_model_bundle.py`
4. Upload `dist/model-bundle.tar.gz` and `dist/model-bundle.tar.gz.sha256` to a GitHub Release
5. Run the `Test and Publish Images` workflow manually and provide that release tag
6. The workflow downloads the release asset into `dist/`
7. The models image is built from that downloaded archive

This gives you:

- smaller source history
- reproducible Docker image builds
- explicit runtime artifact versioning

## Certificates

For local HTTPS/WSS, the app expects:

- `certificates/certificate.crt`
- `certificates/private.key`

Generate self-signed development certificates if needed:

```bash
mkdir -p certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certificates/private.key \
  -out certificates/certificate.crt
```

## Notes

- `data/` is local-only and gitignored
- notebook cache/checkpoint/export folders are gitignored
- `model/` is for unpacked local runtime files, not tracked binaries
- the release bundle is the source of truth for shipping runtime artifacts
