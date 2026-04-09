# StreamCamServer

Real-time browser camera streaming over HTTPS/WSS with server-side mask detection and hot-swappable inference models.

The app serves a web UI, receives webcam frames over Socket.IO, runs inference on the server, and sends back an annotated frame plus structured detections.

## What It Supports

- `MobileNetV2` TFLite classifier
- `ViT Base` ONNX classifier
- `Faster R-CNN` ONNX detector
- live model switching from the UI
- HTTPS + WSS with local certificates
- notebook workflow for training and ONNX export

## Project Layout

```text
.
├── server.py                      # Flask + Socket.IO server
├── model_manager.py               # model loading, switching, inference, drawing
├── models.json                    # model registry and active default
├── notebooks/
│   └── mask_detector_4_DEBI.ipynb # training + export notebook
├── model/                         # runtime model artifacts used by the server
├── data/                          # local dataset storage (gitignored)
├── templates/
├── static/
└── certificates/
```

## Requirements

- Python `3.12`
- `uv`
- local certificate files at:
  - `certificates/certificate.crt`
  - `certificates/private.key`

For the Faster R-CNN notebook section, your Python build must support `lzma` so `torchvision` imports cleanly.

## Setup

Create the environment and install project dependencies:

```bash
uv sync --group notebook
```

Use the project environment for everything:

```bash
uv run python --version
```

For Jupyter:

```bash
uv run --group notebook jupyter lab
```

## Running The Server

Start the app with the project environment:

```bash
uv run python server.py
```

The server listens on:

- `https://0.0.0.0:8080`

Open it in your browser at:

```text
https://localhost:8080
```

## Docker

This repo now supports a split container layout:

- `streamcamserver-app`: server + UI
- `streamcamserver-models`: runtime model bundle

The app container keeps the current UI model chooser by reading all model files from a shared Docker volume mounted at `/app/model`.

### Build Local Images

```bash
docker build -t streamcamserver-app:local .
docker build -f Dockerfile.models -t streamcamserver-models:local .
```

### Run With Compose

```bash
docker compose up
```

That starts:

- a `models` init container that copies model files into a named Docker volume
- an `app` container that mounts the same volume and serves the UI on `https://localhost:8080`
- the app image uses the bundled `certificates/` directory by default

## Certificates

If you do not already have local certs, create them with:

```bash
mkdir -p certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certificates/private.key \
  -out certificates/certificate.crt
```

## Models

The server reads model definitions from [models.json](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/models.json).

Configured models:

- `mobilenet`: TFLite 2-class classifier
- `vit`: ONNX 3-class classifier
- `faster_rcnn`: ONNX detector

At runtime:

- `GET /api/models` returns model availability and active state
- Socket event `switch_model` swaps the active model
- Socket event `video_frame` sends a frame to the server

## Notebook Workflow

The notebook is:

- [mask_detector_4_DEBI.ipynb](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/notebooks/mask_detector_4_DEBI.ipynb)

For Google Colab, use:

- [train_in_colab.ipynb](/Users/nasser/ettbtm/work/DEBI/StreamCamServer/notebooks/train_in_colab.ipynb)

It is set up to:

- require the repo `.venv`
- download the Kaggle dataset
- copy the dataset into:
  - `data/face-mask-detection`
- remove the temporary KaggleHub cache copy after the local copy is created
- train the ViT classifier
- train the Faster R-CNN detector
- export ONNX models into `notebooks/exports`
- copy exported models into `model/`

Dataset contents are intentionally gitignored.

## Script Training

You can run training outside Jupyter with:

```bash
uv run python train_models.py vit
uv run python train_models.py rcnn
uv run python train_models.py all
```

The script defaults to `--workers 2`, which is a safer baseline for a laptop workflow than using aggressive notebook multiprocessing.

## Expected Model Files

The server can run with the following files in `model/`:

- `model.tflite`
- `vit_model.onnx`
- `faster_rcnn.onnx`

If an ONNX model is missing, it will show as unavailable in the UI and API.

## Docker Hub Workflow

If you publish both images to Docker Hub, users can run the full stack without cloning the repo.

Pull:

```bash
docker pull <dockerhub-user>/streamcamserver-app:latest
docker pull <dockerhub-user>/streamcamserver-models:latest
```

Run with Compose:

```bash
APP_IMAGE=<dockerhub-user>/streamcamserver-app:latest \
MODELS_IMAGE=<dockerhub-user>/streamcamserver-models:latest \
docker compose up
```

The current UI model chooser will still work because the app container sees all model files locally in the shared volume.

## Development Notes

- Use `uv`, not `pip`, for environment management in this repo.
- The notebook and pyright config are set up for the repo-local `.venv`.
- `data/face-mask-detection/` is ignored and should not be committed.
- The app image is code-only and expects model files from the shared `/app/model` volume.
- The models image publishes runtime artifacts separately from the server/UI image.

## Common Commands

Sync dependencies:

```bash
uv sync --group notebook
```

Run the server:

```bash
uv run python server.py
```

Open Jupyter:

```bash
uv run --group notebook jupyter lab
```

Run training as a script:

```bash
uv run python train_models.py all
```

Verify the runtime environment:

```bash
./.venv/bin/python -c "import cv2, numpy, flask, flask_socketio; print('ok')"
```

Build split Docker images:

```bash
docker build -t streamcamserver-app:local .
docker build -f Dockerfile.models -t streamcamserver-models:local .
```

Run split Docker stack:

```bash
docker compose up
```
