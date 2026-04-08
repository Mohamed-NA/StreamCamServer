# StreamCamServer — Full Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [Backend (server.py)](#4-backend-serverpy)
5. [Frontend (index.html + script.js)](#5-frontend-indexhtml--scriptjs)
6. [Machine Learning Component](#6-machine-learning-component)
7. [Dependencies](#7-dependencies)
8. [Configuration Files](#8-configuration-files)
9. [Security & SSL](#9-security--ssl)
10. [Docker & Deployment](#10-docker--deployment)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [End-to-End Workflow](#12-end-to-end-workflow)
13. [Running the Project](#13-running-the-project)

---

## 1. Project Overview

StreamCamServer is a **real-time WebSocket-based camera streaming server** with integrated **face mask detection** using machine learning. A web client captures frames from the user's camera and streams them to a Python Flask server, which runs face detection and mask classification on each frame, then returns annotated results back to the browser in near real-time.

**Key capabilities**:

- Live camera streaming from browser to server via WebSocket over TLS (WSS)
- Server-side face detection using OpenCV Haar Cascade
- Per-face mask classification using a TensorFlow Lite MobileNetV2 model
- Annotated frame returned to the browser with bounding boxes and labels
- Configurable frame rate, JPEG quality, and video resolution
- Fully containerized with Docker, with a GitHub Actions CI/CD pipeline

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser (Client)                     │
│                                                                 │
│  getUserMedia API ──→ Local Canvas ──→ JPEG Encode ──→ Base64   │
│                                                        |        |
│  Display Canvas ←── Annotated Frame (Base64) ←─────────┘        |
│  Detection Panel ←── Detections JSON                            |
└───────────────────────────── WebSocket (WSS) ───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask + Flask-SocketIO Server               │
│                                                                 │
│  Frame Decode ──→ OpenCV Haar Cascade (Face Detection)         │
│                          │                                      │
│                     For each face:                              │
│                     └──→ Crop + Resize (224×224)               │
│                          └──→ Normalize → TFLite Inference     │
│                               └──→ Class + Confidence          │
│                                                                 │
│  Annotate Frame ──→ Encode JPEG ──→ Emit server_frame          │
│  Build JSON ──→ Emit detections                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Communication model**: The client sends one frame and waits for the server to respond before sending the next. This **ack-based flow control** prevents the WebSocket queue from filling up under a slow server or heavy inference load.

---

## 3. File Structure

```
StreamCamServer/
├── server.py                          # Flask application — all backend logic
├── pyproject.toml                     # Project metadata and Python dependencies
├── uv.lock                            # Locked dependency graph (uv package manager)
├── .python-version                    # Pins Python version to 3.13
├── Dockerfile                         # Container build instructions
├── .dockerignore                      # Files excluded from Docker context
├── .gitignore                         # Files excluded from git
├── LICENSE                            # MIT License
├── Readme.md                          # Quick-start guide
│
├── templates/
│   └── index.html                     # Single-page UI (631 lines)
│
├── static/
│   └── script.js                      # WebSocket client + camera logic (237 lines)
│
├── model/
│   ├── model.tflite                   # Pre-trained TFLite mask detection model (~9.8 MB)
│   └── labels.txt                     # Class labels: WithMask, WithoutMask
│
├── certificates/
│   ├── certificate.crt                # Self-signed SSL certificate
│   └── private.key                    # SSL private key
│
├── notebooks/                         # Empty — reserved for Jupyter exploration
├── covid-19-mask-detector.ipynb       # Training / exploration notebook
│
└── .github/
    └── workflows/
        └── docker-publish.yml         # GitHub Actions CI/CD pipeline
```

---

## 4. Backend (server.py)

The entire server is 98 lines of Python. It does four things: initializes the ML model, sets up Flask/SocketIO, handles incoming WebSocket frames, and serves the HTML page.

### 4.1 Imports and Initialization

```python
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2, numpy as np, base64, ssl, tensorflow as tf
```

**TFLite model setup** runs at startup:

```python
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

The model expects a `(1, 224, 224, 3)` float32 tensor and outputs a `(1, 2)` probability array.

**Face detector setup**:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

OpenCV ships with pre-trained Haar Cascade XML files. This uses the frontal face classifier bundled with the library.

**Flask and SocketIO**:

```python
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
```

`async_mode="threading"` is required when combining Flask-SocketIO with TensorFlow (which is not async-safe).

### 4.2 `classify_face(face_bgr)` — ML inference

```python
def classify_face(face_bgr):
    img = cv2.resize(face_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 127.5) - 1.0      # normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)                  # add batch dimension

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx  = int(np.argmax(output))
    confidence = float(output[class_idx])
    label      = labels[class_idx]                     # "WithMask" or "WithoutMask"
    return label, confidence
```

The normalization `(x / 127.5) - 1.0` maps pixel values from `[0, 255]` to `[-1, 1]`, which matches the preprocessing used during MobileNetV2 training.

### 4.3 `handle_video_frame(data)` — WebSocket handler

Triggered by the `video_frame` SocketIO event:

1. **Decode**: Strip the `data:image/jpeg;base64,` prefix and decode to bytes, then to a numpy array with OpenCV.
2. **Face detection**:
   ```python
   gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
               minNeighbors=5, minSize=(60, 60))
   ```
3. **Per-face classification**: Call `classify_face()` on each detected face crop.
4. **Annotation**: Draw colored rectangles and label text on the frame.
   - Green (`(0, 255, 0)`) for `WithMask`
   - Red (`(0, 0, 255)`) for `WithoutMask`
5. **Save debug frame**: Write annotated frame to `received_frame.jpg`.
6. **Emit results** back to the caller:
   ```python
   emit('server_frame', {'frame': encoded_frame})
   emit('detections',   {'detections': detections_list})
   ```

### 4.4 Routes and Server Start

```python
@app.route('/')
def index():
    return render_template('index.html')
```

```python
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('certificates/certificate.crt',
                             'certificates/private.key')

socketio.run(app, host='0.0.0.0', port=8080,
             ssl_context=ssl_context, allow_unsafe_werkzeug=True)
```

The server listens on all interfaces at port `8080` with HTTPS/WSS enabled.

---

## 5. Frontend (index.html + script.js)

### 5.1 Layout (index.html)

The UI is a single HTML file rendered by Flask's Jinja2 engine (no dynamic template variables are used — the file is purely static HTML/CSS/JS served through Flask).

**Visual structure** (CSS Grid):

```
┌──────────────────────────────────────────────┐
│  Header: "StreamCam" logo + connection badge │
├──────────────────────┬───────────────────────┤
│                      │  Detection Stats      │
│   Video Canvas       │  ─ Faces detected     │
│   (with REC badge    │  ─ With mask          │
│    and pause button) │  ─ Without mask       │
│                      │                       │
│                      │  Controls Panel       │
│                      │  ─ FPS slider         │
│                      │  ─ Quality slider     │
│                      │  ─ Resolution picker  │
│                      │                       │
│                      │  Activity Log         │
├──────────────────────┴───────────────────────┤
│  Footer: copyright + resolution display      │
└──────────────────────────────────────────────┘
```

**Visual design**:
- Dark background (`#070a10`)
- Indigo accent (`#6366f1`)
- Glassmorphism panels (backdrop blur + subtle border)
- Pulsing connection indicator
- Blinking REC badge while streaming

### 5.2 Camera Capture (script.js)

```javascript
const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: resW }, height: { ideal: resH } }
});
video.srcObject = stream;
```

Resolution options:
- 240p → 426×240
- 480p → 640×480 (default)
- 720p → 1280×720

### 5.3 Frame Sending Loop

```javascript
async function sendFrame() {
    if (paused || !socket.connected) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', quality / 100);

    socket.emit('video_frame', dataUrl, () => {
        // ack received — schedule next frame
        setTimeout(sendFrame, 1000 / targetFps);
    });
}
```

The callback-based acknowledgement creates a natural back-pressure mechanism: the next frame is only scheduled after the server has processed the current one.

### 5.4 Receiving Server Results

```javascript
socket.on('server_frame', (data) => {
    serverImg.src = data.frame;       // replace canvas with annotated frame
    frameCount++;
});

socket.on('detections', (data) => {
    updateDetectionPanel(data.detections);   // update sidebar stats
});
```

### 5.5 UI State Machine

| State | Description |
|-------|-------------|
| `disconnected` | Socket not connected, no camera |
| `connecting` | Socket connecting or camera permission pending |
| `connected` | Socket live, stream not yet started |
| `live` | Sending frames, receiving results |
| `paused` | Stream suspended by user |
| `error` | Camera denied or socket error |

The connection badge and REC indicator update to reflect the current state.

### 5.6 User Controls

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| FPS slider | 1–30 fps | 15 fps | Inter-frame delay = 1000 / fps ms |
| Quality slider | 10–100% | 80% | JPEG compression level |
| Resolution picker | 240p / 480p / 720p | 480p | Restarts camera stream |
| Pause button | toggle | — | Stops/resumes frame sending |

### 5.7 Activity Log

A timestamped list of events (max 30 entries) shown in the sidebar:

- Camera started / stopped
- Stream paused / resumed
- Connection established / lost
- Errors (permission denied, camera unavailable)

---

## 6. Machine Learning Component

### 6.1 Model Architecture

The model is a **MobileNetV2**-based binary classifier fine-tuned for face mask detection. MobileNetV2 was chosen because:
- It is lightweight (~9.8 MB as TFLite)
- Fast enough for real-time inference on CPU
- Pre-trained on ImageNet, requiring less data to fine-tune

The training process is documented in `covid-19-mask-detector.ipynb`.

### 6.2 Classes

`model/labels.txt`:
```
WithMask
WithoutMask
```

### 6.3 Inference Pipeline

```
Face crop (BGR numpy array)
      │
      ▼
cv2.resize → (224, 224)
      │
      ▼
cvtColor BGR → RGB
      │
      ▼
Normalize: (pixel / 127.5) - 1.0  →  float32 in [-1, 1]
      │
      ▼
Expand dims → shape (1, 224, 224, 3)
      │
      ▼
TFLite interpreter.invoke()
      │
      ▼
Output tensor shape (1, 2) → [p_WithMask, p_WithoutMask]
      │
      ▼
argmax → class index
confidence = output[class_index]
label = labels[class_index]
```

### 6.4 Face Detector

Haar Cascade (`haarcascade_frontalface_default.xml`) parameters used:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `scaleFactor` | 1.1 | Scale step between detection windows |
| `minNeighbors` | 5 | Minimum overlapping rectangles to confirm detection |
| `minSize` | (60, 60) | Ignore faces smaller than 60×60 pixels |

---

## 7. Dependencies

### 7.1 Python Version

The project requires **Python 3.13** (pinned in `.python-version` and `pyproject.toml`). The Docker image uses Python 3.12-slim.

### 7.2 Direct Dependencies (`pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | >=3.1.3 | Web framework and template rendering |
| `flask-socketio` | >=5.6.1 | WebSocket event handling |
| `numpy` | >=2.4.4 | Numerical array operations |
| `opencv-python-headless` | >=4.13.0.92 | Image processing and face detection (no GUI) |
| `tensorflow` | >=2.21.0 | TFLite interpreter for ML inference |

### 7.3 Key Transitive Dependencies

| Package | Purpose |
|---------|---------|
| `werkzeug` | WSGI server (Flask backend) |
| `python-socketio` | SocketIO protocol implementation |
| `python-engineio` | WebSocket/polling engine |
| `simple-websocket` | WebSocket transport layer |
| `keras` | High-level ML API (bundled with TensorFlow) |

### 7.4 Package Manager

The project uses **uv** (a fast Rust-based Python package manager) for dependency management. `uv.lock` pins every dependency's version and hash for reproducible installs.

---

## 8. Configuration Files

### 8.1 `pyproject.toml`

Standard PEP 621 project metadata:

```toml
[project]
name            = "streamcamserver"
version         = "0.1.0"
description     = "WebSocket camera streaming with mask detection"
requires-python = ">=3.13"
dependencies    = [
    "flask>=3.1.3",
    "flask-socketio>=5.6.1",
    "numpy>=2.4.4",
    "opencv-python-headless>=4.13.0.92",
    "tensorflow>=2.21.0",
]
```

### 8.2 `.python-version`

```
3.13
```

Used by `pyenv` and `uv` to auto-select the correct Python interpreter.

### 8.3 `.gitignore`

Excludes:
- `__pycache__/`, `*.pyc`
- `.venv*/`, `venv*/`
- `.pytest_cache/`
- `received_frame.jpg` (runtime output)

### 8.4 `.dockerignore`

Excludes the same patterns as `.gitignore` to keep the Docker build context minimal.

---

## 9. Security & SSL

### 9.1 Self-Signed Certificates

```
certificates/
├── certificate.crt   (1245 bytes)
└── private.key       (1704 bytes)
```

These are pre-generated self-signed certificates bundled with the repository for development convenience. **For production**, replace these with certificates from a trusted CA (e.g., Let's Encrypt).

### 9.2 Why HTTPS/WSS is Required

The browser's `getUserMedia` API that provides access to the camera is only available in **secure contexts** (HTTPS or localhost). Without SSL, the camera stream cannot be initiated at all — this is a browser security policy, not a server choice.

### 9.3 SSL Context Setup

```python
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(
    'certificates/certificate.crt',
    'certificates/private.key'
)
```

---

## 10. Docker & Deployment

### 10.1 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "server.py"]
```

**Key notes**:
- `libgl1` and `libglib2.0-0` are required at runtime by OpenCV (even the headless variant links against these system libraries)
- The `EXPOSE 5000` declaration in the Dockerfile is informational; the actual Flask server binds to port `8080` inside the container

### 10.2 Running Locally with Docker

```bash
# Build
docker build -t streamcamserver .

# Run (map container 8080 → host 5000)
docker run -p 5000:8080 streamcamserver

# Access
open https://localhost:5000
```

Accept the browser's self-signed certificate warning to proceed.

### 10.3 Running Without Docker

```bash
# With uv (recommended)
uv sync
uv run python server.py

# With pip
pip install -r requirements.txt   # or use pyproject.toml
python server.py
```

---

## 11. CI/CD Pipeline

### 11.1 Trigger Conditions (`.github/workflows/docker-publish.yml`)

| Event | Jobs Run |
|-------|----------|
| Push to any branch | `test` only |
| Push to `main` | `test` + `build-and-push` |
| Manual (`workflow_dispatch`) | `test` + `build-and-push` |

### 11.2 Test Job

Runs on every push. Steps:
1. Checkout code
2. Setup Python 3.12
3. Install system dependencies (`libgl1`, `libglib2.0-0`) for OpenCV
4. Install Python dependencies from `pyproject.toml`
5. Smoke test — imports the server module and verifies `app` and `socketio` objects exist:
   ```bash
   python -c "from server import app, socketio; assert app; assert socketio"
   ```

### 11.3 Build & Push Job

Runs on `main` or manual dispatch. Steps:
1. Checkout code
2. Log in to Docker Hub using repository secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
3. Set up Docker Buildx (multi-platform builder)
4. Build and push image as `<username>/streamcamserver:latest`
5. **Health check**: Pull the image, run a temporary container, and hit `https://localhost:5000` with curl (ignoring self-signed cert)
6. **Cleanup**: Stop and remove the test container

---

## 12. End-to-End Workflow

```
1. Browser → GET https://<host>:5000
         ← server returns index.html

2. Client JS → WebSocket handshake (wss://<host>:5000)
         ← socket.on('connect')

3. User clicks "Start Camera"
   Client JS → getUserMedia({ video: true })
         ← MediaStream

4. Streaming loop begins:
   ┌─────────────────────────────────────────────────────────────┐
   │ a. drawImage(video) onto canvas                             │
   │ b. canvas.toDataURL('image/jpeg', quality)  → base64 JPEG  │
   │ c. socket.emit('video_frame', base64data, ackCallback)      │
   │                                                             │
   │    Server receives 'video_frame':                           │
   │    d. base64 decode → numpy array                           │
   │    e. cv2.cvtColor → grayscale                              │
   │    f. face_cascade.detectMultiScale() → face rectangles     │
   │    g. for each face:                                        │
   │         crop → resize 224×224 → normalize → TFLite.invoke  │
   │         → label (WithMask/WithoutMask) + confidence         │
   │    h. draw bounding boxes + labels on frame                 │
   │    i. save to received_frame.jpg                            │
   │    j. emit('server_frame', {frame: base64})                 │
   │    k. emit('detections',   {detections: [...]})             │
   │                                                             │
   │    Client receives results:                                 │
   │    l. socket.on('server_frame') → display annotated frame   │
   │    m. socket.on('detections')  → update detection sidebar   │
   │    n. ackCallback fires → schedule next frame after delay   │
   └─────────────────────────────────────────────────────────────┘

5. User pauses → frame sending stops, REC badge goes static
6. User resumes → loop restarts from step 4a
```

---

## 13. Running the Project

### Prerequisites

- Python 3.13+
- `uv` package manager (or `pip`)
- A modern browser (Chrome, Firefox, Edge)

### Quick Start

```bash
# Clone
git clone https://github.com/MohamedEshmawy/StreamCamServer.git
cd StreamCamServer

# Install dependencies
uv sync

# Start server
uv run python server.py
```

Then open **https://localhost:8080** in your browser. Accept the self-signed certificate warning, then click **Start Camera**.

### Environment Notes

- The server **must** run over HTTPS for `getUserMedia` to work (browser requirement)
- All five Python packages must be installed — TensorFlow is the largest (~500 MB)
- The `received_frame.jpg` file is created/overwritten by the server on every processed frame; it is excluded from git

---

*MIT License — MohamedEshmawy*
