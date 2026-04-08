"""
Model manager: load and hot-swap TFLite / ONNX inference backends.

Supported model types
─────────────────────
classifier  – Haar-cascade face detection → crop → classify each face
detector    – Model handles full-frame detection + classification itself

Supported backends
──────────────────
tflite  – TensorFlow Lite (channels-last NHWC)
onnx    – ONNX Runtime (channels-first NCHW)
"""

import json
import threading
from pathlib import Path

import cv2
import numpy as np

# ── Optional backends ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:
    _TF_OK = False

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False


# ── Haar face detector (shared) ────────────────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Label → bounding-box colour ────────────────────────────────────────────
LABEL_COLORS = {
    "with_mask":               (0, 200,  80),   # green
    "With Mask":               (0, 200,  80),
    "without_mask":            (0,   0, 220),   # red
    "Without Mask":            (0,   0, 220),
    "mask_weared_incorrectly": (0, 165, 255),   # orange
    "mask_weared_incorrect":   (0, 165, 255),
}
_DEFAULT_COLOR = (180, 180, 180)

_PROJECT_ROOT = Path(__file__).resolve().parent


# ── Preprocessing helpers ──────────────────────────────────────────────────
def _prep_mobilenet(img_bgr: np.ndarray, size: tuple) -> np.ndarray:
    """MobileNetV2: scale pixel values to [-1, 1], NHWC float32."""
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype("float32") / 127.5) - 1.0
    return img[np.newaxis]                          # (1, H, W, 3)


def _prep_imagenet_nchw(img_bgr: np.ndarray, size: tuple) -> np.ndarray:
    """ImageNet normalisation, channels-first NCHW float32 (ViT)."""
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    img  = (img - mean) / std                       # (H, W, 3)
    img  = img.transpose(2, 0, 1)                   # (3, H, W)
    return img[np.newaxis]                           # (1, 3, H, W)


def _prep_rcnn_nchw(img_bgr: np.ndarray, size: tuple) -> np.ndarray:
    """Faster R-CNN: scale to [0, 1], channels-first NCHW float32."""
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    img = img.transpose(2, 0, 1)                    # (3, H, W)
    return img[np.newaxis]                           # (1, 3, H, W)


_PREPROCESSORS = {
    "mobilenet":     _prep_mobilenet,
    "imagenet_nchw": _prep_imagenet_nchw,
    "rcnn_nchw":     _prep_rcnn_nchw,
}


# ── Base model ─────────────────────────────────────────────────────────────
class _BaseModel:
    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a BGR frame.
        Returns list of dicts: {label, confidence, box: [x, y, w, h]}
        """
        raise NotImplementedError


# ── TFLite classifier (face crop → classify) ──────────────────────────────
class _TFLiteClassifier(_BaseModel):
    def __init__(self, cfg: dict):
        if not _TF_OK:
            raise RuntimeError("tensorflow is not installed")
        self._classes  = cfg["classes"]
        self._size     = tuple(cfg["input_size"])
        self._preproc  = _PREPROCESSORS[cfg["preprocessing"]]
        interp = tf.lite.Interpreter(model_path=cfg["path"])
        interp.allocate_tensors()
        self._interp = interp
        self._in_idx  = interp.get_input_details()[0]["index"]
        self._out_idx = interp.get_output_details()[0]["index"]

    def predict(self, frame: np.ndarray) -> list[dict]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        results = []
        for (x, y, w, h) in faces:
            inp = self._preproc(frame[y : y + h, x : x + w], self._size)
            self._interp.set_tensor(self._in_idx, inp)
            self._interp.invoke()
            probs = self._interp.get_tensor(self._out_idx)[0]
            idx   = int(np.argmax(probs))
            results.append({
                "label":      self._classes[idx],
                "confidence": float(probs[idx]),
                "box":        [int(x), int(y), int(w), int(h)],
            })
        return results


# ── ONNX classifier (face crop → classify) ────────────────────────────────
class _OnnxClassifier(_BaseModel):
    def __init__(self, cfg: dict):
        if not _ORT_OK:
            raise RuntimeError("onnxruntime is not installed")
        self._classes = cfg["classes"]
        self._size    = tuple(cfg["input_size"])
        self._preproc = _PREPROCESSORS[cfg["preprocessing"]]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session    = ort.InferenceSession(cfg["path"], sess_options=opts)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, frame: np.ndarray) -> list[dict]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        results = []
        for (x, y, w, h) in faces:
            inp    = self._preproc(frame[y : y + h, x : x + w], self._size)
            logits = self._session.run(None, {self._input_name: inp})[0][0]
            # Softmax to get probabilities
            e      = np.exp(logits - logits.max())
            probs  = e / e.sum()
            idx    = int(np.argmax(probs))
            results.append({
                "label":      self._classes[idx],
                "confidence": float(probs[idx]),
                "box":        [int(x), int(y), int(w), int(h)],
            })
        return results


# ── ONNX detector (Faster R-CNN: full-frame → boxes + labels + scores) ─────
class _OnnxDetector(_BaseModel):
    def __init__(self, cfg: dict):
        if not _ORT_OK:
            raise RuntimeError("onnxruntime is not installed")
        self._classes    = cfg["classes"]
        self._size       = tuple(cfg["input_size"])
        self._preproc    = _PREPROCESSORS[cfg["preprocessing"]]
        self._conf_thr   = float(cfg.get("conf_threshold", 0.5))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session    = ort.InferenceSession(cfg["path"], sess_options=opts)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, frame: np.ndarray) -> list[dict]:
        h_orig, w_orig = frame.shape[:2]
        inp = self._preproc(frame, self._size)
        outputs = self._session.run(None, {self._input_name: inp})

        # torchvision Faster R-CNN ONNX outputs: boxes [N,4], labels [N], scores [N]
        boxes, labels, scores = outputs[0], outputs[1], outputs[2]
        scale_x = w_orig / self._size[0]
        scale_y = h_orig / self._size[1]

        results = []
        for box, lbl, score in zip(boxes, labels, scores):
            if float(score) < self._conf_thr:
                continue
            x1, y1, x2, y2 = box
            x = int(x1 * scale_x)
            y = int(y1 * scale_y)
            w = int((x2 - x1) * scale_x)
            h = int((y2 - y1) * scale_y)
            class_idx = int(lbl)
            label = (
                self._classes[class_idx]
                if class_idx < len(self._classes)
                else "unknown"
            )
            results.append({
                "label":      label,
                "confidence": float(score),
                "box":        [x, y, w, h],
            })
        return results


# ── Factory ────────────────────────────────────────────────────────────────
def _make_model(cfg: dict) -> _BaseModel:
    backend = cfg["backend"]
    mtype   = cfg["type"]
    if backend == "tflite" and mtype == "classifier":
        return _TFLiteClassifier(cfg)
    if backend == "onnx" and mtype == "classifier":
        return _OnnxClassifier(cfg)
    if backend == "onnx" and mtype == "detector":
        return _OnnxDetector(cfg)
    raise ValueError(f"Unsupported backend/type combination: {backend}/{mtype}")


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


# ── Public ModelManager ────────────────────────────────────────────────────
class ModelManager:
    """
    Thread-safe registry for inference models.

    Usage
    -----
    mgr = ModelManager("models.json")
    results = mgr.predict(frame)          # uses active model
    mgr.switch("vit")                     # hot-swap mid-run
    frame   = mgr.draw(frame, results)    # annotate frame
    """

    def __init__(self, config_path: str | Path = "models.json"):
        self._config_path = _resolve_path(config_path, base_dir=_PROJECT_ROOT)
        with self._config_path.open(encoding="utf-8") as f:
            cfg = json.load(f)

        config_dir = self._config_path.parent
        self._configs: dict[str, dict] = {}
        for model_cfg in cfg["models"]:
            cfg_copy = dict(model_cfg)
            cfg_copy["_resolved_path"] = _resolve_path(
                model_cfg["path"], base_dir=config_dir
            )
            self._configs[cfg_copy["id"]] = cfg_copy
        self._loaded:  dict[str, _BaseModel] = {}
        self._active:  str | None = None
        self._lock     = threading.Lock()

        default = cfg.get("default", cfg["models"][0]["id"])
        self.switch(default)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def active_id(self) -> str | None:
        return self._active

    def list_models(self) -> list[dict]:
        """Return metadata for all configured models."""
        return [
            {
                "id":          m["id"],
                "name":        m["name"],
                "description": m.get("description", ""),
                "available":   m["_resolved_path"].exists(),
                "active":      m["id"] == self._active,
            }
            for m in self._configs.values()
        ]

    def switch(self, model_id: str) -> dict:
        """
        Load (if needed) and activate a model by id.
        Returns the model's metadata dict.
        Raises ValueError / FileNotFoundError / RuntimeError on failure.
        """
        if model_id not in self._configs:
            raise ValueError(f"Unknown model id: '{model_id}'")

        cfg = self._configs[model_id]
        model_path = cfg["_resolved_path"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run the notebook to generate this model first."
            )

        with self._lock:
            if model_id not in self._loaded:
                load_cfg = dict(cfg)
                load_cfg["path"] = str(model_path)
                self._loaded[model_id] = _make_model(load_cfg)
            self._active = model_id

        return {
            "id":          cfg["id"],
            "name":        cfg["name"],
            "description": cfg.get("description", ""),
        }

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Run inference with the currently active model."""
        with self._lock:
            model = self._loaded.get(self._active)
        if model is None:
            return []
        return model.predict(frame)

    def draw(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto *frame* in-place."""
        for r in results:
            label = r["label"]
            conf  = r["confidence"]
            x, y, w, h = r["box"]
            color = LABEL_COLORS.get(label, _DEFAULT_COLOR)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
            cv2.putText(
                frame, text, (x + 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
        return frame
