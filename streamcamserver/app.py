from __future__ import annotations

import base64
import binascii
import ssl

import cv2
import numpy as np
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

from streamcamserver.inference import ModelManager
from streamcamserver.paths import CERTIFICATES_DIR, CONFIG_DIR, PROJECT_ROOT


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
socketio = SocketIO(app, async_mode="threading")
model_mgr = ModelManager(CONFIG_DIR / "models.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    return jsonify(model_mgr.list_models())


@socketio.on("switch_model")
def handle_switch_model(model_id: str):
    try:
        info = model_mgr.switch(model_id)
        socketio.emit("model_changed", {"ok": True, **info})
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        socketio.emit("model_changed", {"ok": False, "error": str(exc)})


@socketio.on("video_frame")
def handle_video_frame(data: str):
    if not isinstance(data, str) or "," not in data:
        socketio.emit("server_error", {"error": "Invalid frame payload"})
        return

    try:
        img_data = base64.b64decode(data.split(",", 1)[1])
    except (ValueError, binascii.Error):
        socketio.emit("server_error", {"error": "Could not decode frame"})
        return

    frame = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        socketio.emit("server_error", {"error": "Could not parse frame image"})
        return

    results = model_mgr.predict(frame)
    model_mgr.draw(frame, results)

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    socketio.emit("server_frame", frame_b64)
    socketio.emit(
        "detections",
        [{"label": result["label"], "confidence": round(result["confidence"], 3)} for result in results],
    )


def main() -> None:
    cert_file = CERTIFICATES_DIR / "certificate.crt"
    key_file = CERTIFICATES_DIR / "private.key"

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    socketio.run(
        app,
        host="0.0.0.0",
        port=8080,
        ssl_context=ssl_context,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
