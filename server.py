import ssl
import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# ── TFLite mask classifier (MobileNetV2, 224×224) ──
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES = ["With Mask", "Without Mask"]
COLORS  = {
    "With Mask":    (0, 200, 80),   # green
    "Without Mask": (0, 0, 220),    # red
}

# ── Haar face detector ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def classify_face(face_bgr):
    """Crop → preprocess → TFLite inference → (label, confidence)."""
    img = cv2.resize(face_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    img = (img.astype("float32") / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("video_frame")
def handle_video_frame(data):
    # Decode frame
    img_data = base64.b64decode(data.split(",")[1])
    frame    = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Detect faces
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    results = []
    for (x, y, w, h) in faces:
        label, confidence = classify_face(frame[y : y + h, x : x + w])
        color = COLORS[label]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Filled label pill above box
        text = f"{label}  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
        cv2.putText(frame, text, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({"label": label, "confidence": round(confidence, 3)})

    # Save and broadcast annotated frame
    cv2.imwrite("received_frame.jpg", frame)
    _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    socketio.emit("server_frame", frame_b64)
    socketio.emit("detections", results)

if __name__ == "__main__":
    cert_file = "certificates/certificate.crt"
    key_file  = "certificates/private.key"

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    socketio.run(
        app,
        host="0.0.0.0",
        port=8080,
        ssl_context=ssl_context,
        allow_unsafe_werkzeug=True,
    )
