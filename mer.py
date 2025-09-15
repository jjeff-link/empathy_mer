import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from transformers import pipeline
import sounddevice as sd

# -----------------------------
# Emotion Mappings (Ekman + Neutral)
# -----------------------------
FACE_EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

AUDIO_EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprised": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}

# -----------------------------
# Load models
# -----------------------------
print("Loading audio model...")
audio_classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Shared state
audio_label = {"value": "neutral"}
audio_lock = threading.Lock()


# -----------------------------
# Fusion Rule
# -----------------------------
def fuse_emotions(face, audio):
    if not face and not audio:
        return "neutral"
    if face == audio:
        return face or audio
    if face == "neutral" and audio and audio != "neutral":
        return audio
    if audio == "neutral" and face and face != "neutral":
        return face
    return audio or face


# -----------------------------
# Audio Worker (threaded)
# -----------------------------
def audio_worker():
    buffer = []
    last_pred_time = 0

    def callback(indata, frames, time_info, status):
        nonlocal buffer, last_pred_time
        if status:
            print("Audio status:", status)

        buffer.extend(indata[:, 0])
        now = time.time()

        # process every ~2.5 seconds
        if len(buffer) >= 32000 and (now - last_pred_time > 2.5):
            segment = np.array(buffer[:32000])
            del buffer[:16000]  # overlap
            try:
                preds = audio_classifier(segment, top_k=1)
                label = preds[0]["label"].lower()
                mapped = AUDIO_EMOTION_MAP.get(label, "neutral")
                with audio_lock:
                    audio_label["value"] = mapped
                last_pred_time = now
            except Exception as e:
                print("Audio error:", e)

    with sd.InputStream(channels=1, samplerate=16000, callback=callback):
        sd.sleep(int(1e12))  # keep alive


# -----------------------------
# Start audio thread
# -----------------------------
t = threading.Thread(target=audio_worker, daemon=True)
t.start()

# -----------------------------
# Video Loop
# -----------------------------
cap = cv2.VideoCapture(0)

frame_count = 0
face_label = "neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run face detection every 10th frame
    if frame_count % 10 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(
                    roi,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )
                face_label = FACE_EMOTION_MAP.get(result[0]["dominant_emotion"].lower(), "neutral")
            except Exception:
                pass

    # Get audio label safely
    with audio_lock:
        al = audio_label["value"]

    # Fuse
    final_emotion = fuse_emotions(face_label, al)

    # Display text
    cv2.putText(frame, f"Face: {face_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Speech: {al}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Final: {final_emotion}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Multimodal Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
