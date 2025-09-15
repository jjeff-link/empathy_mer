import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from transformers import pipeline
import sounddevice as sd

# -----------------------------
# Emotion Mappings (Ekman + Neutral) — Expanded for case and variants
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
    # Full words (lowercase)
    "angry": "anger",
    "disgust": "disgust",
    "fearful": "fear",
    "afraid": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprised": "surprise",
    "neutral": "neutral",
    "calm": "neutral",

    # Capitalized
    "Angry": "anger",
    "Disgust": "disgust",
    "Fearful": "fear",
    "Afraid": "fear",
    "Happy": "happiness",
    "Sad": "sadness",
    "Surprised": "surprise",
    "Neutral": "neutral",
    "Calm": "neutral",

    # Dataset abbreviations (IEMOCAP style)
    "ang": "anger",
    "hap": "happiness",
    "neu": "neutral",
    "sad": "sadness",
    "sur": "surprise",
    "fea": "fear",
    "dis": "disgust",
    "exc": "happiness",   # excited → happy
    "fru": "anger",       # frustrated → angry
    "oth": "neutral",     # other → neutral
    "xxx": "neutral",     # unknown → neutral
    "cal": "neutral",     # calm → neutral

    # Just in case
    "fear": "fear",
    "surprise": "surprise",
    "happiness": "happiness",
    "sadness": "sadness",
}

# -----------------------------
# Load models
# -----------------------------
print("Loading audio model...")

# 💡 Try this lighter/faster model if hubert-large is too slow or insensitive:
audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
# audio_classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

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

        buffer.extend(indata[:, 0])  # mono channel
        now = time.time()

        # Process every ~2.5 seconds if we have enough data
        if len(buffer) >= 32000 and (now - last_pred_time > 2.5):
            segment = np.array(buffer[:32000])
            del buffer[:16000]  # 50% overlap

            # 🔊 Check volume (RMS) — skip if too quiet
            rms = np.sqrt(np.mean(segment**2))
            if rms < 0.02:
                print(f"🔇 Too quiet (RMS: {rms:.4f}), skipping...")
                return

            try:
                preds = audio_classifier(segment, top_k=1)
                print("🎙️ Raw audio predictions:", preds)  # 👈 DEBUG: See what model ACTUALLY returns

                label = preds[0]["label"]  # Keep original case
                score = preds[0]["score"]
                mapped = AUDIO_EMOTION_MAP.get(label, "neutral")

                with audio_lock:
                    audio_label["value"] = mapped

                last_pred_time = now
                print(f"🎤 Audio Emotion: {mapped} (from '{label}', score: {score:.2f})")

            except Exception as e:
                print("Audio error:", e)

    # Start audio stream
    with sd.InputStream(channels=1, samplerate=16000, callback=callback):
        print("\n🎙️ Audio stream started.")
        print("🗣️  SPEAK WITH EMOTION! Try: 'I am so angry!' or 'I'm really happy!'")
        print("🔊 Make sure mic is not muted and volume is up.\n")
        while True:
            time.sleep(1)


# -----------------------------
# Start audio thread
# -----------------------------
t = threading.Thread(target=audio_worker, daemon=True)
t.start()

# -----------------------------
# Video Loop
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

frame_count = 0
face_label = "neutral"

print("📹 Starting video stream. Press 'q' to quit.")
print("💡 Tip: Speak loudly and clearly with emotion — e.g., 'I am so happy!' or 'This is terrifying!'\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_count += 1

    # Run face detection every 10th frame (to reduce CPU load)
    if frame_count % 10 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first face only
            roi = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(
                    roi,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )
                detected_emotion = result[0]["dominant_emotion"].lower()
                face_label = FACE_EMOTION_MAP.get(detected_emotion, "neutral")
                print(f"👁️ Face Emotion: {face_label}")
            except Exception as e:
                print("Face analysis error:", e)
                pass  # Ignore and reuse last face_label

    # Get latest audio emotion (thread-safe)
    with audio_lock:
        al = audio_label["value"]

    # Fuse face + audio emotion
    final_emotion = fuse_emotions(face_label, al)

    # Display on screen
    cv2.putText(frame, f"Face: {face_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Speech: {al}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Final: {final_emotion}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Multimodal Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
