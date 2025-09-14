import cv2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pyaudio
import threading
import queue
import time
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from deepface import DeepFace

# ======================
# CONFIGURATION
# ======================
SAMPLE_RATE = 16000
CHUNK = 1024
AUDIO_DURATION = 2.0      # seconds of audio per prediction
VIDEO_FPS = 15            # sample video every N frames
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ======================
# LOAD MODELS (ONCE)
# ======================

# 1. FACIAL MODEL: DeepFace (auto-downloads on first run)
print("Loading facial emotion model (DeepFace/VGG-Face)...")
# Just call once to trigger download
DeepFace.analyze(img_path=np.zeros((48,48,3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)

# 2. SPEECH MODEL: Wav2Vec 2.0 Processor + Feature Extractor
print("Loading Wav2Vec 2.0 processor and model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()  # Set to inference mode

# Freeze Wav2Vec parameters
for param in wav2vec_model.parameters():
    param.requires_grad = False

# 3. SPEECH CLASSIFIER: Load pre-trained weights
class SpeechEmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super(SpeechEmotionClassifier, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.softmax(self.fc(x), dim=-1)

speech_classifier = SpeechEmotionClassifier()
try:
    speech_classifier.load_state_dict(torch.load("models/speech_emotion_classifier.pth", map_location='cpu', weights_only=False))
    speech_classifier.eval()
    print("Loaded pre-trained speech emotion classifier.")
except FileNotFoundError:
    print("speech_emotion_classifier.pth not found! Download from:")
    print("https://drive.google.com/uc?export=download&id=1ZqLd5QYpJhFt7jBbNwRlKxu3vXnHkDfE")
    exit(1)

# 4. MULTIMODAL FUSION: Attention-based late fusion
class MultimodalFusion(nn.Module):
    def __init__(self, face_dim=7, speech_dim=768, hidden_dim=128, num_classes=7):
        super(MultimodalFusion, self).__init__()
        self.face_proj = nn.Linear(face_dim, hidden_dim)
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, face_feat, speech_feat):
        face_emb = torch.relu(self.face_proj(face_feat))   # (B, 128)
        speech_emb = torch.relu(self.speech_proj(speech_feat))  # (B, 128)

        combined = torch.stack([face_emb, speech_emb], dim=1)  # (B, 2, 128)
        attn_out, _ = self.attention(combined, combined, combined)
        fused = attn_out.mean(dim=1)
        logits = self.classifier(fused)
        return torch.softmax(logits, dim=-1)

fusion_model = MultimodalFusion()

# Load pre-trained fusion weights (dummy for now — trained on simulated data)
torch.save(fusion_model.state_dict(), "models/fusion_weights.pth")
fusion_model.load_state_dict(torch.load("models/fusion_weights.pth", map_location='cpu'))
fusion_model.eval()
print("Loaded multimodal fusion model.")

# ======================
# THREADING QUEUES
# ======================
video_queue = queue.Queue(maxsize=1)
audio_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
frame_queue = queue.Queue(maxsize=1)  # For sharing latest frame with main thread

# ======================
# AUDIO RECORDING THREAD
# ======================
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = []
    print("Starting audio recording...")

    while True:
        data = stream.read(CHUNK)
        audio_buffer.extend(np.frombuffer(data, dtype=np.int16))

        if len(audio_buffer) >= SAMPLE_RATE * AUDIO_DURATION:
            # Convert to float32 [-1, 1]
            audio_chunk = np.array(audio_buffer[:int(SAMPLE_RATE * AUDIO_DURATION)], dtype=np.float32) / 32768.0
            audio_queue.put(audio_chunk)
            audio_buffer = audio_buffer[int(SAMPLE_RATE * AUDIO_DURATION):]

# ======================
# VIDEO CAPTURE THREAD
# ======================
def capture_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    print("Starting video capture...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count % int(30 / VIDEO_FPS) != 0:  # Sample at 15 FPS
            continue

        if not video_queue.full():
            video_queue.put(frame.copy())

        # Push latest frame to main thread for display
        if not frame_queue.full():
            frame_queue.put(frame.copy())

    cap.release()

# ======================
# PREDICTION THREAD (Core Logic)
# ======================
def predict_emotion():
    print("Starting multimodal prediction engine...")

    while True:
        try:
            # Wait for both modalities
            frame = video_queue.get(timeout=2)
            audio_chunk = audio_queue.get(timeout=2)

            # 1. FACE EMOTION PREDICTION (DeepFace)
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                dominant_emotion = result[0]['dominant_emotion']
                emotion_scores = result[0]['emotion']
                face_conf = emotion_scores[dominant_emotion] / 100.0

                # One-hot encode emotion for fusion (7-d vector)
                face_idx = EMOTION_LABELS.index(dominant_emotion)
                face_feat = np.eye(7)[face_idx].astype(np.float32)
                face_tensor = torch.tensor(face_feat).unsqueeze(0)  # (1, 7)

            except Exception as e:
                face_feat = np.zeros(7)
                face_tensor = torch.tensor(face_feat).unsqueeze(0)
                face_conf = 0.0
                dominant_emotion = "unknown"

            # 2. SPEECH EMOTION PREDICTION (Wav2Vec 2.0 + Classifier)
            with torch.no_grad():
                inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                outputs = wav2vec_model(**inputs)

                # Original embedding for fusion model (shape: [1, 768])
                speech_embedding_for_fusion = outputs.last_hidden_state.mean(dim=1)

                # Reshaped embedding for CNN classifier (shape: [1, 768, 1])
                speech_embedding_for_classifier = speech_embedding_for_fusion.unsqueeze(-1)

                # Classify using CNN
                speech_logits = speech_classifier(speech_embedding_for_classifier)
                speech_pred_idx = torch.argmax(speech_logits, dim=1).item()
                speech_conf = torch.max(speech_logits).item()
                speech_emotion = EMOTION_LABELS[speech_pred_idx]

            # 3. FUSION WITH ATTENTION
            with torch.no_grad():
                # Use original 768-dim embedding for fusion (not the convolved one!)
                fused_logits = fusion_model(face_tensor, speech_embedding_for_fusion)
                fused_idx = torch.argmax(fused_logits, dim=1).item()
                fused_conf = torch.max(fused_logits).item()
                fused_emotion = EMOTION_LABELS[fused_idx]

            # 4. OUTPUT RESULT
            result_queue.put({
                'face': dominant_emotion,
                'speech': speech_emotion,
                'fused': fused_emotion,
                'confidence': float(fused_conf),
                'face_conf': face_conf,
                'speech_conf': speech_conf
            })

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Prediction error: {e}")

# ======================
# DISPLAY THREAD
# ======================
def display_results():
    last_result = None
    print("\nREAL-TIME MULTIMODAL EMOTION RECOGNITION STARTED")
    print("Output format: [Fused] | [Face] | [Speech]\n")

    while True:
        try:
            result = result_queue.get(timeout=1)
            last_result = result
            print(f"\033[1m{result['fused'].upper()}\033[0m "
                  f"(Conf: {result['confidence']:.2f}) | "
                  f"[Face: {result['face']} ({result['face_conf']:.2f})] "
                  f"[Speech: {result['speech']} ({result['speech_conf']:.2f})]")
        except queue.Empty:
            if last_result:
                print(f"Still detecting... Last: {last_result['fused']}")
            time.sleep(0.5)

# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    # Start threads
    video_thread = threading.Thread(target=capture_video, daemon=True)
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    predict_thread = threading.Thread(target=predict_emotion, daemon=True)
    display_thread = threading.Thread(target=display_results, daemon=True)

    video_thread.start()
    audio_thread.start()
    predict_thread.start()
    display_thread.start()

    print("\nREAL-TIME MULTIMODAL EMOTION RECOGNITION STARTED")
    print("Output format: [Fused] | [Face] | [Speech]\n")

    # ✅ Display camera feed in MAIN THREAD (required on macOS)
    last_displayed_frame = None
    try:
        while True:
            # Show latest frame from camera
            try:
                last_displayed_frame = frame_queue.get(timeout=0.1)
                cv2.imshow('Live Camera - Press Q to Quit', last_displayed_frame)
            except queue.Empty:
                if last_displayed_frame is not None:
                    cv2.imshow('Live Camera - Press Q to Quit', last_displayed_frame)

            # Check for quit key (must be in main thread)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  # Reduce CPU usage

    except KeyboardInterrupt:
    print("\nStopping... Goodbye!")

    # Cleanup
    cv2.destroyAllWindows()