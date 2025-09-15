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

# Mapping from various emotion labels to standardized Ekman emotions
EMOTION_MAPPING = {
    # DeepFace variants
    'angry': 'anger',
    'surprised': 'surprise',
    'fearful': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'disgusted': 'disgust',
    # RAVDESS variants
    'calm': 'neutral',
    'ps': 'surprise',
    # Standard forms
    'anger': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'surprise': 'surprise',
    'neutral': 'neutral',
    'happiness': 'happy'
}

# Ekman's six basic emotions plus neutral
EKMAN_EMOTIONS = {
    'anger': 0,    # anger
    'disgust': 1,  # disgust
    'fear': 2,     # fear
    'happy': 3,    # happiness
    'sad': 4,      # sadness
    'surprise': 5, # surprise
    'neutral': 6   # neutral (additional)
}

# List version for indexing
EMOTION_LABELS = list(EKMAN_EMOTIONS.keys())

# ======================
# LOAD MODELS (ONCE)
# ======================

# 1. FACIAL MODEL: DeepFace (auto-downloads on first run)
print("Loading facial emotion model (DeepFace/VGG-Face)...")

# Initialize DeepFace with configuration for reuse
deepface_config = {
    'detector_backend': 'opencv',
    'enforce_detection': False,
    'silent': True,
    'actions': ['emotion']
}

# Function to safely analyze frame
def deepface_analyzer(frame):
    try:
        # Ensure frame is valid
        if frame is None or not isinstance(frame, np.ndarray):
            print("Invalid frame format")
            return None
            
        # Ensure frame is BGR (DeepFace expects BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print("Invalid frame dimensions")
            return None
            
        # Save frame temporarily
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
            
        # Analyze with DeepFace using file path
        try:
            result = DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Process result
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            elif isinstance(result, dict):
                return result
            else:
                print("No valid face analysis result")
                return None
                
        except Exception as inner_e:
            print(f"DeepFace analysis error: {str(inner_e)}")
            return None
            
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return None
        
# Ensure models directory exists
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Warm up DeepFace with dummy frame
warmup_frame = np.zeros((224, 224, 3), dtype=np.uint8)  # Larger frame for better initialization
_ = deepface_analyzer(warmup_frame)

# 2. SPEECH MODEL: Wav2Vec 2.0 for Speech Emotion Recognition
print("Loading Wav2Vec 2.0 Speech Emotion Recognition model...")
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import Wav2Vec2Model

# Load the pre-trained speech emotion recognition model and processor
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
speech_model = AutoModelForAudioClassification.from_pretrained(model_name, output_hidden_states=True)
speech_model.eval()  # Set to inference mode

# Load base Wav2Vec2 model for embeddings
base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
base_model.eval()

# Define emotion mapping from RAVDESS (used in speech model) to Ekman
id2label = speech_model.config.id2label
SPEECH_EMOTION_LABELS = list(id2label.values())

# Use the unified EMOTION_MAPPING for all conversions
print("Initializing emotion recognition models:")

print("Initializing emotion recognition models:")
print(f"- Speech (RAVDESS): {SPEECH_EMOTION_LABELS}")
print(f"- Unified (Ekman): {EMOTION_LABELS}")

# 4. MULTIMODAL FUSION: Weighted Average Based on Confidence (Simple & Robust)
class MultimodalFusion(nn.Module):
    def __init__(self, emotion_dim=len(EKMAN_EMOTIONS)):
        super(MultimodalFusion, self).__init__()
        self.emotion_dim = emotion_dim
        # Learnable weights for face and speech (initialized to equal importance)
        self.face_weight = nn.Parameter(torch.tensor([0.5]))
        self.speech_weight = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, face_feat, speech_feat):
        # Ensure inputs are [B, 7]
        if face_feat.dim() == 1:
            face_feat = face_feat.unsqueeze(0)
        if speech_feat.dim() == 1:
            speech_feat = speech_feat.unsqueeze(0)
            
        # Convert to float32
        face_feat = face_feat.float()
        speech_feat = speech_feat.float()
            
        # Normalize weights to sum to 1
        total_weight = self.face_weight + self.speech_weight
        w_face = self.face_weight / total_weight
        w_speech = self.speech_weight / total_weight
        
        # Weighted average of emotion distributions
        fused_dist = w_face * face_feat + w_speech * speech_feat
        
        # Return normalized distribution
        return fused_dist / fused_dist.sum(dim=1, keepdim=True)

# Initialize fusion model
fusion_model = MultimodalFusion()

# Save dummy weights (this will be learned during training — but for demo, we use equal weights)
torch.save(fusion_model.state_dict(), "models/fusion_weights.pth")
fusion_model.load_state_dict(torch.load("models/fusion_weights.pth", map_location='cpu'))
fusion_model.eval()
print("Loaded simple weighted fusion model.")

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
                # Get face analysis result (now returns single dict or None)
                result_dict = deepface_analyzer(frame)
                
                if result_dict is None:
                    raise ValueError("No face detected or invalid frame")
                
                if not isinstance(result_dict, dict):
                    raise ValueError("Invalid result format")
                
                # Extract emotion data with defaults
                dominant_emotion = result_dict.get('dominant_emotion', 'neutral')
                emotion_scores = result_dict.get('emotion', {})
                
                # Ensure we have valid emotion scores
                if not emotion_scores:
                    raise ValueError("No emotion scores available")
                
                # Safely get confidence
                face_conf = emotion_scores.get(dominant_emotion, 0.0) / 100.0
                
                # Map to Ekman emotion
                face_emotion = EMOTION_MAPPING.get(dominant_emotion, 'neutral')
                
                # Create continuous feature vector from emotion scores
                face_feat = np.zeros(len(EKMAN_EMOTIONS))
                for emotion, score in emotion_scores.items():
                    mapped_emotion = EMOTION_MAPPING.get(emotion, 'neutral')
                    idx = EKMAN_EMOTIONS.get(mapped_emotion, EKMAN_EMOTIONS['neutral'])
                    face_feat[idx] = score / 100.0  # Normalize to [0,1]
                    
                face_tensor = torch.tensor(face_feat.astype(np.float32))
                
            except Exception as e:
                # print(f"Face analysis error: {str(e)}")  # Uncomment for debugging
                face_feat = np.zeros(len(EKMAN_EMOTIONS))
                face_feat[EKMAN_EMOTIONS['neutral']] = 1.0  # Default to neutral
                face_tensor = torch.tensor(face_feat.astype(np.float32))
                face_conf = 0.0
                dominant_emotion = "no_face"

            # 2. SPEECH EMOTION PREDICTION using fine-tuned Wav2Vec2
            try:
                with torch.no_grad():
                    # Process audio with the feature extractor
                    inputs = feature_extractor(
                        audio_chunk, 
                        sampling_rate=SAMPLE_RATE, 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    if inputs is None:
                        raise ValueError("Feature extraction failed")
                    
                    # Get emotion predictions
                    outputs = speech_model(**inputs)
                    speech_logits = outputs.logits
                    speech_probs = torch.nn.functional.softmax(speech_logits, dim=-1)
                    
                    # Create emotion distribution tensor matching Ekman emotions
                    speech_feat = torch.zeros(len(EKMAN_EMOTIONS))
                    
                    # Map RAVDESS probabilities to Ekman emotions
                    for idx, ravdess_emotion in id2label.items():
                        ekman_emotion = EMOTION_MAPPING.get(ravdess_emotion, 'neutral')
                        ekman_idx = EKMAN_EMOTIONS.get(ekman_emotion, EKMAN_EMOTIONS['neutral'])
                        speech_feat[ekman_idx] += speech_probs[0, idx].item()
                    
                    # Normalize to ensure sum to 1
                    speech_feat = speech_feat / speech_feat.sum() if speech_feat.sum() > 0 else speech_feat
                    
                    # Use this as our speech embedding for fusion
                    speech_embedding_for_fusion = speech_feat.unsqueeze(0)  # Shape: [1, 7]
                    
                    # Get prediction and confidence for display
                    speech_pred_idx = torch.argmax(speech_logits, dim=-1).item()
                    speech_conf = torch.max(speech_probs).item()
                    
                    # Map emotion for display
                    ravdess_emotion = id2label.get(speech_pred_idx, 'neutral')
                    speech_emotion = EMOTION_MAPPING.get(ravdess_emotion, 'neutral')
                        
                    # Map the emotion for display purposes
                    ravdess_emotion = id2label.get(speech_pred_idx, 'neutral')
                    speech_emotion = EMOTION_MAPPING.get(ravdess_emotion, 'neutral')
                    
            except Exception as e:
                print(f"Speech processing error: {str(e)}")
                # Fallback to neutral emotion distribution
                speech_feat = torch.zeros(len(EKMAN_EMOTIONS))
                speech_feat[EKMAN_EMOTIONS['neutral']] = 1.0  # All probability on neutral
                speech_embedding_for_fusion = speech_feat.unsqueeze(0)  # Shape: [1, 7]
                speech_conf = 0.0
                speech_emotion = 'neutral'

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