import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from transformers import pipeline
import sounddevice as sd
from collections import deque
import scipy.signal

# -----------------------------
# Performance Configuration
# -----------------------------
CONFIG = {
    "face_backend": "opencv",  # "opencv" (fast) or "retinaface" (accurate)
    "face_detection_interval": 0.5,  # seconds between face detections
    "audio_processing_interval": 2.0,  # Reduced to 2 seconds for more responsive audio
    "use_gpu": True,  # Use GPU if available
    "resize_for_face_detection": True,  # Resize frames for faster face detection
    "adaptive_frame_skip": True,  # Skip frames when FPS is low
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "audio_debug": True,  # Enable detailed audio debugging
    "audio_enhancement": False,  # Disable audio preprocessing to avoid filter issues
    "simple_mode": True,  # Use simpler logic for testing
}

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
    "anger": "anger",
    "disgust": "disgust", 
    "fear": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}

# Audio confidence thresholds to improve accuracy
AUDIO_CONFIDENCE_THRESHOLD = 0.1  # Much lower threshold to accept more predictions
EMOTION_SIMILARITY_MAP = {
    # Map similar emotions that might be confused
    ("anger", "sadness"): "anger",  # Prioritize anger when confused with sadness
    ("sadness", "anger"): "anger",  # Same for reverse
    ("fear", "sadness"): "sadness", # Prioritize sadness when confused with fear
    ("sadness", "fear"): "sadness", # Same for reverse
    ("happiness", "neutral"): "happiness",  # Prioritize happiness over neutral
    ("neutral", "happiness"): "happiness",  # Same for reverse
}

# Audio preprocessing functions
def preprocess_audio(audio_segment):
    """Enhanced audio preprocessing for better emotion recognition"""
    if not CONFIG["audio_enhancement"]:
        return audio_segment
    
    # Check if audio has any signal
    if np.max(np.abs(audio_segment)) < 1e-6:
        return audio_segment
    
    # Normalize audio (less aggressive)
    max_val = np.max(np.abs(audio_segment))
    audio_segment = audio_segment / (max_val + 1e-8)
    
    # Apply simple high-pass filter instead of bandpass to preserve more signal
    # Remove very low frequencies but keep most of the signal
    if len(audio_segment) > 100:
        try:
            nyquist = 16000 / 2.0  # Use float division
            low_freq = 50.0  # Hz
            normalized_freq = low_freq / nyquist
            
            # Ensure frequency is in valid range (0 < Wn < 1)
            if 0 < normalized_freq < 1:
                b, a = scipy.signal.butter(2, normalized_freq, btype='high')
                audio_segment = scipy.signal.filtfilt(b, a, audio_segment)
            else:
                # Skip filtering if frequency is invalid
                pass
        except Exception as e:
            # If filtering fails, just return the normalized audio
            print(f"Filter warning: {e}")
    
    return audio_segment

# -----------------------------
# Load models
# -----------------------------
print("=== Performance Configuration ===")
for key, value in CONFIG.items():
    print(f"{key}: {value}")
print("=================================")

print("Loading speech emotion model (wav2vec2)...")
# Using a smaller, faster model for better performance
# You can switch back to the original for better accuracy if needed
try:
    use_gpu = CONFIG["use_gpu"] and __import__("torch").cuda.is_available()
    audio_classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=0 if use_gpu else -1
    )
    print(f"Audio model loaded on {'GPU' if use_gpu else 'CPU'}")
except:
    print("Failed to load on GPU, falling back to CPU")
    audio_classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=-1
    )

print("Loading face detector...")
# Use configurable backend for balance between speed and accuracy
FACE_BACKEND = CONFIG["face_backend"]

# Pre-load DeepFace models to avoid segmentation faults
print("Initializing DeepFace models (this may take a moment)...")
try:
    # Initialize with a small test image to pre-load models
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = DeepFace.analyze(
        test_image,
        actions=["emotion"],
        detector_backend=FACE_BACKEND,
        enforce_detection=False,
        silent=True
    )
    print(f"DeepFace initialized successfully with {FACE_BACKEND} backend")
except Exception as e:
    print(f"Error initializing DeepFace with {FACE_BACKEND}, trying opencv backend...")
    FACE_BACKEND = "opencv"
    try:
        _ = DeepFace.analyze(
            test_image,
            actions=["emotion"],
            detector_backend=FACE_BACKEND,
            enforce_detection=False,
            silent=True
        )
        print(f"DeepFace initialized successfully with opencv backend")
    except Exception as e2:
        print(f"Failed to initialize DeepFace: {e2}")
        print("Face detection will be disabled")

# Performance tracking
performance_stats = {
    "frame_count": 0,
    "face_detection_time": deque(maxlen=50),
    "audio_processing_time": deque(maxlen=50),
    "fps": 0,
    "last_fps_time": time.time()
}

# Shared state
audio_label = {"value": "neutral", "confidence": 0.0, "raw_predictions": [], "audio_stats": {}}
audio_lock = threading.Lock()

# Face tracking variables
last_face_detection = 0
face_detection_interval = CONFIG["face_detection_interval"]
last_known_face_bbox = None
skip_frame_count = 0  # For adaptive frame skipping
current_face_region = None  # Store current face bounding box

# -----------------------------
# Improved Fusion Rule with confidence
# -----------------------------
def fuse_emotions(face, audio, audio_confidence=0.0):
    if not face and not audio:
        return "neutral"
    
    # If audio confidence is low, trust face more
    if audio_confidence < AUDIO_CONFIDENCE_THRESHOLD and face:
        return face
    
    if face == audio:
        return face or audio
    
    # Handle similar emotion confusion
    if face and audio:
        emotion_pair = (face, audio)
        if emotion_pair in EMOTION_SIMILARITY_MAP:
            return EMOTION_SIMILARITY_MAP[emotion_pair]
    
    if face == "neutral" and audio and audio != "neutral":
        return audio
    if audio == "neutral" and face and face != "neutral":
        return face
    
    # Prioritize audio when confidence is high
    if audio_confidence > 0.7:
        return audio
        
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

        # Process based on configurable interval
        if len(buffer) >= 32000 and (now - last_pred_time > CONFIG["audio_processing_interval"]):
            start_time = time.time()
            segment = np.array(buffer[:32000])
            del buffer[:16000]  # More overlap for better analysis
            
            try:
                # Enhanced preprocessing (if enabled)
                try:
                    if CONFIG["audio_enhancement"]:
                        processed_segment = preprocess_audio(segment)
                    else:
                        # Simple normalization only
                        max_val = np.max(np.abs(segment))
                        processed_segment = segment / (max_val + 1e-8) if max_val > 1e-8 else segment
                except Exception as filter_error:
                    print(f"Audio preprocessing error: {filter_error}")
                    # Fallback to simple normalization
                    max_val = np.max(np.abs(segment))
                    processed_segment = segment / (max_val + 1e-8) if max_val > 1e-8 else segment
                
                # Calculate audio statistics for debugging
                audio_volume = np.sqrt(np.mean(processed_segment**2))
                audio_energy = np.sum(processed_segment**2)
                
                # Skip if audio is too quiet (no actual speech)
                if audio_volume < 0.001:
                    if CONFIG["audio_debug"]:
                        print(f"Audio too quiet: {audio_volume:.6f}")
                    # Don't update audio label if too quiet
                    processing_time = time.time() - start_time
                    performance_stats["audio_processing_time"].append(processing_time)
                    last_pred_time = now
                    return
                
                # Get predictions
                preds = audio_classifier(processed_segment, top_k=5)  # Get top 5 for better analysis
                
                if CONFIG["audio_debug"]:
                    print(f"\n=== AUDIO DEBUG ===")
                    print(f"Volume: {audio_volume:.4f}, Energy: {audio_energy:.2f}")
                    print(f"All predictions:")
                    for i, pred in enumerate(preds):
                        emotion = AUDIO_EMOTION_MAP.get(pred["label"].lower(), pred["label"])
                        print(f"  {i+1}. {pred['label']} -> {emotion} (confidence: {pred['score']:.3f})")
                
                if CONFIG["simple_mode"]:
                    # Simple mode: just use the top prediction with very low threshold
                    top_pred = preds[0]
                    mapped = AUDIO_EMOTION_MAP.get(top_pred["label"].lower(), "neutral")
                    confidence = top_pred["score"]
                    
                    # If top is neutral, try the second prediction
                    if mapped == "neutral" and len(preds) > 1:
                        second_pred = preds[1]
                        if second_pred["score"] > 0.1:  # Very low threshold
                            mapped = AUDIO_EMOTION_MAP.get(second_pred["label"].lower(), "neutral")
                            confidence = second_pred["score"]
                            if CONFIG["audio_debug"]:
                                print(f"SIMPLE MODE: Using second prediction {mapped}")
                
                else:
                    # Complex mode (original logic)
                    # Get the top prediction
                    top_pred = preds[0]
                    confidence = top_pred["score"]
                    label = top_pred["label"].lower()
                    mapped = AUDIO_EMOTION_MAP.get(label, "neutral")
                    
                    # If top prediction is neutral but there are other emotions, consider them
                    if mapped == "neutral" and len(preds) > 1:
                        for pred in preds[1:3]:  # Check next 2 predictions
                            pred_label = pred["label"].lower()
                            pred_mapped = AUDIO_EMOTION_MAP.get(pred_label, "neutral")
                            if pred_mapped != "neutral" and pred["score"] > 0.15:  # Very low threshold
                                mapped = pred_mapped
                                confidence = pred["score"]
                                if CONFIG["audio_debug"]:
                                    print(f"OVERRIDING NEUTRAL: Using {pred_mapped} instead")
                                break
                    
                    # Enhanced emotion analysis - look at multiple predictions
                    emotion_scores = {}
                    for pred in preds[:5]:
                        pred_label = pred["label"].lower()
                        mapped_emotion = AUDIO_EMOTION_MAP.get(pred_label, "neutral")
                        if mapped_emotion not in emotion_scores:
                            emotion_scores[mapped_emotion] = 0
                        emotion_scores[mapped_emotion] += pred["score"]
                    
                    # Find non-neutral emotion with highest combined score
                    non_neutral_emotions = {k: v for k, v in emotion_scores.items() if k != "neutral"}
                    if non_neutral_emotions:
                        best_emotion = max(non_neutral_emotions, key=non_neutral_emotions.get)
                        best_score = non_neutral_emotions[best_emotion]
                        
                        # Use non-neutral emotion if it has reasonable score
                        if best_score > 0.2 and (mapped == "neutral" or best_score > confidence):
                            if CONFIG["audio_debug"]:
                                print(f"PREFERRING NON-NEUTRAL: {best_emotion} (score: {best_score:.3f})")
                            mapped = best_emotion
                            confidence = best_score
                    
                    if CONFIG["audio_debug"]:
                        print(f"Emotion scores: {emotion_scores}")
                
                # Always show final decision
                if CONFIG["audio_debug"]:
                    print(f"Final decision: {mapped} (conf: {confidence:.3f})")
                
                with audio_lock:
                    audio_label["value"] = mapped
                    audio_label["confidence"] = confidence
                    audio_label["raw_predictions"] = [(p["label"], p["score"]) for p in preds[:5]]
                    audio_label["audio_stats"] = {
                        "volume": audio_volume,
                        "energy": audio_energy,
                    }
                
                # Track performance
                processing_time = time.time() - start_time
                performance_stats["audio_processing_time"].append(processing_time)
                last_pred_time = now
                
                if CONFIG["audio_debug"]:
                    print(f"FINAL RESULT: {mapped} (conf: {confidence:.3f})")
                    print("==================\n")
                    
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

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["camera_fps"])

frame_count = 0
face_label = "neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # Adaptive frame skipping based on performance
    if CONFIG["adaptive_frame_skip"] and performance_stats["fps"] < 15 and skip_frame_count < 2:
        skip_frame_count += 1
        continue
    else:
        skip_frame_count = 0
    
    # Update FPS counter
    if current_time - performance_stats["last_fps_time"] >= 1.0:
        performance_stats["fps"] = performance_stats["frame_count"]
        performance_stats["frame_count"] = 0
        performance_stats["last_fps_time"] = current_time
    performance_stats["frame_count"] += 1

    # Run face detection based on time interval instead of frame count
    if current_time - last_face_detection >= face_detection_interval:
        face_detection_start = time.time()
        try:
            # Optionally resize frame for faster processing
            if CONFIG["resize_for_face_detection"]:
                small_frame = cv2.resize(frame, (320, 240))
                analysis_frame = small_frame
                scale_x = frame.shape[1] / 320  # Scale factor for x coordinates
                scale_y = frame.shape[0] / 240  # Scale factor for y coordinates
            else:
                analysis_frame = frame
                scale_x = scale_y = 1.0
                
            result = DeepFace.analyze(
                analysis_frame,
                actions=["emotion"],
                detector_backend=FACE_BACKEND,
                enforce_detection=False,
                silent=True
            )
            
            face_label = FACE_EMOTION_MAP.get(
                result[0]["dominant_emotion"].lower(), "neutral"
            )
            
            # Extract face region information if available
            if "region" in result[0]:
                region = result[0]["region"]
                # Scale coordinates back to original frame size if we resized
                x = int(region["x"] * scale_x)
                y = int(region["y"] * scale_y)
                w = int(region["w"] * scale_x)
                h = int(region["h"] * scale_y)
                current_face_region = (x, y, w, h)
            else:
                current_face_region = None
                
            last_face_detection = current_time
            
            # Track performance
            detection_time = time.time() - face_detection_start
            performance_stats["face_detection_time"].append(detection_time)
            
        except Exception as e:
            # print("Face error:", e)  # Reduce console spam
            face_label = "neutral"
            current_face_region = None
            last_face_detection = current_time

    # Get audio label safely
    with audio_lock:
        al = audio_label["value"]
        audio_conf = audio_label["confidence"]
        audio_stats = audio_label.get("audio_stats", {})

    # Fuse with confidence information
    final_emotion = fuse_emotions(face_label, al, audio_conf)

    # Draw face bounding box if detected
    if current_face_region is not None:
        x, y, w, h = current_face_region
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add small label above the box
        cv2.putText(frame, "FACE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display text with performance info
    cv2.putText(frame, f"Face: {face_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Speech: {al} ({audio_conf:.2f})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Final: {final_emotion}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Audio debugging info on screen
    if audio_stats:
        volume = audio_stats.get("volume", 0)
        cv2.putText(frame, f"Vol: {volume:.3f}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show top emotion scores
        emotion_scores = audio_stats.get("emotion_scores", {})
        y_offset = 150
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            cv2.putText(frame, f"{emotion}: {score:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
    
    # Performance display
    cv2.putText(frame, f"FPS: {performance_stats['fps']}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if performance_stats["face_detection_time"]:
        avg_face_time = np.mean(performance_stats["face_detection_time"]) * 1000
        cv2.putText(frame, f"Face Det: {avg_face_time:.1f}ms", (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if performance_stats["audio_processing_time"]:
        avg_audio_time = np.mean(performance_stats["audio_processing_time"]) * 1000
        cv2.putText(frame, f"Audio: {avg_audio_time:.1f}ms", (10, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'd' to toggle debug, 's' for simple mode, 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Multimodal Emotion Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        CONFIG["audio_debug"] = not CONFIG["audio_debug"]
        print(f"Audio debug: {'ON' if CONFIG['audio_debug'] else 'OFF'}")
    elif key == ord("s"):
        CONFIG["simple_mode"] = not CONFIG["simple_mode"]
        print(f"Simple mode: {'ON' if CONFIG['simple_mode'] else 'OFF'}")
    elif key == ord("e"):
        CONFIG["audio_enhancement"] = not CONFIG["audio_enhancement"]
        print(f"Audio enhancement: {'ON' if CONFIG['audio_enhancement'] else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
