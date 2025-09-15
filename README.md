# Multimodal Emotion Recognition (MER) Prototype

A high-performance **real-time multimodal emotion recognition system** that combines computer vision and speech processing to detect human emotions across seven categories: *Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral*.

##  Core Models & Architecture

###  **System Overview**
```
📹 Video Input (Webcam) ──┐
                          ├──► 🤖 Fusion Engine ──► 😊 Final Emotion
🎤 Audio Input (Mic) ─────┘
```

### 🔬 **Model Components**

## 1. 👁️ **Computer Vision Pipeline**

### **Face Detection Models**
| Model | Backend | Performance | Accuracy | Use Case |
|-------|---------|-------------|----------|----------|
| **OpenCV Haar Cascades** | `opencv` | ⚡ Fast | 🔵 Good | Real-time applications |
| **RetinaFace** | `retinaface` | 🐌 Slower | 🟢 Excellent | High-accuracy scenarios |

### **Facial Emotion Recognition**
- **Framework**: [DeepFace](https://github.com/serengil/deepface) v0.0.93
- **Model Architecture**: Convolutional Neural Networks (CNN)
- **Training Data**: FER2013, VGGFace2, and other emotion datasets
- **Output Classes**: 7 emotions (Ekman model + Neutral)
- **Processing**: Real-time face detection with emotion classification

**Technical Details:**
```python
# DeepFace Model Configuration
FACE_BACKEND = "opencv"  # or "retinaface" 
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
```

## 2. 🎵 **Speech Emotion Recognition Pipeline**

### **Audio Model: wav2vec2-lg-xlsr-en-speech-emotion-recognition**
- **Source**: [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- **Base Architecture**: wav2vec2.0 Large XLSR-53
- **Framework**: Hugging Face Transformers
- **Model Type**: Self-supervised speech representation learning
- **Language**: English-optimized
- **Sampling Rate**: 16kHz

**Model Specifications:**
```python
# Audio Model Configuration
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
ARCHITECTURE = "Wav2Vec2ForSequenceClassification"
PARAMETERS = "Large XLSR-53 (300M+ parameters)"
```

### **Audio Processing Components**
| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **SoundDevice** | Real-time audio capture | 16kHz sampling, mono channel |
| **Signal Processing** | Audio preprocessing | Scipy filters, normalization |
| **Feature Extraction** | wav2vec2 embeddings | Self-supervised representations |
| **Classification** | Emotion prediction | Transformer head with softmax |

## 3. 🔄 **Fusion Architecture**

### **Multimodal Fusion Strategy**
- **Method**: Confidence-weighted late fusion
- **Logic**: Intelligent emotion combination with similarity mapping
- **Fallback**: Modality-specific predictions when confidence is low

```python
# Fusion Algorithm
def fuse_emotions(face_emotion, audio_emotion, audio_confidence):
    # 1. Confidence-based weighting
    # 2. Emotion similarity mapping  
    # 3. Anti-neutral bias
    # 4. Adaptive thresholds
    return final_emotion
```

## 🏗️ **System Architecture**

### **Processing Pipeline**
```
┌─────────────────┐    ┌─────────────────┐
│   Video Thread  │    │   Audio Thread  │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  OpenCV/    │ │    │ │ SoundDevice │ │
│ │ RetinaFace  │ │    │ │  Capture    │ │
│ └─────────────┘ │    │ └─────────────┘ │
│        │        │    │        │        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  DeepFace   │ │    │ │ wav2vec2    │ │
│ │ Emotion CNN │ │    │ │ Transformer │ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┐       ┌───────┘
                 │       │
        ┌─────────────────┐
        │ Fusion Engine   │
        │ - Confidence    │
        │ - Similarity    │ 
        │ - Temporal      │
        └─────────────────┘
                 │
        ┌─────────────────┐
        │ Final Emotion   │
        │   Prediction    │
        └─────────────────┘
```

## 📋 **Model Performance Specifications**

### **Face Detection Performance**
| Backend | Speed (FPS) | Accuracy | Memory | Best For |
|---------|-------------|----------|---------|----------|
| OpenCV | 25-30 | 85% | Low | Real-time |
| RetinaFace | 10-15 | 95% | High | Accuracy |

### **Audio Model Performance**
| Metric | Value | Notes |
|--------|-------|--------|
| **Model Size** | ~300MB | Large transformer model |
| **Processing Time** | 200-500ms | Per 2-second audio segment |
| **Accuracy** | ~80-85% | On standard emotion datasets |
| **Languages** | English | Optimized for English speech |



This is a simple **multimodal emotion recognition prototype** that combines **facial expressions** (from webcam) and **speech prosody** (from microphone) to predict a user’s emotional state in real time.


## ⚙️ Installation

### 1. Clone repository
```bash
git clone https://github.com/jjeff-link/empathy_mer.git
cd empathy_mer
```

### 2. Create Virtual Environment
```bash
python3 -m venv multimodal-env
source multimodal-env/bin/activate   # macOS/Linux
multimodal-env\Scripts\activate      # Windows (PowerShell)
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the prototype:
```bash
python mer.py
```
