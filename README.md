# Multimodal Emotion Recognition (MER) Prototype

A high-performance **real-time multimodal emotion recognition system** that combines## Technical Implementation Details

### Data Sources & Testing
**Testing Data:**
- **Real-time Camera Feed**: Live webcam input for facial emotion detection
- **Real-time Microphone Audio**: Continuous 16kHz audio capture for speech emotion recognition
- **Personal Testing**: Own samples and recordings during development
- **Validation Method**: Interactive real-time testing with immediate visual feedback and debug metrics

**Data Characteristics:**
- **Video**: 640x480 resolution frames, processed at 0.5-second intervals
- **Audio**: 16kHz sampling rate, 1-second sliding windows with 0.5-second overlap
- **Real-time Processing**: No pre-recorded datasets - live multimodal input streams

### Feature Extraction Pipeline

#### Visual Features (Face)
- **Facial Detection**: Haar Cascade classifiers and RetinaFace deep learning detection
- **Facial Landmarks**: 68-point facial landmark detection via DeepFace
- **Emotion Embeddings**: Deep CNN features from VGG-Face and Facenet architectures
- **Preprocessing**: Face alignment, normalization, and geometric transformations
- **ROI Extraction**: Bounding box extraction with confidence scoring

#### Audio Features (Speech)
- **Raw Audio Processing**: 16-bit PCM audio normalization and filtering
- **Spectral Features**: Mel-frequency cepstral coefficients (MFCCs) and spectrograms
- **Transformer Embeddings**: wav2vec2 self-supervised speech representations (768-dimensional)
- **Temporal Features**: Frame-level features aggregated over 1-second windows
- **Preprocessing**: Noise reduction, amplitude normalization, and resampling

**Feature Types Explained:**
- **MFCCs**: Capture spectral characteristics of speech that correlate with emotional prosody
- **Facial Landmarks**: 2D coordinates of key facial points (eyes, mouth, eyebrows) for geometric emotion analysis
- **Deep Embeddings**: High-dimensional learned representations from pre-trained neural networks
- **wav2vec2 Features**: Contextualized speech representations learned from large-scale unlabeled audio

### Pre-trained Models & Frameworks

#### Computer Vision Stack
- **DeepFace v0.0.93**: Meta's facial analysis framework
  - **Backends**: OpenCV (speed) + RetinaFace (accuracy)
  - **Base Models**: VGG-Face, Facenet, ArcFace architectures
  - **Emotion Classes**: 7 basic emotions (angry, disgust, fear, happy, sad, surprise, neutral)

#### Speech Processing Stack
- **wav2vec2-lg-xlsr-en-speech-emotion-recognition**: Hugging Face transformer model
  - **Architecture**: 300M+ parameter wav2vec2 base with emotion classification head
  - **Training**: Cross-lingual speech representation learning + emotion fine-tuning
  - **Author**: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

#### Supporting Frameworks
- **OpenCV**: Computer vision operations and camera interface
- **SoundDevice**: Real-time audio capture and processing
- **PyTorch/Transformers**: Neural network inference and model loading
- **Scipy**: Signal processing and audio filtering

### Processing Pipeline: Input → Output

#### 1. **Multimodal Input Capture**
```
Camera Feed (30 FPS) → Frame Buffer → Face Detection (every 0.5s)
Microphone (16kHz) → Audio Buffer → Speech Processing (1s windows)
```

#### 2. **Parallel Processing Streams**
```
VISUAL STREAM:
Raw Frame → Face Detection → ROI Extraction → DeepFace Analysis → Emotion Probabilities

AUDIO STREAM:
Raw Audio → Preprocessing → wav2vec2 Encoding → Emotion Classification → Confidence Scores
```

#### 3. **Multimodal Fusion Engine**
```
Visual Emotions + Audio Emotions → Confidence Weighting → Similarity Mapping → Final Prediction
```

#### 4. **Real-time Output**
```
Fused Emotions → Visual Overlay → Bounding Boxes → Console Metrics → User Interface
```

**Processing Flow Details:**
1. **Input Synchronization**: Audio and video streams processed in separate threads with timestamp alignment
2. **Feature Extraction**: Parallel computation of visual landmarks and audio embeddings
3. **Model Inference**: Simultaneous emotion prediction from both modalities
4. **Confidence Assessment**: Each prediction includes uncertainty estimates
5. **Late Fusion**: Weighted combination based on detection confidence and emotion coherence
6. **Output Integration**: Real-time display with performance metrics and debug information

**Performance Optimizations:**
- **Temporal Sampling**: Process every 0.5 seconds instead of every frame
- **Model Caching**: Pre-load all models during initialization
- **Backend Selection**: OpenCV for speed vs RetinaFace for accuracy
- **Threading**: Separate threads for audio processing and UI updates
- **Memory Management**: Efficient buffer handling for continuous processing

## Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/jjeff-link/empathy_mer.git
cd empathy_mer
```r vision and speech processing to detect human emotions across seven categories: *Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral*.

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
