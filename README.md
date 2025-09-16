# Multimodal Emotion Recognition (MER) App Prototype

A high-performance **real-time multimodal emotion recognition system** that combines facial emotion detection and speech emotion recognition using state-of-the-art deep learning models.


## **System Architecture**

### **Processing Pipeline**
```
┌─────────────────┐    ┌─────────────────┐
│   Video Thread  │    │   Audio Thread  │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  OpenCV     │ │    │ │ SoundDevice │ │
│ │ Haar Cascade│ │    │ │  Capture    │ │
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



##  Acknowledgments

### **Models & Frameworks**
- **SuperB Consortium**: `wav2vec2-base-superb-er` emotion recognition model
- **DeepFace**: Meta's facial analysis framework  
- **Hugging Face**: Transformers library and model hosting
- **OpenCV**: Computer vision library

### **Architecture**
- **wav2vec2**: Facebook's self-supervised speech representation learning
- **PyTorch**: Deep learning framework
- **SoundDevice**: Real-time audio processing

