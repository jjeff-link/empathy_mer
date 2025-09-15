# Multimodal Emotion Recognition (Prototype)

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
