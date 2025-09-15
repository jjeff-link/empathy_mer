# Multimodal Emotion Recognition (Prototype)

This is a simple **multimodal emotion recognition prototype** that combines **facial expressions** (from webcam) and **speech prosody** (from microphone) to predict a user’s emotional state in real time.


## ⚙️ Installation

### 1. Clone repository
```bash
git clone https://github.com/yourusername/empathy_mer.git
cd empathy_mer
```

### 2. Create Virtual Environment
python3 -m venv multimodal-env
source multimodal-env/bin/activate   # macOS/Linux
multimodal-env\Scripts\activate      # Windows (PowerShell)

### 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4. Run the prototype:
python mer.py