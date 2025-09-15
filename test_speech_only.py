import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

# Load processor and pretrained Wav2Vec 2.0 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()  # Set to inference mode

# Freeze Wav2Vec parameters (optional, for efficiency)
for param in wav2vec_model.parameters():
    param.requires_grad = False

# Define your speech classifier class (must match saved model!)
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

# Load trained weights
classifier = SpeechEmotionClassifier()
try:
    state_dict = torch.load("models/speech_emotion_classifier.pth", map_location='cpu', weights_only=False)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    print("✅ Loaded trained speech emotion classifier.")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")
    exit(1)

# Define emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Generate test audio signals (2 seconds at 16kHz)
sample_rate = 16000
duration = 2.0

# Silent audio
silence = np.zeros(int(sample_rate * duration), dtype=np.float32)

# Happy tone: higher pitch, energetic
happy_tone = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))) * 0.3

# Sad tone: lower pitch, slow
sad_tone = np.sin(2 * np.pi * 110 * np.linspace(0, duration, int(sample_rate * duration))) * 0.3

# Angry tone: sharp, noisy
t = np.linspace(0, duration, int(sample_rate * duration))
angry_tone = (
    np.sin(2 * np.pi * 300 * t) * 0.2 +
    np.random.normal(0, 0.1, len(t)) * 0.2
)

# Test each
audio_tests = [
    ("Silence", silence),
    ("Happy Tone", happy_tone),
    ("Sad Tone", sad_tone),
    ("Angry Tone", angry_tone),
]

print("\n🧪 Testing Trained Speech Emotion Classifier\n" + "="*50)

for label, audio in audio_tests:
    # Step 1: Preprocess audio with Wav2Vec2 Processor
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # Step 2: Pass through Wav2Vec2 model to get last_hidden_state
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)  # <-- THIS IS CRITICAL!
        hidden_states = outputs.last_hidden_state  # Shape: [1, seq_len, 768]

    # Step 3: Mean pooling over time dimension → [1, 768]
    speech_embedding = hidden_states.mean(dim=1)

    # Step 4: Add dummy sequence dim → [1, 768, 1] for Conv1d
    speech_embedding = speech_embedding.unsqueeze(-1)

    # Step 5: Classify using your trained CNN
    with torch.no_grad():
        logits = classifier(speech_embedding)
        probs = logits[0].numpy()
        pred_idx = np.argmax(probs)
        pred_label = EMOTION_LABELS[pred_idx]

    print(f"\n🔊 {label}")
    print(f"   Probabilities: {probs.round(3)}")
    print(f"   Predicted: {pred_label} (Conf: {probs[pred_idx]:.3f})")