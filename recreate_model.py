import torch
import torch.nn as nn

class SpeechEmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super(SpeechEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.softmax(self.fc(x), dim=-1)

# Create model instance
model = SpeechEmotionClassifier()

# Save the model's state_dict (this is what we need)
torch.save(model.state_dict(), "models/speech_emotion_classifier.pth")

print("✅ Created a fresh, valid speech_emotion_classifier.pth file!")
print("📁 Saved at: models/speech_emotion_classifier.pth")