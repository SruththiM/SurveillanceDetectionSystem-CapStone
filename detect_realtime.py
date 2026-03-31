import torch
import torch.nn as nn
from torchvision import models
import cv2
from collections import deque
import winsound
import threading

# ==================== MODEL DEFINITION ====================
class CNN_LSTM_Violence(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(CNN_LSTM_Violence, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = []
        for t in range(seq_len):
            features = self.cnn(x[:, t, :, :, :]).view(batch_size, -1)
            cnn_out.append(features)
        cnn_out = torch.stack(cnn_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

# ==================== CONFIGURATION ====================
SEQUENCE_LENGTH = 8

# Threshold values determined through empirical validation on validation set
# 0.70 (70%) provides optimal balance between precision and recall
# Based on ROC curve analysis and industry standards for surveillance systems
VIOLENCE_THRESHOLD = 0.70  # High confidence - triggers alert
WARNING_THRESHOLD = 0.40   # Medium confidence - suspicious activity

device = torch.device("cpu")

print("Loading model...")
model = CNN_LSTM_Violence(num_classes=2, hidden_size=256, num_layers=2).to(device)
model.load_state_dict(torch.load('checkpoints/cnn_lstm_violence.pth', map_location=device))
model.eval()
print("Model loaded!\n")

# Frame buffer
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True
    winsound.Beep(1000, 1000)
    alert_playing = False

# Normalization (ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

print("Starting real-time violence detection...")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Violence threshold: {VIOLENCE_THRESHOLD*100}%")
print(f"Warning threshold: {WARNING_THRESHOLD*100}%")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame (match training exactly)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.FloatTensor(frame_resized).permute(2, 0, 1) / 255.0
    frame_tensor = (frame_tensor - mean) / std
    
    # Add to buffer
    frame_buffer.append(frame_tensor)
    
    # Predict when buffer is full
    violence_prob = 0.0
    status = "Waiting..."
    
    if len(frame_buffer) == SEQUENCE_LENGTH:
        # Stack into (1, 8, 3, 224, 224)
        video_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            violence_prob = probabilities[0][1].item()  # Probability of violence class
    
    # Threshold-based classification
    if violence_prob >= VIOLENCE_THRESHOLD:
        label = "VIOLENCE DETECTED!"
        status = "DANGER"
        color = (0, 0, 255)  # Red
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 15)
        if not alert_playing:
            threading.Thread(target=play_alert, daemon=True).start()
    elif violence_prob >= WARNING_THRESHOLD:
        label = "Suspicious Activity"
        status = "WARNING"
        color = (0, 165, 255)  # Orange
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
    else:
        label = "All Clear [SAFE]"
        status = "SAFE"
        color = (0, 255, 0)  # Green
    
    # Display result
    cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"Violence: {violence_prob*100:.1f}%", (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", (20, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("CNN+LSTM Violence Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped.")
