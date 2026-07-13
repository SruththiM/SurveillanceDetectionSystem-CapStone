import torch
import torch.nn as nn
from torchvision import models
import cv2
from collections import deque
import winsound
import threading
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage


class CNN_LSTM_Violence(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(CNN_LSTM_Violence, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
       
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


SEQUENCE_LENGTH = 8


VIOLENCE_THRESHOLD = 0.85  
WARNING_THRESHOLD = 0.75   

device = torch.device("cpu")

print("Loading model...")
model = CNN_LSTM_Violence(num_classes=2, hidden_size=256, num_layers=2).to(device)
model.load_state_dict(torch.load('checkpoints/cnn_lstm_violence.pth', map_location=device))
model.eval()
print("Model loaded!\n")


frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
alert_playing = False


location = "School Main Gate"
LATITUDE  = 13.0827
LONGITUDE = 80.2707
os.makedirs("alerts", exist_ok=True)
alerts_dir = os.path.abspath("alerts")
last_saved_time = 0


SENDER_EMAIL   = "sruththimurugavel@gmail.com"
APP_PASSWORD   = "dgkwlzgmgjpesgiw"
RECEIVER_EMAIL = "sruththim@gmail.com"
last_email_time = 0

def send_email_alert(image_path, timestamp, location):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Violence Detected Alert'
        msg['From']    = SENDER_EMAIL
        msg['To']      = RECEIVER_EMAIL
        msg.set_content(
            f"Violence Detected!\n"
            f"Location  : {location}\n"
            f"Time      : {timestamp}\n"
            f"Latitude  : {LATITUDE}\n"
            f"Longitude : {LONGITUDE}\n"
            f"Maps Link : https://maps.google.com/?q={LATITUDE},{LONGITUDE}\n\n"
            f"Please take immediate action."
        )
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype='image',
                                   subtype='jpeg',
                                   filename=os.path.basename(image_path))
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"Email error: {e}")

def play_alert():
    global alert_playing
    alert_playing = True
    winsound.Beep(1000, 1000)
    alert_playing = False


mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

print("Starting real-time violence detection...")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Violence threshold: {VIOLENCE_THRESHOLD*100}%")
print(f"Warning threshold: {WARNING_THRESHOLD*100}%")
print("Press 'q' to quit\n")

cap = None
for index in [0, 1, 2]:
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Camera opened: index={index}, backend={backend}")
            break
    if cap.isOpened():
        break

if not cap or not cap.isOpened():
    print("ERROR: No camera found! Check if webcam is connected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.FloatTensor(frame_resized).permute(2, 0, 1) / 255.0
    frame_tensor = (frame_tensor - mean) / std
    
   
    frame_buffer.append(frame_tensor)
    
   
    violence_prob = 0.0
    status = "Waiting..."
    
    if len(frame_buffer) == SEQUENCE_LENGTH:
       
        video_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            violence_prob = probabilities[0][1].item()  
    
   
    if violence_prob >= VIOLENCE_THRESHOLD:
        label = "VIOLENCE DETECTED!"
        status = "DANGER"
        color = (0, 0, 255)  
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 15)
        if not alert_playing:
            threading.Thread(target=play_alert, daemon=True).start()
        current_time = datetime.now()
        if (current_time.timestamp() - last_saved_time) >= 30:
            timestamp_display  = current_time.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_filename = current_time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(alerts_dir, f"{location.replace(' ','_')}_{timestamp_filename}.jpg")
            saved = cv2.imwrite(filename, frame)
            if saved:
                print(f"Alert saved: {filename}")
                threading.Thread(
                    target=send_email_alert,
                    args=(filename, timestamp_display, location),
                    daemon=True
                ).start()
            last_saved_time = current_time.timestamp()
    elif violence_prob >= WARNING_THRESHOLD:
        label = "Suspicious Activity"
        status = "WARNING"
        color = (0, 165, 255)  
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
    else:
        label = "All Clear [SAFE]"
        status = "SAFE"
        color = (0, 255, 0)  
    
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Location: {location}",          (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255,   0), 2)
    cv2.putText(frame, f"Time    : {now_str}",            (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"GPS     : {LATITUDE},{LONGITUDE}",(20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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


