import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import cv2
import os
import numpy as np
import time


class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.video_paths = []
        self.labels = []
        
        for label, folder in enumerate(['NonViolence', 'Violence']):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):
                    self.video_paths.append(os.path.join(folder_path, video_file))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.video_paths)} videos ({self.labels.count(0)} NonViolence, {self.labels.count(1)} Violence)")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.extract_frames(video_path)
        return frames, label
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return torch.zeros(self.sequence_length, 3, 224, 224)
        
        indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                frame = (frame - mean) / std
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.sequence_length])


class CNN_LSTM_Violence(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(CNN_LSTM_Violence, self).__init__()
        
        # CNN: ResNet18 pretrained - FROZEN
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ALL CNN parameters
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # LSTM - TRAINABLE
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # Classifier - TRAINABLE
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract CNN features (no gradients)
        cnn_out = []
        with torch.no_grad():
            for t in range(seq_len):
                features = self.cnn(x[:, t, :, :, :]).view(batch_size, -1)
                cnn_out.append(features)
        
        cnn_out = torch.stack(cnn_out, dim=1)
        
        # LSTM (with gradients)
        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        
        return out

if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Device: {device}")
    print("CNN: FROZEN (feature extractor only)")
    print("LSTM + FC: TRAINABLE\n")
    
    # Load dataset
    dataset = ViolenceVideoDataset("Real Life Violence Dataset", sequence_length=8)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Train: {train_size} | Val: {val_size}\n")
    
    # Model
    model = CNN_LSTM_Violence(num_classes=2, hidden_size=256, num_layers=2).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Training loop
    best_acc = 0
    EPOCHS = 15
    
    print("="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*train_correct/train_total:.1f}%")
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS} COMPLETE")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/cnn_lstm_violence.pth')
            print(f"*** MODEL SAVED! Best Val Acc: {best_acc:.2f}% ***\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("="*60)
