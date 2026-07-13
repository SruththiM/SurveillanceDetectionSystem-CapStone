import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import cv2
import os
import numpy as np
from quantum_model import CNN_LSTM_QNN_Violence


class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8):   
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.video_paths = []
        self.labels = []
        
        for label, folder in enumerate(['NonViolence', 'Violence']):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found!")
                continue
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):
                    self.video_paths.append(os.path.join(folder_path, video_file))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.video_paths)} videos "
              f"({self.labels.count(0)} NonViolence, "
              f"{self.labels.count(1)} Violence)")
    
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
        
        indices = np.linspace(0, max(0, total_frames - 1),
                              self.sequence_length, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
                
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                frame = (frame - mean) / std
                
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.sequence_length])



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    dataset = ViolenceVideoDataset("Real Life Violence Dataset", sequence_length=8)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"Train: {train_size} | Val: {val_size}\n")
    
    model = CNN_LSTM_QNN_Violence().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_acc = 0
    EPOCHS = 15
    
    print("Training started...\n")
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        
       
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
        
        print(f"\nEpoch {epoch+1} Complete | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%\n")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/cnn_lstm_qnn_violence.pth')
            print(f"*** Model Saved! Best Val Acc: {best_acc:.2f}% ***\n")
    
    print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")