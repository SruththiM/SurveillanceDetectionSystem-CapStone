import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==================== VIDEO DATASET ====================
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
        
        print(f"Loaded {len(self.video_paths)} videos")
    
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
                
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                frame = (frame - mean) / std
                
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames[:self.sequence_length])

# ==================== MODEL DEFINITION ====================
class CNN_LSTM_Violence(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(CNN_LSTM_Violence, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in list(self.cnn.parameters())[:-10]:
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

# ==================== EVALUATION ====================
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            
            # Store probabilities and labels
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Apply threshold
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ViolenceVideoDataset("Real Life Violence Dataset", sequence_length=8)
    
    # Split dataset (same as training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"Validation set: {val_size} videos\n")
    
    # Load model
    print("Loading model...")
    model = CNN_LSTM_Violence().to(device)
    model.load_state_dict(torch.load('checkpoints/cnn_lstm_violence.pth', map_location=device))
    print("Model loaded!\n")
    
    # Evaluate with different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for threshold in thresholds:
        print(f"\n{'='*70}")
        print(f"THRESHOLD: {threshold*100:.0f}%")
        print(f"{'='*70}")
        
        preds, labels, probs = evaluate_model(model, val_loader, device, threshold)
        
        # Calculate metrics
        accuracy = (preds == labels).mean() * 100
        precision = precision_score(labels, preds, zero_division=0) * 100
        recall = recall_score(labels, preds, zero_division=0) * 100
        f1 = f1_score(labels, preds, zero_division=0) * 100
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nAccuracy:  {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1-Score:  {f1:.2f}%")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        print(f"\nInterpretation:")
        print(f"  - Correctly identified {tp} violent videos")
        print(f"  - Correctly identified {tn} non-violent videos")
        print(f"  - Missed {fn} violent videos (False Negatives)")
        print(f"  - Wrongly flagged {fp} non-violent videos (False Positives)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print("\nBased on the results above:")
    print("- Threshold 0.5: Balanced, good for general use")
    print("- Threshold 0.7: Reduces false positives, recommended for surveillance")
    print("- Threshold 0.9: Very few false alarms, but may miss some violence")
    print("\nChoose threshold based on your priority:")
    print("  High Recall (catch all violence) → Lower threshold (0.5-0.6)")
    print("  High Precision (few false alarms) → Higher threshold (0.7-0.8)")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
