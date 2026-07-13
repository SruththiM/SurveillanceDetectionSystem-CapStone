# CNN + LSTM Violence Detection System

## Quick Start

### 1. Train the Model
```bash
python train_cnn_lstm_final.py
```
- Uses "Real Life Violence Dataset" folder
- Trains for 15 epochs
- Saves best model to `checkpoints/cnn_lstm_violence.pth`

### 2. Run Real-Time Detection
```bash
python detect_realtime.py
```
- Collects 16 consecutive frames
- Confidence threshold: 85%
- Consecutive threshold: 5 sequences
- Press 'q' to quit

## How It Works

**Training:**
- Extracts 16 frames per video
- ResNet18 extracts spatial features (512-dim)
- LSTM captures temporal patterns (256 hidden units)
- Output: Violence (1) or NonViolence (0)

**Detection:**
- Buffers 16 webcam frames
- Passes through CNN+LSTM
- Triggers alert only if:
  - Violence probability > 85%
  - Detected in 5 consecutive sequences

## Files
- `train_cnn_lstm_final.py` - Training script
- `detect_realtime.py` - Real-time webcam detection
- `checkpoints/cnn_lstm_violence.pth` - Trained model (created after training)

## Requirements
```bash
pip install torch torchvision opencv-python numpy
```
