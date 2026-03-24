# Smart Surveillance Violence Detection System

A deep learning-based violence detection system using CNN + LSTM architecture for real-time surveillance applications.

## 🎯 Overview

This project implements an automated violence detection system that analyzes video sequences to identify violent activities in real-time. The system combines the spatial feature extraction capabilities of ResNet-18 CNN with the temporal modeling power of Long Short-Term Memory (LSTM) networks.

## 🏗️ Architecture

- **CNN Backbone**: ResNet-18 (pre-trained on ImageNet - IMAGENET1K_V1)
- **Temporal Model**: 2-layer LSTM with 256 hidden units, dropout 0.3
- **Classifier**: Fully connected layers (256 → 128 → 2) with ReLU and Dropout 0.5
- **Input**: 8-frame sequences (224×224 RGB)
- **Output**: Binary classification (Violence / Non-Violence)

## ✨ Features

- Transfer learning with ResNet-18 pre-trained on ImageNet
- Frozen early CNN layers for efficient training
- 2-layer LSTM for temporal sequence modeling
- Three-level real-time alert system (Safe / Warning / Danger)
- Threshold-based detection (70% confidence for violence alert)
- Sliding window frame buffering for continuous detection
- Checkpoint saving for best model preservation
- GPU acceleration support (CPU fallback available)
- Model evaluation with Precision, Recall, F1-Score metrics

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- TorchVision 0.15+
- OpenCV 4.8+
- NumPy 1.24+
- Scikit-learn (for evaluation metrics)

## 📁 Project Structure

```
SmartSurveillanceDetection/
├── train_cnn_lstm_final.py    # Main training script
├── detect_realtime.py          # Real-time detection with threshold alerts
├── evaluate_model.py           # Model evaluation (Precision, Recall, F1)
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── checkpoints/                # Saved model weights (not in repo)
└── Real Life Violence Dataset/ # Dataset folder (not in repo)
    ├── Violence/
    └── NonViolence/
```

## 🚀 Usage

### Training

1. **Prepare Dataset**: Organize videos in the following structure:
```
Real Life Violence Dataset/
├── Violence/
│   ├── V_1.mp4
│   ├── V_2.mp4
│   └── ...
└── NonViolence/
    ├── NV_1.mp4
    ├── NV_2.mp4
    └── ...
```

2. **Run Training**:
```bash
python train_cnn_lstm_final.py
```

### Real-time Detection

```bash
python detect_realtime.py
```

### Evaluate Model

```bash
python evaluate_model.py
```

## 🔧 Model Configuration

| Parameter | Value |
|-----------|-------|
| CNN Backbone | ResNet-18 (ImageNet pretrained) |
| LSTM Layers | 2 |
| LSTM Hidden Units | 256 |
| LSTM Dropout | 0.3 |
| FC Layers | 256 → 128 → 2 |
| FC Dropout | 0.5 |
| Sequence Length | 8 frames |
| Batch Size | 2 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 15 |
| Train/Val Split | 80/20 |
| Violence Threshold | 70% |
| Warning Threshold | 40% |

## 🚨 Alert System

| Level | Threshold | Color | Action |
|-------|-----------|-------|--------|
| 🟢 SAFE | < 40% | Green | Normal activity |
| 🟡 WARNING | 40% - 70% | Orange | Suspicious activity |
| 🔴 DANGER | > 70% | Red | Violence detected - Alert! |

## 📊 How It Works

1. **Frame Extraction**: Uniformly samples 8 frames from each video
2. **Preprocessing**: Resizes to 224×224, normalizes with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Feature Extraction**: ResNet-18 extracts 512-dim feature vector per frame
4. **Temporal Modeling**: 2-layer LSTM processes the sequence of 8 feature vectors
5. **Classification**: FC layers output Violence/Non-Violence probability
6. **Threshold Decision**: 70% threshold triggers violence alert

## 🎓 Technical Details

### CNN - ResNet-18
- Pre-trained on ImageNet (IMAGENET1K_V1 weights)
- Removes final classification layer, keeps feature extractor
- Outputs 512-dimensional feature vector per frame
- Early layers frozen, last 10 parameters fine-tuned

### LSTM - Temporal Modeling
- Input size: 512 (from ResNet-18 features)
- Hidden size: 256 units
- Number of layers: 2
- Dropout: 0.3 between layers
- Processes sequence of 8 frames
- Final hidden state used for classification

### Fully Connected Classifier
- Layer 1: 256 → 128 (ReLU + Dropout 0.5)
- Layer 2: 128 → 2 (Violence / Non-Violence)
- Softmax for probability output

### Real-time Detection
- Sliding window approach with 8-frame buffer
- Processes new frames continuously from webcam
- Threshold-based three-level alert system

## 📈 Performance Metrics

Model evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: Of all violence alerts, how many were correct
- **Recall**: Of all actual violence, how many were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN analysis

## 🔮 Future Improvements

- [ ] Implement 3D CNNs for joint spatiotemporal learning
- [ ] Add attention mechanisms for important frame selection
- [ ] Multi-modal learning (audio + video)
- [ ] Pose estimation integration
- [ ] Model quantization for edge deployment
- [ ] Explainability with Grad-CAM visualization
- [ ] Multi-camera support

## 📝 Dataset

This project uses the **Real Life Violence Dataset** containing 2000 videos (1000 Violence + 1000 Non-Violence). Due to size constraints, the dataset is not included in this repository.

## ⚠️ Limitations

- Requires GPU for optimal real-time performance
- May produce false positives on sports/action scenes
- Performance depends on lighting and camera quality
- Fixed sequence length may miss very quick/slow actions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Sruththi

## 🙏 Acknowledgments

- ResNet-18 architecture from torchvision (IMAGENET1K_V1 weights)
- Real Life Violence Dataset creators
- PyTorch and OpenCV communities

---

**Note**: This system is designed for research and educational purposes. For production deployment in surveillance systems, additional testing, validation, and ethical considerations are required.
