# Smart Surveillance Violence Detection System

A deep learning-based violence detection system using CNN + LSTM architecture for real-time surveillance applications.

## 🎯 Overview

This project implements an automated violence detection system that analyzes video sequences to identify violent activities in real-time. The system combines the spatial feature extraction capabilities of Convolutional Neural Networks (CNN) with the temporal modeling power of Long Short-Term Memory (LSTM) networks.

## 🏗️ Architecture

- **CNN Backbone**: ResNet-18 (pre-trained on ImageNet)
- **Temporal Model**: 2-layer LSTM with 256 hidden units
- **Classifier**: Fully connected layers with dropout
- **Input**: 8-frame sequences (224×224 RGB)
- **Output**: Binary classification (Violence/Non-Violence)

## ✨ Features

- Transfer learning with frozen CNN layers for efficient training
- Temporal sequence modeling for action recognition
- Real-time detection capability with sliding window approach
- Checkpoint saving for best model preservation
- GPU acceleration support

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy

## 📁 Project Structure

```
SmartSurveillanceDetection/
├── train_cnn_lstm_final.py    # Main training script
├── detect_realtime.py          # Real-time detection script
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

### Inference

```bash
python detect_realtime.py
```

## 🔧 Model Configuration

| Parameter | Value |
|-----------|-------|
| Sequence Length | 8 frames |
| Batch Size | 2 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 15 |
| Train/Val Split | 80/20 |

## 📊 How It Works

1. **Frame Extraction**: Uniformly samples 8 frames from each video
2. **Preprocessing**: Resizes to 224×224, normalizes with ImageNet statistics
3. **Feature Extraction**: ResNet-18 extracts 512-dim features per frame
4. **Temporal Modeling**: LSTM processes the sequence of features
5. **Classification**: FC layers output Violence/Non-Violence prediction

## 🎓 Technical Details

### Transfer Learning
- Uses ResNet-18 pre-trained on ImageNet
- Freezes early layers to retain general visual features
- Fine-tunes last layers for violence-specific patterns

### Temporal Modeling
- 8-frame sequences capture motion dynamics
- LSTM remembers context across frames
- Distinguishes violent actions from normal activities

### Real-time Detection
- Sliding window approach with frame buffering
- Processes new frames continuously
- Threshold-based alert triggering

## 📈 Performance

The model achieves competitive accuracy on the Real Life Violence Dataset with:
- Efficient training through transfer learning
- Robust temporal feature learning
- Real-time inference capability

## 🔮 Future Improvements

- [ ] Implement 3D CNNs for joint spatiotemporal learning
- [ ] Add attention mechanisms for important frame selection
- [ ] Multi-modal learning (audio + video)
- [ ] Pose estimation integration
- [ ] Model quantization for edge deployment
- [ ] Explainability with Grad-CAM visualization

## 📝 Dataset

This project uses the **Real Life Violence Dataset**. Due to size constraints, the dataset is not included in this repository.

Download from: [Dataset Source Link]

## ⚠️ Limitations

- Requires GPU for real-time performance
- May produce false positives on sports/action scenes
- Performance depends on lighting and camera quality
- Fixed sequence length may miss very quick/slow actions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

[Your Name]

## 🙏 Acknowledgments

- ResNet architecture from torchvision
- Real Life Violence Dataset creators
- PyTorch community

---

**Note**: This system is designed for research and educational purposes. For production deployment in surveillance systems, additional testing, validation, and ethical considerations are required.
