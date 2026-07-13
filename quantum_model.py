import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
import numpy as np

# ── Quantum circuit config ──────────────────────────────────────────────────
N_QUBITS = 4
N_LAYERS  = 2

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ── QNN wrapped via TorchLayer (handles batching + gradients natively) ──────
weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
QuantumLayer  = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


# ── Hybrid CNN-LSTM-QNN model ───────────────────────────────────────────────
class CNN_LSTM_QNN_Violence(nn.Module):
    """
    ResNet-18 (frozen early layers)
      → 2-layer LSTM (256 hidden)
      → dim_reduction Linear (256 → 4) + Tanh
      → QuantumLayer  (4 qubits, AngleEmbedding + BasicEntanglerLayers)
      → Linear (4 → 2)
    """
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, n_qubits=N_QUBITS):
        super().__init__()

        # ── CNN (unchanged) ──────────────────────────────────────────────
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for param in list(self.cnn.parameters())[:-10]:
            param.requires_grad = False

        # ── LSTM (unchanged) ─────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # ── Dimensionality reduction: 256 → n_qubits, scaled to [-π, π] ─
        self.dim_reduction = nn.Sequential(
            nn.Linear(hidden_size, n_qubits),
            nn.Tanh()
        )

        # ── Quantum layer (TorchLayer handles batching automatically) ─────
        self.quantum = QuantumLayer

        # ── Final classical output ────────────────────────────────────────
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # CNN: extract 512-dim features per frame
        cnn_out = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t]).view(batch_size, -1)   # (B, 512)
            cnn_out.append(feat)
        cnn_out = torch.stack(cnn_out, dim=1)               # (B, T, 512)

        # LSTM: temporal modeling
        lstm_out, _ = self.lstm(cnn_out)                    # (B, T, 256)
        last_hidden  = lstm_out[:, -1, :]                   # (B, 256)

        # Reduce to qubit-sized vector, scale to [-π, π]
        reduced = self.dim_reduction(last_hidden) * np.pi   # (B, 4)

        # Quantum processing (TorchLayer handles the batch loop internally)
        q_out = self.quantum(reduced)                       # (B, 4)

        return self.classifier(q_out)                       # (B, 2)
