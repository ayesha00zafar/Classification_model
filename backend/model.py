import torch
import torch.nn as nn

INPUT_DIM = 32

class _Head(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class OccluzNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.head_right       = _Head(32, 5)
        self.head_left        = _Head(32, 5)
        self.head_anterior    = _Head(32, 4)
        self.head_transversal = _Head(32, 3)
        self.head_median      = _Head(32, 2)

    def forward(self, x):
        s = self.backbone(x)
        return {
            "right":       self.head_right(s),
            "left":        self.head_left(s),
            "anterior":    self.head_anterior(s),
            "transversal": self.head_transversal(s),
            "median":      self.head_median(s),
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return {k: torch.argmax(v, dim=1) for k, v in logits.items()}
