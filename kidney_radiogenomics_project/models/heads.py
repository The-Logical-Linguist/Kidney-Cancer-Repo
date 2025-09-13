import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CoxHead(nn.Module):
    """
    Cox proportional hazards: outputs a single log-risk score.
    Loss should be computed externally using partial likelihood.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(-1)
