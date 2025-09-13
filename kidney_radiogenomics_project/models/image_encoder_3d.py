import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm3d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)

class Encoder3D(nn.Module):
    """
    Lightweight 3D encoder for 96x96x96 patches.
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = BasicBlock3D(32, 64, stride=2)
        self.layer2 = BasicBlock3D(64, 128, stride=2)
        self.layer3 = BasicBlock3D(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x
