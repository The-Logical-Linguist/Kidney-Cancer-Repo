import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from models.image_encoder_3d import Encoder3D

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.matmul(z, z.t()) / self.temperature
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        negatives = sim[~mask].view(2*N, -1)
        labels = torch.zeros(2*N, dtype=torch.long, device=z.device)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        return nn.functional.cross_entropy(logits, labels)

def augment_3d(x):
    # Simple intensity flip + noise (placeholder)
    noise = torch.randn_like(x) * 0.01
    return x + noise

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    device = torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg["ssl_pretrain"]["batch_size"]
    epochs = cfg["ssl_pretrain"]["epochs"]
    lr = cfg["ssl_pretrain"]["lr"]
    temperature = cfg["ssl_pretrain"]["temperature"]

    # Placeholder random tensors to demonstrate training loop
    # Replace with real KidneyImagingDataset and DataLoader
    N = 16
    X = torch.randn(N, 1, 96, 96, 96)
    loader = DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Encoder3D(emb_dim=256).to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x in loader:
            x = x.to(device)
            x1 = augment_3d(x)
            x2 = augment_3d(x)
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - SSL loss: {running/len(loader):.4f}")

    torch.save(model.state_dict(), "checkpoints/ssl_image_encoder.pt")

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    main()
