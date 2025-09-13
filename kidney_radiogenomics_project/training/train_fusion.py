import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm

from models.image_encoder_3d import Encoder3D
from models.genomics_encoder import GenomicsAutoencoder, ClinicalEncoder
from models.fusion_transformer import RadiogenomicFusion
from models.heads import ClassificationHead, CoxHead
from training.utils_survival import cox_partial_likelihood_loss

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    device = torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    # ---- Placeholders for dataloaders (replace with real datasets) ----
    N = 32
    x_img = torch.randn(N, 1, 96, 96, 96)
    x_gen = torch.randn(N, 500)  # e.g., selected genes
    x_cli = torch.randn(N, 8)    # clinical features
    y_bin = torch.randint(0, 2, (N,))
    y_sub = torch.randint(0, 4, (N,))  # 4 classes example
    t = torch.rand(N, 1) * 1000.0
    e = torch.randint(0, 2, (N, 1)).float()

    ds = TensorDataset(x_img, x_gen, x_cli, y_bin, y_sub, t, e)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # ---- Models ----
    img_enc = Encoder3D(emb_dim=256).to(device)
    gen_enc = GenomicsAutoencoder(input_dim=x_gen.shape[1], latent_dim=256).to(device)
    clin_enc = ClinicalEncoder(input_dim=x_cli.shape[1], out_dim=64).to(device)
    fusion = RadiogenomicFusion(d_model=256, nhead=4, num_layers=2, clinical_dim=64).to(device)

    head_bin = ClassificationHead(in_dim=256, num_classes=2).to(device)
    head_sub = ClassificationHead(in_dim=256, num_classes=4).to(device)
    head_cox = CoxHead(in_dim=256).to(device)

    # Optionally load SSL pretrained image encoder
    try:
        img_enc.load_state_dict(torch.load("checkpoints/ssl_image_encoder.pt", map_location=device), strict=False)
        print("Loaded SSL pretrained weights for image encoder.")
    except Exception as ex:
        print(f"Could not load SSL weights: {ex}")

    params = list(img_enc.parameters()) + list(gen_enc.parameters()) + list(clin_enc.parameters()) +              list(fusion.parameters()) + list(head_bin.parameters()) + list(head_sub.parameters()) + list(head_cox.parameters())
    opt = optim.AdamW(params, lr=cfg["fusion_train"]["lr"], weight_decay=cfg["fusion_train"]["weight_decay"])

    ce = nn.CrossEntropyLoss()
    for epoch in range(cfg["fusion_train"]["epochs"]):
        img_enc.train(); gen_enc.train(); clin_enc.train(); fusion.train()
        head_bin.train(); head_sub.train(); head_cox.train()
        running = {"bin":0.0, "sub":0.0, "cox":0.0}

        for xb, xg, xc, yb, ys, time, event in loader:
            xb, xg, xc = xb.to(device), xg.to(device), xc.to(device)
            yb, ys = yb.to(device), ys.to(device)
            time, event = time.to(device), event.to(device)

            # forward
            z_img = img_enc(xb)
            x_hat, z_gen = gen_enc(xg)
            z_cli = clin_enc(xc)
            z = fusion(z_img, z_gen, z_cli)

            logit_bin = head_bin(z)
            logit_sub = head_sub(z)
            logit_cox = head_cox(z)

            loss_bin = ce(logit_bin, yb)
            loss_sub = ce(logit_sub, ys)
            loss_cox = cox_partial_likelihood_loss(logit_cox, time, event)

            loss = loss_bin + loss_sub + cfg["fusion_train"]["cox_lambda"] * loss_cox

            opt.zero_grad()
            loss.backward()
            opt.step()

            running["bin"] += loss_bin.item()
            running["sub"] += loss_sub.item()
            running["cox"] += loss_cox.item()

        n = len(loader)
        print(f"Epoch {epoch+1} | Bin: {running['bin']/n:.3f} | Sub: {running['sub']/n:.3f} | Cox: {running['cox']/n:.3f}")

    torch.save({
        "img_enc": img_enc.state_dict(),
        "gen_enc": gen_enc.state_dict(),
        "clin_enc": clin_enc.state_dict(),
        "fusion": fusion.state_dict(),
        "head_bin": head_bin.state_dict(),
        "head_sub": head_sub.state_dict(),
        "head_cox": head_cox.state_dict(),
    }, "checkpoints/fusion_model.pt")

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    main()
