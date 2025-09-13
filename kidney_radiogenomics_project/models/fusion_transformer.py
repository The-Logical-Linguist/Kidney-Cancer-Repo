import torch
import torch.nn as nn
from einops import rearrange

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        # q: (B, Tq, D), kv: (B, Tk, D)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = self.norm1(q + attn_out)
        ff = self.linear(x)
        x = self.norm2(x + ff)
        return x

class RadiogenomicFusion(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, clinical_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model, nhead) for _ in range(num_layers)])
        self.clin_proj = nn.Linear(clinical_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

    def forward(self, img_emb, gen_emb, clin_emb):
        # img_emb, gen_emb: (B, D); clin_emb: (B, C)
        B = img_emb.size(0)
        img_seq = img_emb.unsqueeze(1)
        gen_seq = gen_emb.unsqueeze(1)
        clin_seq = self.clin_proj(clin_emb).unsqueeze(1)
        seq = torch.cat([img_seq, gen_seq, clin_seq], dim=1)  # (B, 3, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, seq], dim=1)  # (B,4,D)

        for layer in self.layers:
            # let image attend to genomics+clinical and vice versa via shared sequence
            x = layer(x, x)

        return x[:,0,:]  # return CLS token
