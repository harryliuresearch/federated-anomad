
import torch, torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, input_dim=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.proj(x)
        h = self.encoder(h)
        recon = self.head(h)
        z = h.mean(dim=1)
        return recon, z

    def anomaly_score(self, x, recon):
        return ((x - recon)**2).mean(dim=(1,2))
