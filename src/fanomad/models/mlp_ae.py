
import torch, torch.nn as nn

class MLPAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bottleneck=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out, z

    def anomaly_score(self, x, recon):
        return ((x - recon)**2).mean(dim=-1)
