
import torch, torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bottleneck=32):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_bottleneck = nn.Linear(hidden_dim, bottleneck)
        self.dec = nn.LSTM(bottleneck, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, F)
        h,_ = self.enc(x)
        h_last = h[:, -1, :]
        z = self.to_bottleneck(h_last).unsqueeze(1).expand(-1, x.size(1), -1)
        y,_ = self.dec(z)
        recon = self.out(y)
        return recon, z[:,0,:]

    def anomaly_score(self, x, recon):
        return ((x - recon)**2).mean(dim=(1,2))
