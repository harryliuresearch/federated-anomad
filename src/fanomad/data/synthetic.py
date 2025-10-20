
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticTSDataset(Dataset):
    def __init__(self, T=240, seq_len=48, feature_dim=8, anomaly_rate=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.X = []
        self.y = []
        t = np.linspace(0, 20, T)
        base = np.sin(t)[:, None] + 0.1*rng.normal(size=(T,1))
        data = base + 0.05*rng.normal(size=(T, feature_dim))
        labels = np.zeros(T, dtype=int)
        anom_idx = rng.choice(T, size=max(1,int(T*anomaly_rate)), replace=False)
        labels[anom_idx] = 1
        for i in range(T - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(labels[i+seq_len-1])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
