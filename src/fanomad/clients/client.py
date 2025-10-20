
from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from ..privacy.dp import dp_sgd_step
from ..privacy.secure_agg import gen_mask_like

def train_local(model: nn.Module, dataset, epochs: int, batch_size: int, lr: float, dp_cfg: dict, device="cpu"):
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = ((xb - recon)**2).mean()
            if dp_cfg.get("dp", False):
                dp_sgd_step(model, loss, opt, dp_cfg.get("clip_norm", 1.0), dp_cfg.get("noise_multiplier", 0.5))
            else:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
    return total_loss / max(1, len(loader)*epochs)

def model_delta(model: nn.Module, initial_state) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        delta = {}
        for (name, p) in model.named_parameters():
            delta[name] = (p - initial_state[name]).detach().clone()
    return delta

def masked_update(delta: Dict[str, torch.Tensor], modulus=1000000, seed=None):
    mask = {k: gen_mask_like(v, modulus=modulus, seed=seed).to(v.dtype) for k, v in delta.items()}
    masked = {k: v + mask[k] for k, v in delta.items()}
    return masked, mask
