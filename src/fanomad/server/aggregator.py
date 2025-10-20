
from __future__ import annotations
import copy, torch
from typing import List, Dict
from ..privacy.secure_agg import remove_mask

def fedavg(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    agg = {}
    for u in updates:
        for k, v in u.items():
            agg.setdefault(k, torch.zeros_like(v))
            agg[k] += v
    for k in agg:
        agg[k] /= max(1, len(updates))
    return agg

def fedprox(updates: List[Dict[str, torch.Tensor]], mu=0.001) -> Dict[str, torch.Tensor]:
    # Placeholder identical to FedAvg; Prox used client-side. Keep for extensibility.
    return fedavg(updates)

class Aggregator:
    def __init__(self, model, method="fedavg"):
        self.global_model = model
        self.method = method

    def aggregate(self, masked_updates: List[Dict[str, torch.Tensor]], masks: List[Dict[str, torch.Tensor]]):
        # Remove masks
        unmasked = []
        for u, m in zip(masked_updates, masks):
            clean = {k: u[k] - m.get(k, torch.zeros_like(u[k])) for k in u}
            unmasked.append(clean)
        if self.method == "fedavg":
            delta = fedavg(unmasked)
        else:
            delta = fedavg(unmasked)
        with torch.no_grad():
            for name, p in self.global_model.named_parameters():
                p += delta[name]
        return copy.deepcopy(self.global_model.state_dict())
