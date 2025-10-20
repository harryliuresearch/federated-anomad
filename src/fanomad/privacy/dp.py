
import math
import torch

def clip_by_l2_norm(grad, max_norm: float):
    norm = grad.norm(2)
    if norm > max_norm:
        grad = grad * (max_norm / (norm + 1e-12))
    return grad

def add_gaussian_noise(tensor, sigma: float):
    return tensor + sigma * torch.randn_like(tensor)

def dp_sgd_step(model, loss, optimizer, clip_norm: float, noise_multiplier: float):
    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: 
                continue
            p.grad = clip_by_l2_norm(p.grad, clip_norm)
            p.grad = add_gaussian_noise(p.grad, noise_multiplier*clip_norm)
    optimizer.step()

def moments_accountant(eps_target: float, delta: float, steps: int, noise_multiplier: float):
    # Very rough analytical placeholder for demo purposes.
    eps = math.sqrt(2*steps*math.log(1/delta)) / noise_multiplier + steps*(math.exp(1/(noise_multiplier**2))-1)
    return eps
