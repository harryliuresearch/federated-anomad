
import torch, random

def gen_mask_like(tensor, modulus=1000000, seed=None):
    if seed is not None:
        random.seed(seed)
    mask = torch.randint(low=-(modulus//2), high=(modulus//2), size=tensor.shape, dtype=tensor.dtype)
    return mask

def apply_mask(update, mask):
    return update + mask

def remove_mask(sum_updates, masks):
    # sum updates include masks; aggregator removes masks (assuming pairwise cancelation)
    return sum_updates - sum(masks)
