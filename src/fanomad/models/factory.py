
from .mlp_ae import MLPAE
from .lstm_ae import LSTMAE
from .tiny_transformer import TinyTransformer

def make_model(name: str, input_dim: int):
    name = name.lower()
    if name == "mlp_ae":
        return MLPAE(input_dim=input_dim, hidden_dim=64, bottleneck=16)
    if name == "lstm_ae":
        return LSTMAE(input_dim=input_dim, hidden_dim=64, bottleneck=32)
    if name == "tiny_transformer":
        return TinyTransformer(d_model=64, nhead=4, num_layers=2, input_dim=input_dim)
    raise ValueError(f"Unknown model: {name}")
