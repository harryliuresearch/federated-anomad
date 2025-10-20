
import argparse, yaml, torch, numpy as np
from fanomad.utils.common import set_seed
from fanomad.data.synthetic import SyntheticTSDataset
from fanomad.models.factory import make_model
from fanomad.clients.client import train_local, model_delta, masked_update
from fanomad.server.aggregator import Aggregator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg.get("seed", 42))

    if cfg["data"]["kind"] == "synthetic_ts":
        dataset = SyntheticTSDataset(
            T=240, seq_len=cfg["data"]["seq_len"],
            feature_dim=cfg["data"]["feature_dim"],
            anomaly_rate=cfg["data"]["anomaly_rate"],
            seed=cfg.get("seed",42)
        )
        input_dim = cfg["data"]["feature_dim"]
    else:
        raise ValueError("Only synthetic_ts supported in demo script.")

    # Split dataset per client
    num_clients = cfg["num_clients"]
    n = len(dataset)
    idxs = np.array_split(np.random.permutation(n), num_clients)

    # Global model
    model = make_model(cfg["model"], input_dim=input_dim)
    agg = Aggregator(model, method=cfg["aggregation"])

    for r in range(cfg["rounds"]):
        masked_updates, masks = [], []
        for c in range(num_clients):
            sub = torch.utils.data.Subset(dataset, idxs[c])
            local = make_model(cfg["model"], input_dim=input_dim)
            local.load_state_dict(agg.global_model.state_dict(), strict=True)
            loss = train_local(local, sub, epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
                               lr=cfg["learning_rate"], dp_cfg=cfg["privacy"], device=cfg["device"])
            delta = model_delta(local, agg.global_model.state_dict())
            m_upd, m_mask = masked_update(delta, modulus=cfg["secure_aggregation"]["mask_modulus"], seed=42+r*31+c)
            masked_updates.append(m_upd); masks.append(m_mask)
            print(f"[Round {r}] Client {c}: local_loss={loss:.4f}")
        new_state = agg.aggregate(masked_updates, masks)
        agg.global_model.load_state_dict(new_state, strict=True)
        print(f"[Round {r}] Aggregation complete.")

    print("Demo finished. You can now adapt for real data/logs.")

if __name__ == "__main__":
    main()
