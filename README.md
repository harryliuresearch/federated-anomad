
# federated-anomad

Federated Anomaly Detection for Privacy-Preserving Distributed Systems (DP + Secure Aggregation + FedAvg).

## ✨ Key Features
- **Federated** training with **FedAvg** and optional **FedProx**.
- **Privacy-preserving** options: **Differential Privacy (Gaussian/Clip)** and **(mock) Secure Aggregation** with additive masks.
- **Multi-modal data**: logs (text), metrics (time-series), traces (synthetic).
- **Anomaly models**: Autoencoder (MLP), LSTM-AE, and a lightweight Transformer encoder.
- **Kubernetes-friendly**: Dockerfiles for server/client, `.env` configs, and sample `docker-compose` orchestration.
- **Reproducible experiments**: configs under `configs/`, seed control, deterministic workers where possible.
- **Benchmarks**: synthetic + toy datasets; metrics: AUROC, F1, Precision@K, PR-AUC; confusion + calibration reports.
- **CI-ready**: GitHub Actions for lint + unit tests.
- **Docs**: architecture diagrams, design rationale, threat model, experiment logs.

> This repository is designed to be **drop-in**: clone, create a virtual env, install, and run with sample data.

## 📦 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
# Run a tiny end-to-end federated round on synthetic data
python scripts/run_federated_demo.py --config configs/demo.yaml
```

## 🧭 Repository Structure
```
federated-anomad/
├── configs/
├── data/
│   ├── sample/
│   └── synthetic/
├── docs/
├── experiments/
├── notebooks/
├── scripts/
├── src/fanomad/
│   ├── clients/
│   ├── data/
│   ├── models/
│   ├── privacy/
│   ├── server/
│   └── utils/
├── tests/
├── docker/
└── .github/workflows/
```

## 🔒 Privacy Notes
- Differential privacy uses per-sample clipping and Gaussian noise; privacy accounting uses moments accountant approximation.
- Secure aggregation is implemented as a **didactic additive-mask protocol** for single-round toy setups (not production-hard).

## 📑 License
Apache-2.0 (see `LICENSE`).

## 🙌 Contributing
See `CONTRIBUTING.md`.
