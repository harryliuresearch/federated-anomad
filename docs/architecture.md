
# Architecture

- `server/aggregator.py`: orchestrates rounds, handles secure aggregation masks, updates global model.
- `clients/client.py`: local training, clipping, DP noise; posts masked updates.
- `privacy/dp.py`: DP-SGD utilities + moments accountant approximation.
- `models/`: MLP AE, LSTM AE, Tiny Transformer for sequence/metrics and bag-of-logs.
- `data/`: synthetic generators for metrics time series and toy log tokenization.
