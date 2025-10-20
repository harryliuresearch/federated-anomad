
from fanomad.data.synthetic import SyntheticTSDataset

def test_synth_len():
    ds = SyntheticTSDataset(T=120, seq_len=16, feature_dim=4, anomaly_rate=0.1, seed=7)
    assert len(ds) == 120-16
