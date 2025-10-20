
import re, numpy as np, torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize_line(line: str):
    return TOKEN_RE.findall(line.lower())

class ToyLogDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 16, vocab=None):
        with open(path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]
        toks = [tokenize_line(l) for l in self.lines]
        if vocab is None:
            vocab = {"<pad>":0, "<unk>":1}
            for t in toks:
                for tok in t:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
        ids = [[vocab.get(tok, 1) for tok in t] for t in toks]
        padded = [seq[:seq_len] + [0]*max(0, seq_len-len(seq)) for seq in ids]
        self.X = torch.tensor(np.array(padded), dtype=torch.long)
        self.y = torch.zeros(len(self.X))  # no labels; use reconstruction or LM loss
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
