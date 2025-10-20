
import argparse, json, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--out", type=str, default="experiments/splits.json")
    args = ap.parse_args()
    idxs = np.random.permutation(args.n)
    parts = np.array_split(idxs, args.clients)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump([p.tolist() for p in parts], open(args.out, "w"))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
