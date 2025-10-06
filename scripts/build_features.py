#!/usr/bin/env python
import argparse
from src.ml.feature_store import build_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="SGS series code (e.g., 1)")
    ap.add_argument("--freq", default="B", help="Pandas freq (default B = business day)")
    ap.add_argument("--lags", type=int, default=7)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    out = build_dataset(code=args.code, freq=args.freq, lags=args.lags, train_ratio=args.train_ratio)
    if not out.get("ok"):
        print(out)
        raise SystemExit(1)
    print("OK:", out["meta"])

if __name__ == "__main__":
    main()
