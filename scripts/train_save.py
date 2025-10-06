#!/usr/bin/env python
import argparse
from src.ml.train import train_and_backtest
from src.ml.registry import make_run_dir, save_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--freq", default="B")
    ap.add_argument("--order", nargs=3, type=int, default=[1,1,1])
    ap.add_argument("--seasonal", nargs=4, type=int, default=[0,0,0,0])
    ap.add_argument("--backtest_h", type=int, default=5)
    args = ap.parse_args()

    out = train_and_backtest(
        code=args.code,
        order=tuple(args.order),
        seasonal_order=tuple(args.seasonal),
        backtest_h=args.backtest_h,
    )
    if not out.get("ok"):
        print(out); raise SystemExit(1)

    run_dir = make_run_dir(args.code, args.freq)
    paths = save_artifacts(
        out_dir=run_dir,
        model_obj=out["model"],
        metrics=out["metrics"],
        meta={
            "code": args.code,
            "freq": args.freq,
            "order": args.order,
            "seasonal_order": args.seasonal,
            "n_obs": out["n_obs"],
        }
    )
    print("Saved:", paths, "in", run_dir)

if __name__ == "__main__":
    main()
