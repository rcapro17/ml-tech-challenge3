#!/usr/bin/env python
import argparse, json
from src.ml.tuning import tune_sarimax_for_code
from src.ml.registry import make_run_dir, save_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--freq", default="B")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--init_ratio", type=float, default=0.7)
    args = ap.parse_args()

    out = tune_sarimax_for_code(
        code=args.code,
        freq=args.freq,
        horizon=args.horizon,
        initial_train_ratio=args.init_ratio
    )
    if not out.get("ok"):
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    best = out["best"]
    run_dir = make_run_dir(args.code, args.freq)
    paths = save_artifacts(
        out_dir=run_dir,
        model_obj=best["model"],
        metrics=best["metrics"],
        meta={
            "code": args.code,
            "freq": args.freq,
            "order": list(best["order"]),
            "seasonal_order": list(best["seasonal_order"]),
            "y_len": out["y_len"],
            "horizon": args.horizon,
            "initial_train_ratio": args.init_ratio,
            "ranked_results": out["results"],  # guarda ranking dos candidatos
        }
    )
    print("Melhor config:", best["order"], best["seasonal_order"], "RMSE:", best["metrics"]["RMSE"])
    print("Artefatos salvos em:", run_dir)
    print("Paths:", paths)

if __name__ == "__main__":
    main()
