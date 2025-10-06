import os
import json
import time
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from src.ml.etl import load_series, make_supervised

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def build_dataset(
    code: str,
    freq: str = "B",
    lags: int = 7,
    train_ratio: float = 0.8,
    base_dir: str = "data/feature_store"
) -> Dict[str, Any]:
    """
    Lê a série do Postgres, cria features (lags) e salva em Parquet + meta.json
    Retorna dict com caminhos e estatísticas.
    """
    df = load_series(code)
    if df.empty:
        return {"ok": False, "error": f"sem dados para série {code}"}

    # supervisionado com lags
    sup = make_supervised(df, freq=freq, lags=lags)
    if sup.empty:
        return {"ok": False, "error": "dataset supervisionado vazio após resample/ffill"}

    n = len(sup)
    cut = int(n * train_ratio)
    sup["split"] = "train"
    sup.loc[sup.index >= cut, "split"] = "test"

    # versionamento por run timestamp
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, "source=SGS", f"series_code={code}", f"freq={freq}", f"run={run_id}")
    _ensure_dir(out_dir)

    ds_path = os.path.join(out_dir, "dataset.parquet")
    meta_path = os.path.join(out_dir, "meta.json")

    # Salva parquet
    sup.to_parquet(ds_path, index=False)

    # Meta com parâmetros e stats
    meta: Dict[str, Any] = {
        "code": code,
        "freq": freq,
        "lags": lags,
        "train_ratio": train_ratio,
        "n_rows": int(n),
        "n_train": int((sup["split"] == "train").sum()),
        "n_test": int((sup["split"] == "test").sum()),
        "columns": sup.columns.tolist(),
        "dataset_path": ds_path,
        "run_id": run_id,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"ok": True, "meta": meta, "paths": {"dataset": ds_path, "meta": meta_path}}

def latest_run_dir(code: str, freq: str = "B", base_dir: str = "data/feature_store") -> Optional[str]:
    root = Path(base_dir) / "source=SGS" / f"series_code={code}" / f"freq={freq}"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.name.startswith("run=")], key=lambda p: p.name)
    return str(runs[-1]) if runs else None
