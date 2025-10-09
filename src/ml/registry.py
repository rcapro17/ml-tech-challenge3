"""
Model registry (local folder or S3) using DATA_URI as base.
Layout:
  <DATA_URI>/models/source=SGS/series_code={code}/freq={freq}/run={run_id}/{model.pkl,metrics.json,meta.json}
"""
from __future__ import annotations

import os
import json
import time
import pickle
from typing import Optional, Dict, Any
from pathlib import Path

DATA_URI = os.environ.get("DATA_URI", "data")
IS_S3 = DATA_URI.startswith("s3://")
BASE = f"{DATA_URI.rstrip('/')}/models" if IS_S3 else str(Path(DATA_URI) / "models")


def _uri(*parts: str) -> str:
    if IS_S3:
        base = BASE.rstrip("/")
        suffix = "/".join(p.strip("/\\") for p in parts if p)
        return f"{base}/{suffix}"
    else:
        base = Path(BASE)
        return str(base.joinpath(*parts))


def _ensure_local_dir(path_str: str) -> None:
    if IS_S3:
        return
    Path(path_str).mkdir(parents=True, exist_ok=True)


def make_run_dir(code: str, freq: str = "B") -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _uri("source=SGS", f"series_code={code}", f"freq={freq}", f"run={run_id}")
    _ensure_local_dir(out_dir)
    return out_dir


def latest_run_dir(code: str, freq: str = "B") -> Optional[str]:
    root = _uri("source=SGS", f"series_code={code}", f"freq={freq}")
    if IS_S3:
        import fsspec
        fs = fsspec.filesystem("s3")
        runs = fs.glob(f"{root}/run=*")
        if not runs:
            return None
        runs = sorted(runs, key=lambda p: p.rsplit("/", 1)[-1])
        return runs[-1]
    else:
        p = Path(root)
        if not p.exists():
            return None
        runs = sorted([x for x in p.iterdir() if x.name.startswith("run=")], key=lambda x: x.name)
        return str(runs[-1]) if runs else None


def save_artifacts(out_dir: str, model_obj, metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, str]:
    model_path = f"{out_dir}/model.pkl"
    metrics_path = f"{out_dir}/metrics.json"
    meta_path = f"{out_dir}/meta.json"

    if IS_S3:
        import fsspec
        with fsspec.open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        with fsspec.open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with fsspec.open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    return {"model": model_path, "metrics": metrics_path, "meta": meta_path}


def load_latest_model(code: str, freq: str = "B"):
    run_dir = latest_run_dir(code, freq)
    if not run_dir:
        return None, None

    model_path = f"{run_dir}/model.pkl"
    metrics_path = f"{run_dir}/metrics.json"

    if IS_S3:
        import fsspec
        with fsspec.open(model_path, "rb") as f:
            model = pickle.load(f)
        metrics: Dict[str, Any] = {}
        try:
            with fsspec.open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
        return model, metrics
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        metrics: Dict[str, Any] = {}
        if Path(metrics_path).exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        return model, metrics
