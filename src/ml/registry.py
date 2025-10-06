import os, json, time, pickle
from pathlib import Path
from typing import Optional, Dict, Any

BASE = "data/models"

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def make_run_dir(code: str, freq: str = "B") -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(BASE, "source=SGS", f"series_code={code}", f"freq={freq}", f"run={run_id}")
    _ensure_dir(out_dir)
    return out_dir

def latest_run_dir(code: str, freq: str = "B") -> Optional[str]:
    root = Path(BASE) / "source=SGS" / f"series_code={code}" / f"freq={freq}"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.name.startswith("run=")], key=lambda p: p.name)
    return str(runs[-1]) if runs else None

def save_artifacts(out_dir: str, model_obj, metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, str]:
    model_path = os.path.join(out_dir, "model.pkl")
    metrics_path = os.path.join(out_dir, "metrics.json")
    meta_path = os.path.join(out_dir, "meta.json")
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
    model_path = os.path.join(run_dir, "model.pkl")
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, metrics
