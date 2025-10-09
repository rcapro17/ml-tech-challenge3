from __future__ import annotations

import os, json, time, pickle
from typing import Optional, Dict, Any, List
from pathlib import Path
import fsspec

from src.storage.s3util import get_fs_for_uri, join_uri

# Onde salvar/carregar modelos:
# - Se MODELS_URI= s3://bucket/prod/models -> usa S3
# - Caso contrário, usa local 'data/models'
MODELS_URI = (os.environ.get("MODELS_URI") or "data/models").strip().rstrip("/")

def _fs_and_root(code: str, freq: str):
    """
    Retorna filesystem e caminho raiz para a série/freq.
    Suporta S3 e local.
    """
    root = join_uri(MODELS_URI, "source=SGS", f"series_code={code}", f"freq={freq}")
    fs = get_fs_for_uri(root)
    return fs, root

def _ensure_dir(fs, path: str) -> None:
    # cria "diretórios" no S3 (na prática chaves com /) e diretórios locais
    if getattr(fs, "makedirs", None):
        fs.makedirs(path, exist_ok=True)
    else:
        # local fallback
        Path(path).mkdir(parents=True, exist_ok=True)

def make_run_dir(code: str, freq: str = "B") -> str:
    fs, root = _fs_and_root(code, freq)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = join_uri(root, f"run={run_id}")
    try:
        _ensure_dir(fs, run_dir)
        return run_dir
    except Exception as e:
        print(f"[registry] WARN: ensure_dir({run_dir}) failed: {e}. Falling back to local.", flush=True)
        local_root = Path("data/models") / "source=SGS" / f"series_code={code}" / f"freq={freq}" / f"run={run_id}"
        local_root.mkdir(parents=True, exist_ok=True)
        return str(local_root)

def latest_run_dir(code: str, freq: str = "B") -> Optional[str]:
    # Tenta no backend configurado
    fs, root = _fs_and_root(code, freq)
    try:
        runs = fs.glob(f"{root}/run=*")
        if runs:
            runs = sorted(runs)
            return runs[-1]
    except Exception as e:
        print(f"[registry] WARN: glob on {root} failed: {e}. Trying local fallback...", flush=True)

    # Fallback local
    local_root = Path("data/models") / "source=SGS" / f"series_code={code}" / f"freq={freq}"
    if local_root.exists():
        candidates = sorted([p for p in local_root.iterdir() if p.name.startswith("run=")], key=lambda p: p.name)
        return str(candidates[-1]) if candidates else None
    return None

def _open_write(fs, path: str):
    return fs.open(path, "wb")

def _open_read(fs, path: str):
    return fs.open(path, "rb")

def save_artifacts(out_dir: str, model_obj, metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Grava model.pkl, metrics.json, meta.json no 'out_dir' (S3 ou local).
    """
    fs = get_fs_for_uri(out_dir)
    try:
        with _open_write(fs, join_uri(out_dir, "model.pkl")) as f:
            pickle.dump(model_obj, f)
        with _open_write(fs, join_uri(out_dir, "metrics.json")) as f:
            f.write(json.dumps(metrics, indent=2).encode("utf-8"))
        with _open_write(fs, join_uri(out_dir, "meta.json")) as f:
            f.write(json.dumps(meta, indent=2, default=str).encode("utf-8"))
        return {
            "model": join_uri(out_dir, "model.pkl"),
            "metrics": join_uri(out_dir, "metrics.json"),
            "meta": join_uri(out_dir, "meta.json"),
        }
    except Exception as e:
        print(f"[registry] WARN: save_artifacts to {out_dir} failed: {e}. Falling back to local.", flush=True)
        # fallback local
        local = Path("data/models") / Path(out_dir).name if "://" in out_dir else Path(out_dir)
        local.mkdir(parents=True, exist_ok=True)
        with open(local / "model.pkl", "wb") as f:
            pickle.dump(model_obj, f)
        with open(local / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(local / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return {
            "model": str(local / "model.pkl"),
            "metrics": str(local / "metrics.json"),
            "meta": str(local / "meta.json"),
        }

def load_latest_model(code: str, freq: str = "B"):
    """
    Carrega último modelo. Tenta no backend configurado; se falhar, tenta local.
    """
    run_dir = latest_run_dir(code, freq)
    if not run_dir:
        return None, None

    fs = get_fs_for_uri(run_dir)
    model_path = join_uri(run_dir, "model.pkl")
    metrics_path = join_uri(run_dir, "metrics.json")

    # tenta backend
    try:
        with _open_read(fs, model_path) as f:
            model = pickle.load(f)
        metrics = {}
        if fs.exists(metrics_path):
            with _open_read(fs, metrics_path) as f:
                metrics = json.loads(f.read().decode("utf-8"))
        return model, metrics
    except Exception as e:
        print(f"[registry] WARN: load from {run_dir} failed: {e}. Trying local fallback...", flush=True)

    # fallback local
    p = Path(run_dir)
    if not p.exists():
        return None, None
    with open(p / "model.pkl", "rb") as f:
        model = pickle.load(f)
    metrics = {}
    mp = p / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            metrics = json.load(f)
    return model, metrics
