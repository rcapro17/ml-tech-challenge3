"""
Data Lake operations for time series data
Handles file-based storage in Parquet format (local or S3 via fsspec)
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import fsspec

from src.storage.s3util import get_fs_for_uri, join_uri

# DATA_URI define a raiz do lake (ex.: s3://bucket/prod ou data)
DATA_URI = (os.environ.get("DATA_URI") or "data").strip().rstrip("/")

def _write_parquet_df(df: pd.DataFrame, url: str) -> None:
    fs = get_fs_for_uri(url)
    with fs.open(url, "wb") as f:
        df.to_parquet(f, index=False)

def _read_parquet_df(url: str) -> pd.DataFrame:
    fs = get_fs_for_uri(url)
    if not fs.exists(url):
        return pd.DataFrame()
    with fs.open(url, "rb") as f:
        return pd.read_parquet(f)

def write_sgs_parquet(code: str, observations: List[Dict[str, Any]]) -> int:
    """
    Escreve parquet único por série:
    {DATA_URI}/raw/source=SGS/{code}.parquet
    """
    if not observations:
        return 0

    # normaliza chaves -> ts/value
    rows = []
    for r in observations:
        ts = r.get("ts") or r.get("date")
        val = r.get("value")
        if ts and val is not None:
            rows.append({"ts": ts, "value": val})
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")

    url = join_uri(DATA_URI, "raw", "source=SGS", f"{code}.parquet")

    try:
        _write_parquet_df(df, url)
        print(f"[lake] wrote {len(df)} rows → {url}", flush=True)
        return len(df)
    except Exception as e:
        # fallback para disco local em caso de falha no S3
        print(f"[lake] WARN: write to {url} failed: {e}. Falling back to local 'data/raw'...", flush=True)
        local_dir = Path("data/raw/source=SGS")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{code}.parquet"
        df.to_parquet(local_path, index=False)
        return len(df)

def read_sgs_parquet(code: str) -> pd.DataFrame:
    """
    Lê parquet {DATA_URI}/raw/source=SGS/{code}.parquet (S3 ou local).
    """
    url = join_uri(DATA_URI, "raw", "source=SGS", f"{code}.parquet")
    try:
        df = _read_parquet_df(url)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception as e:
        print(f"[lake] WARN: read from {url} failed: {e}. Trying local fallback...", flush=True)
        # fallback local
        p = Path("data/raw/source=SGS") / f"{code}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"])
            return df
        return pd.DataFrame()
