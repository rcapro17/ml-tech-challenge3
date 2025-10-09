"""
Data Lake operations (local folder or S3) for time series data in Parquet
- Writes/reads normalized schema: columns ['ts', 'value']
- Destination is controlled by DATA_URI (env var):
    - local (default): "data"
    - S3: "s3://<bucket>/<prefix>"
  Final layout:
    <DATA_URI>/raw/source=SGS/{code}.parquet
"""
from __future__ import annotations

import os
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd


DATA_URI = os.environ.get("DATA_URI", "data")
IS_S3 = DATA_URI.startswith("s3://")


def _uri(*parts: str) -> str:
    """
    Join parts on DATA_URI. Returns s3://... if DATA_URI is S3, otherwise local path string.
    """
    if IS_S3:
        base = DATA_URI.rstrip("/")
        suffix = "/".join(p.strip("/\\") for p in parts if p)
        return f"{base}/{suffix}"
    else:
        base = Path(DATA_URI)
        return str(base.joinpath(*parts))


def _ensure_local_parent(path_str: str) -> None:
    """
    For local filesystem, ensure parent directory exists. On S3 it's a no-op.
    """
    if IS_S3:
        return
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def _normalize_rows(observations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize input rows that may contain 'ts' or 'date' -> DataFrame with ['ts','value'].
    """
    if not observations:
        return pd.DataFrame(columns=["ts", "value"])

    recs: list[dict] = []
    for r in observations:
        ts = r.get("ts") or r.get("date")
        val = r.get("value")
        if ts and val is not None:
            recs.append({"ts": ts, "value": float(val)})

    if not recs:
        return pd.DataFrame(columns=["ts", "value"])

    df = pd.DataFrame(recs)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def write_sgs_parquet(code: str, observations: List[Dict[str, Any]], base_dir: str = "data/raw") -> int:
    """
    Write time series observations to Parquet in the lake.
    NOTE: Even if base_dir is passed (for backward-compat), when DATA_URI points to S3
    the target will be <DATA_URI>/raw/...

    Args:
        code: Series identifier
        observations: list with 'ts' or 'date', and 'value'
        base_dir: kept for backward compatibility; on local DATA_URI it is ignored
                  and we always save under <DATA_URI>/raw/...

    Returns:
        number of rows written (int)
    """
    df = _normalize_rows(observations)
    if df.empty:
        return 0

    out_path = _uri("raw", "source=SGS", f"{code}.parquet")
    _ensure_local_parent(out_path)
    # pandas will use s3fs seamlessly if path is s3://
    df.to_parquet(out_path, index=False)
    return len(df)


def read_sgs_parquet(code: str) -> pd.DataFrame:
    """
    Read time series observations from Parquet in the lake.
    Returns empty DataFrame if not found.
    """
    path = _uri("raw", "source=SGS", f"{code}.parquet")
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["ts", "value"])
    except Exception:
        # On S3, missing file may raise generic errors—hide and return empty
        return pd.DataFrame(columns=["ts", "value"])
