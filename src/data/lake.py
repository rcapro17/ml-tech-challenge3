"""
Data Lake ops (S3 ou local) para séries temporais em Parquet.
- Escreve/lê usando fsspec (S3 quando DATA_URI começa com s3://).
- Normaliza observações aceitando 'ts' ou 'date' e escreve como ['ts','value'].
"""
from __future__ import annotations

from typing import List, Dict, Any
import os
from io import BytesIO
import pandas as pd

from src.storage.s3util import get_fs_for_uri, join_uri


def _get_data_root() -> str:
    """
    Retorna o root do lake:
      - S3: s3://bucket/prefix (ex.: s3://ml-tech-challenge3-xxx/prod)
      - Local fallback: 'data'
    """
    return (os.getenv("DATA_URI") or "data").rstrip("/")


def _to_df(observations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aceita registros {'ts': 'YYYY-MM-DD', 'value': float} OU {'date': ..., 'value': ...}
    e retorna DataFrame com ['ts','value'].
    """
    if not observations:
        return pd.DataFrame(columns=["ts", "value"])

    recs = []
    for r in observations:
        ts = r.get("ts") or r.get("date")
        val = r.get("value")
        if ts is None or val is None:
            continue
        recs.append({"ts": ts, "value": val})

    if not recs:
        return pd.DataFrame(columns=["ts", "value"])

    df = pd.DataFrame(recs)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    return df[["ts", "value"]]


def write_sgs_parquet(code: str, observations: List[Dict[str, Any]]) -> int:
    """
    Escreve Parquet único por série:
      {DATA_URI}/raw/source=SGS/{code}.parquet
    Retorna quantidade de linhas escritas.
    """
    df = _to_df(observations)
    if df.empty:
        return 0

    data_root = _get_data_root()
    out_path = join_uri(data_root, f"raw/source=SGS/{code}.parquet")
    fs = get_fs_for_uri(data_root)

    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    with fs.open(out_path, "wb") as f:
        f.write(buf.read())
    return int(len(df))


def read_sgs_parquet(code: str) -> pd.DataFrame:
    """
    Lê o Parquet do lake se existir. Caso contrário, retorna DF vazio.
      {DATA_URI}/raw/source=SGS/{code}.parquet
    """
    data_root = _get_data_root()
    in_path = join_uri(data_root, f"raw/source=SGS/{code}.parquet")
    fs = get_fs_for_uri(data_root)

    try:
        if not fs.exists(in_path):
            return pd.DataFrame(columns=["ts", "value"])
        with fs.open(in_path, "rb") as f:
            df = pd.read_parquet(f, engine="pyarrow")
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts").drop_duplicates(subset=["ts"])
        return df[["ts", "value"]]
    except Exception as e:
        print(f"[lake] read error for {code}: {e}", flush=True)
        return pd.DataFrame(columns=["ts", "value"])
