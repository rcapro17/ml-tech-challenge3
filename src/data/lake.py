"""
Data Lake ops (S3 ou local) para séries temporais em Parquet.
- Escreve/ lê usando fsspec (S3 quando DATA_URI começa com s3://).
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
    uri = (os.getenv("DATA_URI") or "data").rstrip("/")
    return uri


def _to_df(observations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aceita registros no formato {'ts': 'YYYY-MM-DD', 'value': float} OU {'date': ..., 'value': ...}
    e retorna DataFrame com colunas ['ts','value'] (ts datetime64[ns]).
    """
    if not observations:
        return pd.DataFrame(columns=["ts", "value"])

    # normaliza chaves
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

    Retorna quantidade de linhas escritas (0 se nada).
    """
    df = _to_df(observations)
    if df.empty:
        return 0

    data_root = _get_data_root()
    # caminho lógico no lake
    out_path = join_uri(data_root, f"raw/source=SGS/{code}.parquet")

    # usa fsspec para abrir/grav ar em qualquer backend (S3/local)
    fs = get_fs_for_uri(data_root)

    # pandas aceita file-like buffer; aqui fazemos to_parquet para BytesIO e gravamos via fs.open
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)

    with fs.open(out_path, "wb") as f:
        f.write(buf.read())

    # confirmação leve (não levanta exceção se Listing for bloqueado, então só retorna len(df))
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
        # garante tipos/ordem
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts").drop_duplicates(subset=["ts"])
        return df[["ts", "value"]]
    except Exception as e:
        print(f"[lake] read error for {code}: {e}", flush=True)
        return pd.DataFrame(columns=["ts", "value"])
