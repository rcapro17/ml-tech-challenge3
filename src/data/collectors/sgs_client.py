# src/data/collectors/sgs_client.py
from __future__ import annotations

from typing import Any, Dict, List
from decimal import Decimal, InvalidOperation
from io import StringIO
import time

import pandas as pd
import requests


def _iso_to_br(iso: str) -> str:
    """
    Converte 'YYYY-MM-DD' -> 'DD/MM/YYYY'
    """
    y, m, d = iso.split("-")
    return f"{d.zfill(2)}/{m.zfill(2)}/{y}"


def _parse_brazil_number(vs: str) -> float:
    """
    Converte string numérica do SGS para float, lidando com:
      - formato pt-BR: '5.349,80' (vírgula decimal, ponto milhar)
      - formato EN: '5.3498' (ponto decimal)
    Regra:
      - Se houver vírgula: é decimal; remova pontos e troque vírgula por ponto.
      - Se não houver vírgula: não mexa no ponto (pode ser decimal legítimo).
    """
    if vs is None:
        raise ValueError("empty")
    s = str(vs).strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(Decimal(s))
    except InvalidOperation as e:
        raise ValueError(f"bad number: {vs}") from e


def _fetch_json(code: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    """
    Baixa JSON do SGS (um intervalo) e retorna lista de dicts com chaves 'data','valor'.
    NÃO normaliza aqui (normalização é feita em fetch_sgs_series).
    """
    start_br = _iso_to_br(start_iso)
    end_br = _iso_to_br(end_iso)
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=json&dataInicial={start_br}&dataFinal={end_br}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    if not isinstance(rows, list):
        raise ValueError("unexpected JSON payload")
    # rows: [{'data': 'DD/MM/YYYY', 'valor': 'x,yz'}, ...]
    return rows


def _fetch_csv(code: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    """
    Baixa CSV do SGS (um intervalo) e retorna lista de dicts com chaves normalizadas
    (data/valor em minúsculas).
    """
    start_br = _iso_to_br(start_iso)
    end_br = _iso_to_br(end_iso)
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=csv&dataInicial={start_br}&dataFinal={end_br}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # CSV SGS: separador ';' e decimal ','
    df = pd.read_csv(StringIO(resp.text), sep=";", decimal=",")
    # padroniza nomes
    df.columns = [c.strip().lower() for c in df.columns]
    # retorna [{'data': 'DD/MM/YYYY', 'valor': 5.3498}, ...]
    return df.to_dict(orient="records")


def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Retorna uma lista de dicts: [{'ts': 'YYYY-MM-DD', 'value': float}, ...]
    - Fatiamento anual para reduzir erros de 406/timeout.
    - Retries com backoff exponencial simples.
    - Fallback para CSV quando 406/404/5xx.
    """
    start_y = int(start[:4])
    end_y = int(end[:4])

    results: List[Dict[str, Any]] = []

    for year in range(start_y, end_y + 1):
        y_start = f"{year}-01-01" if year > start_y else start
        y_end = f"{year}-12-31" if year < end_y else end

        last_err: Exception | None = None
        rows: List[Dict[str, Any]] | None = None

        # Até 4 tentativas por fatia (JSON -> CSV fallback)
        for attempt in range(4):
            try:
                rows = _fetch_json(code, y_start, y_end)
                break
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (406, 404) or (status and 500 <= status < 600):
                    try:
                        rows = _fetch_csv(code, y_start, y_end)
                        break
                    except Exception as e2:
                        last_err = e2
                else:
                    last_err = e
            except Exception as e:
                last_err = e

            # backoff simples
            time.sleep(1.5 * (attempt + 1))

        if rows is None:
            # esgotou as tentativas nesta fatia
            raise RuntimeError(f"SGS fetch failed for {code} {y_start}->{y_end}: {last_err}")

        # Normalização final para [{'ts', 'value'}]
        for r in rows:
            ds = r.get("data") or r.get("Data") or ""
            vs = r.get("valor") or r.get("Valor") or r.get("value")
            # converte data pt-BR -> ISO
            try:
                d, m, y = str(ds).split("/")
                iso = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            except Exception:
                continue
            # valor
            try:
                val = _parse_brazil_number(vs)
            except Exception:
                continue
            results.append({"ts": iso, "value": val})

    if not results:
        return results

    # remove duplicatas por 'ts' e ordena
    df = pd.DataFrame(results).drop_duplicates(subset=["ts"]).sort_values("ts")
    return df.to_dict(orient="records")
