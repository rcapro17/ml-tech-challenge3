# src/data/collectors/sgs_client.py
from __future__ import annotations

from typing import Any, Dict, List
import time
import csv
import io

import pandas as pd
import requests


BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


def _parse_bcb_value(raw: Any) -> float:
    """
    Converte string de valor do BCB para float de forma robusta:
      - "1,7240"  -> 1.7240
      - "1.7240"  -> 1.7240 (NÃO remove ponto quando for separador decimal)
      - "17.240,0" -> 17240.0
    Regras:
      - se tem '.' e ',' => '.' é separador de milhar, ',' é decimal
      - se só tem ','    => ',' é decimal
      - se só tem '.'    => '.' é decimal
    """
    s = str(raw).strip()
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        s = s.replace(".", "").replace(",", ".")
    elif has_comma:
        s = s.replace(",", ".")
    # else: mantém '.'
    return float(s)


def _fetch_json(code: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    params = {
        "formato": "json",
        "dataInicial": f"{start_iso[8:10]}/{start_iso[5:7]}/{start_iso[0:4]}",
        "dataFinal": f"{end_iso[8:10]}/{end_iso[5:7]}/{end_iso[0:4]}",
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "ml-tech-challenge/1.0 (+https://render.com)",
    }
    url = BASE.format(code=code)
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    out: List[Dict[str, Any]] = []
    for row in data:
        ds = (row.get("data") or row.get("Data") or "").strip()
        vs = (row.get("valor") or row.get("Valor") or "").strip()
        # "DD/MM/YYYY" -> "YYYY-MM-DD"
        try:
            d, m, y = ds.split("/")
            iso = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        except Exception:
            continue
        try:
            val = _parse_bcb_value(vs)
        except Exception:
            continue
        out.append({"ts": iso, "value": val})
    return out


def _fetch_csv(code: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    params = {
        "formato": "csv",
        "dataInicial": f"{start_iso[8:10]}/{start_iso[5:7]}/{start_iso[0:4]}",
        "dataFinal": f"{end_iso[8:10]}/{end_iso[5:7]}/{end_iso[0:4]}",
    }
    headers = {
        "Accept": "text/csv",
        "User-Agent": "ml-tech-challenge/1.0 (+https://render.com)",
    }
    url = BASE.format(code=code)
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    text = r.text
    # CSV costuma vir com cabeçalhos "data;valor"
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    out: List[Dict[str, Any]] = []
    for row in reader:
        ds = (row.get("data") or row.get("Data") or "").strip()
        vs = (row.get("valor") or row.get("Valor") or "").strip()
        try:
            d, m, y = ds.split("/")
            iso = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        except Exception:
            continue
        try:
            val = _parse_bcb_value(vs)
        except Exception:
            continue
        out.append({"ts": iso, "value": val})
    return out


def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Retorna uma lista:
        [{"ts":"YYYY-MM-DD","value": float}, ...]
    Estratégias:
      - fatiar por ano (reduz risco de 406/timeout);
      - JSON com Accept adequado; fallback CSV se 406/404/5xx;
      - retries com backoff;
      - normaliza e ordena.
    """
    sy, ey = int(start[:4]), int(end[:4])
    results: List[Dict[str, Any]] = []

    for year in range(sy, ey + 1):
        y_start = f"{year}-01-01" if year > sy else start
        y_end = f"{year}-12-31" if year < ey else end

        last_err: Exception | None = None
        rows: List[Dict[str, Any]] | None = None

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
            time.sleep(1.25 * (attempt + 1))

        if rows is None:
            raise RuntimeError(f"SGS fetch failed for {code} {y_start}->{y_end}: {last_err}")

        results.extend(rows)

    if not results:
        return []

    # normaliza/ordena e remove duplicatas por ts
    df = pd.DataFrame(results).drop_duplicates(subset=["ts"]).sort_values("ts")
    # sanity check: valores muito altos vs taxas
    # (não alteramos o valor; apenas deixamos fácil de debugar se necessário)
    return df.to_dict(orient="records")
