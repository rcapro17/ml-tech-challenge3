# src/data/collectors/sgs_client.py
from __future__ import annotations

from typing import List, Dict, Any
import io
import time
from datetime import date
import requests
import pandas as pd


BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


def _fetch_json(code: str, start: str, end: str) -> list[dict]:
    """
    Chama o endpoint JSON do SGS e retorna lista de dicts brutos:
    [{"data": "DD/MM/AAAA", "valor": "1,2345"}, ...]
    """
    url = BASE.format(code=code)
    params = {
        "formato": "json",
        # SGS usa datas BR:
        "dataInicial": _to_br(start),
        "dataFinal": _to_br(end),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _fetch_csv(code: str, start: str, end: str) -> list[dict]:
    """
    Fallback em CSV (se 406/404/5xx no JSON).
    """
    url = BASE.format(code=code)
    params = {
        "formato": "csv",
        "dataInicial": _to_br(start),
        "dataFinal": _to_br(end),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    # CSV do SGS usa ; e vírgula decimal
    df = pd.read_csv(io.StringIO(r.text), sep=";", decimal=",")
    # colunas costumam vir como 'data' e 'valor'
    out = df.rename(columns={"data": "data", "valor": "valor"}).to_dict(orient="records")
    return out


def _to_br(iso_yyyy_mm_dd: str) -> str:
    y, m, d = iso_yyyy_mm_dd.split("-")
    return f"{d}/{m}/{y}"


def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Retorna *sempre* no formato normalizado:
        [{"ts": "YYYY-MM-DD", "value": float}, ...]
    Faz fatiamento por ano + retries e fallback CSV (evita 406).
    """
    start_y = int(start[:4])
    end_y = int(end[:4])

    results: list[dict] = []
    for year in range(start_y, end_y + 1):
        y_start = f"{year}-01-01" if year > start_y else start
        y_end = f"{year}-12-31" if year < end_y else end

        rows_raw: list[dict] | None = None
        last_err: Exception | None = None

        for attempt in range(4):
            try:
                rows_raw = _fetch_json(code, y_start, y_end)
                break
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (404, 406) or (status and status >= 500):
                    try:
                        rows_raw = _fetch_csv(code, y_start, y_end)
                        break
                    except Exception as e2:
                        last_err = e2
                else:
                    last_err = e
            except Exception as e:
                last_err = e
            time.sleep(1.2 * (attempt + 1))

        if rows_raw is None:
            raise RuntimeError(f"SGS fetch failed for {code} {y_start}->{y_end}: {last_err}")

        # normaliza para ts/value
        for r in rows_raw:
            ds = (r.get("data") or r.get("Data") or "").strip()
            vs = (r.get("valor") or r.get("Valor") or "").strip()

            # data BR -> ISO
            try:
                d, m, y = ds.split("/")
                ts = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            except Exception:
                continue

            # vírgula decimal -> ponto; remove separador de milhar se houver
            try:
                # casos típicos: "5,3498" -> "5.3498"; "1.234,56" -> "1234.56"
                clean = vs.replace(".", "").replace(",", ".")
                value = float(clean)
            except Exception:
                continue

            results.append({"ts": ts, "value": value})

    if not results:
        return []

    df = pd.DataFrame(results).drop_duplicates(subset=["ts"]).sort_values("ts")
    return df.to_dict(orient="records")
