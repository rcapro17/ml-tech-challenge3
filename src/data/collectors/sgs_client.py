# src/data/collectors/sgs_client.py
from __future__ import annotations
import time
from datetime import date
from typing import List, Dict, Any
import requests
import pandas as pd

BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# sessão com retry básico
_session = requests.Session()
_session.headers.update({
    "Accept": "application/json",
    # alguns proxies/CDNs do BCB rejeitam UA “genérico”; um UA de browser ajuda
    "User-Agent": "Mozilla/5.0 (compatible; ml-tech-challenge/1.0; +https://onrender.com)"
})

def _brfmt(d: str) -> str:
    # espera "YYYY-MM-DD" e devolve "DD/MM/YYYY"
    y, m, d_ = d.split("-")
    return f"{d_}/{m}/{y}"

def _fetch_json(code: str, start: str, end: str, timeout: int = 30) -> List[Dict[str, Any]]:
    url = BASE.format(code=code)
    params = {"formato": "json", "dataInicial": _brfmt(start), "dataFinal": _brfmt(end)}
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()  # se não for 2xx -> Exception
    return r.json()

def _fetch_csv(code: str, start: str, end: str, timeout: int = 30) -> List[Dict[str, Any]]:
    # fallback quando JSON retorna 406/404 etc.
    url = BASE.format(code=code)
    params = {"formato": "csv", "dataInicial": _brfmt(start), "dataFinal": _brfmt(end)}
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    # CSV do BCB vem com ; e decimal , e colunas "data" e "valor"
    df = pd.read_csv(
        pd.compat.StringIO(r.text),
        sep=";",
        decimal=",",
        dtype={"data": "string", "valor": "string"},
    )
    if df.empty:
        return []
    # normaliza para o mesmo formato do JSON
    out = []
    for _, row in df.iterrows():
        out.append({"data": row["data"], "valor": row["valor"]})
    return out

def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Retorna uma lista de dicts como:
      [{"ts": "YYYY-MM-DD", "value": float}, ...]
    com retries, fallback para CSV e fatiamento anual para reduzir chance de 406.
    """
    # fatiar por anos (evita respostas muito grandes e alguns 406 esquisitos)
    start_y = int(start[:4])
    end_y = int(end[:4])

    results: List[Dict[str, Any]] = []
    for year in range(start_y, end_y + 1):
        y_start = f"{year}-01-01" if year > start_y else start
        y_end = f"{year}-12-31" if year < end_y else end

        # retries com backoff simples
        last_err = None
        for attempt in range(4):
            try:
                rows = _fetch_json(code, y_start, y_end)
                break
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # se 406/404/5xx, tenta fallback para CSV
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
            time.sleep(1.5 * (attempt + 1))  # backoff

        else:
            # esgotou retries
            raise RuntimeError(f"SGS fetch failed for {code} {y_start}->{y_end}: {last_err}")

        # normaliza campos: JSON vem como "data": "DD/MM/YYYY", "valor": "N,NN"
        for r in rows:
            ds = r.get("data") or r.get("Data") or ""
            vs = r.get("valor") or r.get("Valor") or ""
            # converte data pt-BR para ISO
            try:
                d, m, y = ds.split("/")
                iso = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            except Exception:
                continue
            # valor para float (troca vírgula por ponto)
            try:
                val = float(str(vs).replace(".", "").replace(",", "."))
            except Exception:
                continue
            results.append({"ts": iso, "value": val})

    # ordena por data e remove duplicatas se houver
    if results:
        df = pd.DataFrame(results).drop_duplicates(subset=["ts"]).sort_values("ts")
        return df.to_dict(orient="records")
    return results
