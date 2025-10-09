# src/data/collectors/sgs_client.py
from __future__ import annotations

from typing import List, Dict, Any
import io
import time
import requests
import pandas as pd

BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


def _to_br(iso_yyyy_mm_dd: str) -> str:
    """Converte 'YYYY-MM-DD' -> 'DD/MM/YYYY' (formato aceito pelo SGS)."""
    y, m, d = iso_yyyy_mm_dd.split("-")
    return f"{d}/{m}/{y}"


def _fetch_json(code: str, start: str, end: str) -> list[dict]:
    """
    Chama o endpoint JSON do SGS e retorna lista de dicts brutos:
    [{"data": "DD/MM/AAAA", "valor": "1,2345"}, ...]
    """
    url = BASE.format(code=code)
    params = {"formato": "json", "dataInicial": _to_br(start), "dataFinal": _to_br(end)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _fetch_csv(code: str, start: str, end: str) -> list[dict]:
    """
    Fallback via CSV quando JSON retorna 404/406/5xx.
    """
    url = BASE.format(code=code)
    params = {"formato": "csv", "dataInicial": _to_br(start), "dataFinal": _to_br(end)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    # CSV do SGS usa ';' e vírgula decimal
    df = pd.read_csv(io.StringIO(r.text), sep=";", decimal=",")
    # colunas típicas: data/valor
    out = df.rename(columns={"data": "data", "valor": "valor"}).to_dict(orient="records")
    return out


def _parse_val(vs: str) -> float:
    """
    Converte string numérica do SGS para float sem inflar (evita 20k→20).
    - "5,1234"    -> 5.1234
    - "1.234,56"  -> 1234.56  (remove ponto como milhar e troca vírgula por ponto)
    - "5.1234"    -> 5.1234   (já com ponto decimal; não mexe)
    """
    s = str(vs).strip()
    if not s:
        raise ValueError("empty value")

    # Se tem vírgula decimal (pt-BR): remove separador de milhar e troca vírgula por ponto
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    # Caso já venha com ponto como decimal, apenas converte
    return float(s)


def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Retorna *sempre* no formato normalizado:
        [{"ts": "YYYY-MM-DD", "value": float}, ...]
    Estratégias:
      - fatiamento por ano (reduz payload e minimiza 406)
      - retries com backoff exponencial simples
      - fallback para CSV quando JSON falha (404/406/5xx)
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
            time.sleep(1.2 * (attempt + 1))  # backoff simples

        if rows_raw is None:
            raise RuntimeError(f"SGS fetch failed for {code} {y_start}->{y_end}: {last_err}")

        # normaliza (data BR → ISO; valor string → float)
        for r in rows_raw:
            ds = (r.get("data") or r.get("Data") or "").strip()
            vs = (r.get("valor") or r.get("Valor") or "").strip()

            try:
                d, m, y = ds.split("/")
                ts = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                value = _parse_val(vs)
            except Exception:
                continue

            results.append({"ts": ts, "value": value})

    if not results:
        return []

    df = pd.DataFrame(results).drop_duplicates(subset=["ts"]).sort_values("ts")
    return df.to_dict(orient="records")
