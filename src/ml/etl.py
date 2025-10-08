import pandas as pd
from sqlalchemy import text
from src.db import get_engine


from typing import Optional
import pandas as pd
from sqlalchemy import text
from src.db import get_engine

def load_series(code: str) -> pd.DataFrame:
    """
    Lê a série do warehouse com o schema novo:
    observations(code VARCHAR, ts DATE, value NUMERIC)
    Retorna DataFrame com colunas ['ts','value'].
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ts, value
                FROM observations
                WHERE code = :code
                ORDER BY ts
            """),
            conn,
            params={"code": str(code)},
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def make_supervised(df: pd.DataFrame, freq: str = "B", lags: int = 7) -> pd.DataFrame:
    """
    Resample para frequência desejada (padrão B = dias úteis), forward-fill valores,
    e cria lags simples como features (lag1..lagN).
    """
    if df.empty:
        return df
    s = (df.set_index("ts")["value"]
           .resample(freq).ffill()
           .rename("value"))
    out = pd.DataFrame({"value": s})
    for k in range(1, lags + 1):
        out[f"lag{k}"] = out["value"].shift(k)
    out = out.dropna().reset_index().rename(columns={"index": "ts"})
    return out
