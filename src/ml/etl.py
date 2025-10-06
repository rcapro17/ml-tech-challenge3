import pandas as pd
from sqlalchemy import text
from src.db import get_engine

def load_series(code: str) -> pd.DataFrame:
    """
    Lê a série 'code' da tabela observations e retorna DataFrame com colunas:
    ts (datetime64[ns, UTC-naive]) e value (float).
    """
    eng = get_engine()
    query = text("""
        SELECT ts, value
        FROM observations
        WHERE series_code = :code
        ORDER BY ts
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, params={"code": code})

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.dropna(subset=["value"]).sort_values("ts").reset_index(drop=True)
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
