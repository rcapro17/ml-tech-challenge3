import pandas as pd
from typing import Dict, Any
from src.ml.etl import load_series
from src.ml.models.sarimax import fit_sarimax, forecast, rolling_backtest

def train_and_backtest(code: str,
                       freq: str = "B",
                       order=(1,1,1),
                       seasonal_order=(0,0,0,0),
                       backtest_h=5) -> Dict[str, Any]:
    df = load_series(code)
    if df.empty:
        return {"ok": False, "error": f"sem dados para série {code}"}

    y = (df.set_index("ts")["value"]
           .asfreq(freq, method="ffill")
           .dropna())

    if len(y) < 30:
        return {"ok": False, "error": "dados insuficientes para backtest (mínimo ~30 pontos)"}

    bt = rolling_backtest(y, order=order, seasonal_order=seasonal_order, horizon=backtest_h)
    res = fit_sarimax(y, order=order, seasonal_order=seasonal_order)
    return {"ok": True, "metrics": bt["metrics"], "n_obs": int(len(y)), "model": res}

def forecast_next(model, steps: int = 5) -> pd.DataFrame:
    fc = forecast(model, steps=steps)
    idx = pd.date_range(start=model.data.row_labels[-1], periods=steps+1, freq=model.model._index_freq, inclusive="neither")
    return pd.DataFrame({"ts": idx, "forecast": fc})
