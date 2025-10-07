from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.ml.metrics import mae, rmse, mape

def fit_sarimax(y: pd.Series,
                order=(1,1,1),
                seasonal_order=(0,0,0,0)) -> SARIMAX:
    """
    Treina SARIMAX em uma série univariada 'y' (index = datetime).
    """
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=1000, method="lbfgs")
    return res

def forecast(res, steps: int) -> np.ndarray:
    fc = res.forecast(steps=steps)
    return fc.values if hasattr(fc, "values") else np.array(fc)

def rolling_backtest(y: pd.Series,
                     order=(1,1,1),
                     seasonal_order=(0,0,0,0),
                     horizon=5,
                     initial_train_ratio=0.7) -> Dict:
    """
    Rolling-origin backtest: treina até t e prevê próximos 'horizon'.
    Retorna métricas agregadas e últimos forecasts.
    """
    n = len(y)
    start = int(n * initial_train_ratio)
    preds: List[float] = []
    trues: List[float] = []
    last_fc = None

    for cut in range(start, n - horizon + 1):
        y_train = y.iloc[:cut]
        y_test  = y.iloc[cut:cut + horizon]
        res = fit_sarimax(y_train, order=order, seasonal_order=seasonal_order)
        fc = forecast(res, steps=horizon)
        preds.extend(fc)
        trues.extend(y_test.values)
        last_fc = (y_test.index, fc)

    metrics = {
        "MAE": mae(trues, preds),
        "RMSE": rmse(trues, preds),
        "MAPE": mape(trues, preds),
        "n_fold_points": len(trues)
    }
    return {"metrics": metrics, "last_fold": last_fc}
