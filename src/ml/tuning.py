import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.ml.etl import load_series
from src.ml.models.sarimax import fit_sarimax, forecast, rolling_backtest

warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class Candidate:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]

def generate_grid() -> List[Candidate]:
    # Grid enxuto e robusto
    orders = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    seasonal = [(0,0,0,0)]  # câmbio diário costuma não precisar sazonalidade explícita
    return [Candidate(o, s) for o in orders for s in seasonal]

def tune_sarimax_for_code(
    code: str,
    freq: str = "B",
    horizon: int = 5,
    initial_train_ratio: float = 0.7,
    grid: Iterable[Candidate] | None = None
) -> Dict[str, Any]:
    """
    Carrega série, roda backtesting para cada candidato e escolhe menor RMSE.
    Retorna dict com 'best', 'results' (lista) e 'y_len'.
    """
    df = load_series(code)
    if df.empty:
        return {"ok": False, "error": f"sem dados para série {code}"}

    y = (df.set_index("ts")["value"]
           .asfreq(freq, method="ffill")
           .dropna())
    n = len(y)
    if n < 50:
        return {"ok": False, "error": "dados insuficientes para tuning (>=50 pontos)"}

    cand = list(grid or generate_grid())
    results: List[Dict[str, Any]] = []
    best = None

    for c in cand:
        try:
            bt = rolling_backtest(y, order=c.order, seasonal_order=c.seasonal_order,
                                  horizon=horizon, initial_train_ratio=initial_train_ratio)
            metrics = bt["metrics"]
            rmse = metrics["RMSE"]
            results.append({
                "order": c.order,
                "seasonal_order": c.seasonal_order,
                "metrics": metrics
            })
            if (best is None) or (rmse < best["metrics"]["RMSE"]):
                # treina final no full y para guardar modelo vencedor
                model = fit_sarimax(y, order=c.order, seasonal_order=c.seasonal_order)
                best = {
                    "order": c.order,
                    "seasonal_order": c.seasonal_order,
                    "metrics": metrics,
                    "model": model,
                }
        except Exception as e:
            results.append({
                "order": c.order,
                "seasonal_order": c.seasonal_order,
                "error": str(e)
            })

    results = sorted(results, key=lambda r: r["metrics"]["RMSE"] if "metrics" in r else np.inf)
    if best is None:
        return {"ok": False, "error": "nenhum candidato convergiu", "results": results}

    return {"ok": True, "best": best, "results": results, "y_len": n}
