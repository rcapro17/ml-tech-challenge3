# src/ml/tuning.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.ml.etl import load_series
from src.ml.models.sarimax import fit_sarimax, rolling_backtest

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class Candidate:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]


def _grid(fast: bool) -> List[Candidate]:
    """Gera uma grade pequena no modo rápido e um pouco maior no modo completo."""
    if fast:
        orders = [(1, 1, 1), (1, 1, 2), (2, 1, 1)]
    else:
        orders = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (3, 1, 1)]
    seasonal = [(0, 0, 0, 0)]  # câmbio diário geralmente sem sazonalidade explícita
    return [Candidate(o, s) for o in orders for s in seasonal]


def _trim_series(y: pd.Series, fast: bool, max_points_fast: int = 800, max_points_full: int = 1500) -> pd.Series:
    """
    Para acelerar: mantém apenas o trecho mais recente.
    - fast: ~800 dias úteis (~3 anos+)
    - full: ~1500 dias úteis (~6 anos+)
    """
    cap = max_points_fast if fast else max_points_full
    if len(y) > cap:
        return y.iloc[-cap:]
    return y


def tune_sarimax_for_code(
    code: str,
    freq: str = "B",
    horizon: int = 5,
    initial_train_ratio: float = 0.7,
    grid: Optional[Iterable[Candidate]] = None,
    fast: bool = True,
) -> Dict[str, Any]:
    """
    Carrega a série, faz backtest rolling nos candidatos e escolhe o menor RMSE.
    Retorna:
      {"ok": True, "best": {...}, "results": [...], "y_len": n}
    """
    df = load_series(code)
    if df is None or df.empty:
        return {"ok": False, "error": f"sem dados para série {code}"}

    # Série contínua na freq desejada
    y = (
        df.set_index("ts")["value"]
        .asfreq(freq, method="ffill")
        .dropna()
    )

    # Corta histórico muito antigo para acelerar
    y = _trim_series(y, fast=fast)

    n = len(y)
    if n < max(50, horizon * 10):
        return {"ok": False, "error": f"dados insuficientes para tuning (n={n})"}

    cand = list(grid or _grid(fast=fast))
    results: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    # Backtest por candidato
    for c in cand:
        try:
            bt = rolling_backtest(
                y,
                order=c.order,
                seasonal_order=c.seasonal_order,
                horizon=horizon,
                initial_train_ratio=max(0.6, min(0.9, initial_train_ratio)),
            )
            metrics = bt["metrics"]
            rmse = metrics.get("RMSE", np.inf)

            results.append(
                {"order": c.order, "seasonal_order": c.seasonal_order, "metrics": metrics}
            )

            if best is None or rmse < best["metrics"]["RMSE"]:
                # Treina final em todo y para salvar
                model = fit_sarimax(
                    y,
                    order=c.order,
                    seasonal_order=c.seasonal_order,
                )
                best = {
                    "order": c.order,
                    "seasonal_order": c.seasonal_order,
                    "metrics": metrics,
                    "model": model,
                }
        except Exception as e:
            results.append(
                {"order": c.order, "seasonal_order": c.seasonal_order, "error": str(e)}
            )

    # Ordena por RMSE (candidatos com erro vão para o fim)
    def _rank_key(r: Dict[str, Any]) -> float:
        return r["metrics"]["RMSE"] if "metrics" in r and "RMSE" in r["metrics"] else np.inf

    results = sorted(results, key=_rank_key)

    if best is None:
        return {"ok": False, "error": "nenhum candidato convergiu", "results": results}

    return {"ok": True, "best": best, "results": results, "y_len": n}
