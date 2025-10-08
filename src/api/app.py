# src/api/app.py
from __future__ import annotations

from flask import Flask, request, jsonify, render_template_string, render_template
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import os
import time

from src.db import ping_db
from src.data.collectors.sgs_client import fetch_sgs_series
from src.data.warehouse import upsert_series, insert_observations
from src.data.lake import write_sgs_parquet
from src.ml.train import train_and_backtest, forecast_next
from src.ml.registry import (
    load_latest_model,
    make_run_dir,
    save_artifacts,
    latest_run_dir as latest_model_run,
)
from src.ml.tuning import tune_sarimax_for_code
from src.ml.etl import load_series


app = Flask(__name__)

# --- Bootstrap 1x por worker (Flask 3 compatível) ---
_BOOTSTRAP_FLAG = "_BOOTSTRAP_DONE"
_bootstrapped = False

def _bootstrap_if_empty() -> None:
    """Se a série '1' não tiver histórico no warehouse, busca do SGS e grava."""
    if os.environ.get(_BOOTSTRAP_FLAG) == "1":
        return
    try:
        df = load_series("1")
        if df is None or df.empty:
            end = (date.today() - timedelta(days=1)).isoformat()
            rows = fetch_sgs_series("1", "2010-01-01", end)
            upsert_series("1", source="SGS", name="USD/BRL PTAX venda", frequency="daily")
            n = insert_observations("1", rows)
            # também grava no lake para futuras features
            try:
                write_sgs_parquet("1", rows, base_dir="data/raw")
            except Exception as e:
                print(f"[bootstrap] erro ao gravar parquet: {e}", flush=True)
            print(f"[bootstrap] Série 1 populada ({n} linhas) até {end}.", flush=True)
        else:
            print("[bootstrap] Série 1 já possui histórico.", flush=True)
        os.environ[_BOOTSTRAP_FLAG] = "1"
    except Exception as e:
        # não derruba o app
        print("[bootstrap] erro:", e, flush=True)

@app.before_request
def _ensure_bootstrap_once():
    global _bootstrapped
    if not _bootstrapped:
        _bootstrap_if_empty()
        _bootstrapped = True


# ---------- helpers ----------
def _respond_error(msg: str, status: int = 400):
    """Responde em HTML quando é HTMX/Accept:text/html; em JSON caso contrário."""
    accepts_html = "text/html" in (request.headers.get("Accept", "") or "")
    is_htmx = (request.headers.get("HX-Request") == "true")
    if accepts_html or is_htmx:
        html = f"""
        <div class="flash-error relative p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800">
          <button type="button" aria-label="Fechar"
                  class="absolute top-2 right-2 text-rose-700 hover:text-rose-900"
                  onclick="this.closest('.flash-error').remove()">
            &times;
          </button>
          <div class="font-semibold">Falha</div>
          <div class="mt-1 text-sm break-all">{msg}</div>
        </div>
        """
        return render_template_string(html), status
    return jsonify({"ok": False, "error": msg}), status


def ensure_features(code: str) -> int:
    """
    Gera features em data/feature_store/source=SGS/{code}.parquet
    a partir do lake (data/raw) ou do warehouse (load_series), se ainda não existir.
    Retorna número de linhas escritas (0 se nada foi feito, -1 se não há dados).
    """
    fs_path = Path(f"data/feature_store/source=SGS/{code}.parquet")
    if fs_path.exists():
        return 0

    lake_path = Path(f"data/raw/source=SGS/{code}.parquet")
    if lake_path.exists():
        df = pd.read_parquet(lake_path)
    else:
        df = load_series(code)

    if df is None or df.empty:
        return -1

    s = (df.set_index("ts")["value"]
           .asfreq("B", method="ffill")
           .dropna()
           .rename("value"))
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().reset_index().to_parquet(fs_path, index=False)
    return int(s.size)


# ---------- health ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ml-tech-challenge", "framework": "flask"})

@app.get("/health/db")
def health_db():
    ok = ping_db()
    code = 200 if ok else 500
    return jsonify({"db_ok": ok}), code


# ---------- coleta ----------
@app.post("/v1/collect/sgs")
def collect_sgs():
    try:
        payload = request.get_json(silent=True) or {}
        codes = payload.get("codes") or []
        start = payload.get("start")
        end = payload.get("end")
        metadata = payload.get("metadata") or {}
        write_lake = bool(payload.get("write_lake", False))

        if not codes or not start or not end:
            return _respond_error("Informe 'codes' (lista), 'start' e 'end' (YYYY-MM-DD).", 400)

        summary = []
        for c in codes:
            code = str(c)
            meta = metadata.get(code, {})
            upsert_series(
                code,
                source="SGS",
                name=meta.get("name", f"SGS {code}"),
                frequency=meta.get("frequency", "daily"),
            )

            rows = fetch_sgs_series(code, start, end)
            n_db = insert_observations(code, rows)

            n_lake = 0
            if write_lake:
                try:
                    n_lake = write_sgs_parquet(code, rows, base_dir="data/raw")
                except Exception as e:
                    print(f"[collect] erro ao gravar parquet {code}: {e}", flush=True)

            summary.append({"code": code, "db_upserts": n_db, "lake_rows": n_lake})

        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


# ---------- build features (opcional/manual) ----------
@app.post("/v1/build_features")
def build_features_endpoint():
    try:
        payload = request.get_json(silent=True) or {}
        codes = payload.get("codes") or []
        if not codes:
            return _respond_error("informe 'codes'", 400)

        out = []
        for code in map(str, codes):
            written = ensure_features(code)
            if written == -1:
                out.append({"code": code, "written": 0, "reason": "sem dados no lake/DB"})
            else:
                out.append({"code": code, "written": max(0, written)})

        return jsonify({"ok": True, "summary": out})
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


# ---------- previsão "ad hoc" ----------
@app.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        code = str(payload.get("code", "")).strip()
        h = int(payload.get("h", 5))
        order = tuple(payload.get("order", [1, 1, 1]))
        seasonal_order = tuple(payload.get("seasonal_order", [0, 0, 0, 0]))

        if not code or h <= 0:
            return _respond_error("informe 'code' e 'h' > 0", 400)

        out = train_and_backtest(code, order=order, seasonal_order=seasonal_order, backtest_h=min(h, 5))
        if not out.get("ok"):
            return _respond_error(out.get("error", "falha no treino"), 400)

        model = out["model"]
        fc_df = forecast_next(model, steps=h)
        return jsonify({
            "ok": True,
            "code": code,
            "metrics": out["metrics"],
            "n_obs": out["n_obs"],
            "forecast": [
                {"ts": ts.strftime("%Y-%m-%d"), "yhat": float(v)}
                for ts, v in zip(fc_df["ts"], fc_df["forecast"])
            ]
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


# ---------- previsão do último modelo salvo ----------
@app.post("/v1/predict_latest")
def predict_latest():
    try:
        payload = request.get_json(silent=True) or {}
        code = str(payload.get("code", "")).strip()
        h = int(payload.get("h", 5))
        freq = str(payload.get("freq", "B"))
        if not code or h <= 0:
            return _respond_error("informe 'code' e 'h' > 0", 400)

        model, metrics = load_latest_model(code, freq)
        if model is None:
            return _respond_error("nenhum modelo salvo para este code/freq", 404)

        last_ts = pd.to_datetime(model.data.row_labels[-1])
        idx = pd.bdate_range(start=last_ts, periods=h+1, inclusive="neither")
        yhat = model.forecast(steps=h)

        forecast = [{"ts": ts.strftime("%Y-%m-%d"), "yhat": float(val)} for ts, val in zip(idx, yhat)]
        return jsonify({"ok": True, "code": code, "metrics": metrics, "forecast": forecast})
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


# ---------- tuning + treino + salvar artefatos ----------
@app.post("/v1/tune_train")
def tune_train():
    try:
        t0 = time.time()
        payload = request.get_json(silent=True) or request.form.to_dict() or request.args.to_dict()
        code = str(payload.get("code", "")).strip()
        freq = str(payload.get("freq", "B")).strip()
        h = int(payload.get("horizon", 5))
        init_ratio = float(payload.get("init_ratio", 0.7))

        if not code:
            return _respond_error("informe 'code'", 400)

        # Garante features
        written = ensure_features(code)
        fs_path = Path(f"data/feature_store/source=SGS/{code}.parquet")
        if written == -1 and not fs_path.exists():
            return _respond_error(f"sem dados para série {code}", 400)

        # (Opcional) modo rápido em produção para não estourar timeout do Render
        # Se quiser controlar pela UI, envie {"fast": true} no payload.
        fast = payload.get("fast")
        fast = True if (fast is True or str(fast).lower() in {"1", "true", "yes"}) else False
        if os.environ.get("RENDER_QUICK_TUNING") == "1":
            fast = True

        # A função existente não aceita grade? tudo bem: o "fast" pode ser
        # interpretado dentro de tune_sarimax_for_code (deixe ela ignorar se não suportar).
        out = tune_sarimax_for_code(
            code=code,
            freq=freq,
            horizon=h,
            initial_train_ratio=init_ratio,
            fast=fast  # parâmetro opcional/backward-compatible
        )
        if not out.get("ok"):
            return _respond_error(out.get("error", "falha no tuning"), 400)

        best = out["best"]
        run_dir = make_run_dir(code, freq)
        paths = save_artifacts(
            out_dir=run_dir,
            model_obj=best["model"],
            metrics=best["metrics"],
            meta={
                "code": code, "freq": freq,
                "order": list(best["order"]),
                "seasonal_order": list(best["seasonal_order"]),
                "y_len": out["y_len"],
                "horizon": h,
                "initial_train_ratio": init_ratio,
                "ranked_results": out["results"],
                "duration_sec": round(time.time() - t0, 3),
                "fast": fast,
            }
        )

        accepts_html = "text/html" in (request.headers.get("Accept", "") or "")
        if request.headers.get("HX-Request") == "true" or accepts_html:
            return render_template_string(
                """
                <div class="flash-success relative p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800">
                  <button type="button" aria-label="Fechar"
                          class="absolute top-2 right-2 text-emerald-700 hover:text-emerald-900"
                          onclick="this.closest('.flash-success').remove()">
                    &times;
                  </button>

                  <div class="font-semibold">✓ Tuning concluído</div>
                  <div class="mt-1 text-sm">
                    <span class="font-medium">RMSE:</span> {{ rmse|round(4) }} |
                    <span class="font-medium">order:</span> {{ order }} |
                    <span class="font-medium">seasonal:</span> {{ seasonal }}
                  </div>
                  <div class="mt-1 text-xs text-gray-600 break-all">run: {{ run }}</div>
                </div>
                """,
                rmse=best["metrics"]["RMSE"],
                order=best["order"],
                seasonal=best["seasonal_order"],
                run=run_dir
            )

        return jsonify({
            "ok": True,
            "best": {
                "order": best["order"],
                "seasonal_order": best["seasonal_order"],
                "metrics": best["metrics"]
            },
            "artifacts": paths
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


# ---------- dashboard ----------
@app.get("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        title="Dashboard",
        subtitle="SGS → Lake → Feature Store → Modelo",
        default_code="1"
    )


# ---------- dados para o gráfico ----------
@app.get("/v1/history")
def history():
    try:
        code = str(request.args.get("code", "")).strip()
        freq = str(request.args.get("freq", "B"))
        if not code:
            return _respond_error("informe 'code'", 400)

        # 1) tenta warehouse
        df = load_series(code)
        if df is not None and not df.empty:
            s = df.set_index("ts")["value"].asfreq(freq, method="ffill").dropna()
            out = [{"ts": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]
            return jsonify(out)

        # 2) fallback: serie do último modelo salvo
        model, _ = load_latest_model(code, freq)
        if model is not None:
            y = pd.Series(model.data.endog,
                          index=pd.to_datetime(model.data.row_labels),
                          name="value")
            s = y.asfreq(freq, method="ffill").dropna()
            out = [{"ts": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]
            return jsonify(out)

        return jsonify([])

    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


@app.get("/v1/forecast_latest")
def forecast_latest():
    try:
        code = str(request.args.get("code", "")).strip()
        h = int(request.args.get("h", 15))
        freq = str(request.args.get("freq", "B"))
        if not code or h <= 0:
            return _respond_error("informe 'code' e 'h' > 0", 400)

        model, _ = load_latest_model(code, freq)
        if model is None:
            return jsonify([])

        last_ts = pd.to_datetime(model.data.row_labels[-1])
        idx = pd.bdate_range(start=last_ts, periods=h+1, inclusive="neither")
        yhat = model.forecast(steps=h)
        out = [{"ts": ts.strftime("%Y-%m-%d"), "yhat": float(val)} for ts, val in zip(idx, yhat)]
        return jsonify(out)
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


@app.get("/v1/last_model_info")
def last_model_info():
    try:
        code = str(request.args.get("code", "")).strip()
        freq = str(request.args.get("freq", "B"))
        if not code:
            return _respond_error("informe 'code'", 400)

        model, metrics = load_latest_model(code, freq)
        run = latest_model_run(code, freq)
        if model is None:
            return jsonify({"metrics": {}, "order": None, "seasonal_order": None, "run": run})

        order = getattr(model.model, "order", None)
        seasonal = getattr(model.model, "seasonal_order", None)
        return jsonify({"metrics": metrics or {}, "order": order, "seasonal_order": seasonal, "run": run})
    except Exception as e:
        import traceback; traceback.print_exc()
        return _respond_error(str(e), 500)


if __name__ == "__main__":
    # dev local
    app.run(host="0.0.0.0", port=8000, debug=False)
