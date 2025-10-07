# IMPORTS ABSOLUTOS (melhor para a aplicação rodando como WSGI)
from flask import Flask, request, jsonify, render_template_string, render_template
import pandas as pd
from datetime import date, timedelta 
from pathlib import Path
import os

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
 
# --- BOOTSTRAP "RUN-ONCE" COMPATÍVEL ---
_BOOTSTRAP_FLAG = "_BOOTSTRAP_DONE"
_bootstrapped = False  # flag em memória

def _bootstrap_if_empty():
    """Sem redeclarar decorators raros. Roda rápido e só 1x."""
    if os.environ.get(_BOOTSTRAP_FLAG) == "1":
        return
    try:
        df = load_series("1")
        if df is None or df.empty:
            end = (date.today() - timedelta(days=1)).isoformat()
            rows = fetch_sgs_series("1", "2010-01-01", end)
            upsert_series("1", source="SGS", name="USD/BRL PTAX venda", frequency="daily")
            insert_observations("1", rows)
        os.environ[_BOOTSTRAP_FLAG] = "1"
        print("Bootstrap concluído (ou já existia).", flush=True)
    except Exception as e:
        print("bootstrap error:", e, flush=True)

@app.before_request
def _ensure_bootstrap_once():
    global _bootstrapped
    if not _bootstrapped:
        _bootstrap_if_empty()
        _bootstrapped = True

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ml-tech-challenge", "framework": "flask"})

@app.get("/health/db")
def health_db():
    ok = ping_db()
    code = 200 if ok else 500
    return jsonify({"db_ok": ok}), code


# ------------------------------
# NOVA ROTA: /v1/collect/sgs
# ------------------------------
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
            return jsonify({"error": "Informe 'codes' (lista), 'start' e 'end' (YYYY-MM-DD)."}), 400

        summary = []
        for c in codes:
            code = str(c)
            meta = metadata.get(code, {})
            upsert_series(
                code, source="SGS",
                name=meta.get("name", f"SGS {code}"),
                frequency=meta.get("frequency", "daily"),
            )

            rows = fetch_sgs_series(code, start, end)
            n_db = insert_observations(code, rows)

            n_lake = 0
            if write_lake:
                n_lake = write_sgs_parquet(code, rows, base_dir="data/raw")

            summary.append({"code": code, "db_upserts": n_db, "lake_rows": n_lake})

        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500
    

@app.post("/v1/dev/seed")
def dev_seed():
    try:
        payload = request.get_json(silent=True) or {}
        codes = payload.get("codes") or ["1"]
        start = payload.get("start") or "2010-01-01"
        from datetime import date, timedelta
        end = payload.get("end") or (date.today() - timedelta(days=1)).isoformat()

        summary = []
        for code in codes:
            upsert_series(str(code), source="SGS", name="USD/BRL PTAX venda", frequency="daily")
            rows = fetch_sgs_series(str(code), start, end)
            n = insert_observations(str(code), rows)
            summary.append({"code": str(code), "db_upserts": n})
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

    

# exemplo bem direto; adapte para sua função real de build
@app.post("/v1/build_features")
def build_features_endpoint():
    try:
        payload = request.get_json(silent=True) or {}
        codes = payload.get("codes") or []
        if not codes:
            return jsonify({"ok": False, "error": "informe 'codes'"}), 400

        out = []
        for code in codes:
            code = str(code)
            # 1) leia do lake (parquet) OU do DB
            # Exemplo via lake parquet (ajuste caminho conforme seu writer):
            lake_path = Path(f"data/raw/source=SGS/{code}.parquet")
            if not lake_path.exists():
                # se não existir parquet, tente buscar do DB:
                df = load_series(code)  # se esta função lê do DB, OK
            else:
                df = pd.read_parquet(lake_path)

            if df is None or df.empty:
                out.append({"code": code, "written": 0, "reason": "sem dados no lake/DB"})
                continue

            # 2) faça o processamento de features (resample, ffill, etc.)
            s = (df.set_index("ts")["value"]
                   .asfreq("B", method="ffill")
                   .dropna()
                   .rename("value"))
            fs_dir = Path("data/feature_store/source=SGS")
            fs_dir.mkdir(parents=True, exist_ok=True)
            s.to_frame().reset_index().to_parquet(fs_dir / f"{code}.parquet", index=False)
            out.append({"code": code, "written": int(s.size)})

        return jsonify({"ok": True, "summary": out})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500
    


@app.post("/v1/predict")
def predict():
    """
    Body:
    {
      "code": "10844",
      "h": 5,                    # horizonte (dias úteis)
      "order": [1,1,1],          # opcional
      "seasonal_order": [0,0,0,0]# opcional
    }
    """
    try:
        payload = request.get_json(silent=True) or {}
        code = str(payload.get("code", "")).strip()
        h = int(payload.get("h", 5))
        order = tuple(payload.get("order", [1,1,1]))
        seasonal_order = tuple(payload.get("seasonal_order", [0,0,0,0]))

        if not code or h <= 0:
            return jsonify({"error": "informe 'code' e 'h' > 0"}), 400

        out = train_and_backtest(code, order=order, seasonal_order=seasonal_order, backtest_h=min(h, 5))
        if not out.get("ok"):
            return jsonify(out), 400

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
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.post("/v1/predict_latest")
def predict_latest():
    try:
        payload = request.get_json(silent=True) or {}
        code = str(payload.get("code", "")).strip()
        h = int(payload.get("h", 5))
        freq = str(payload.get("freq", "B"))
        if not code or h <= 0:
            return jsonify({"error": "informe 'code' e 'h' > 0"}), 400

        model, metrics = load_latest_model(code, freq)
        if model is None:
            return jsonify({"ok": False, "error": "nenhum modelo salvo para este code/freq"}), 404

        last_ts = pd.to_datetime(model.data.row_labels[-1])
        idx = pd.bdate_range(start=last_ts, periods=h+1, inclusive="neither")
        yhat = model.forecast(steps=h)

        forecast = [{"ts": ts.strftime("%Y-%m-%d"), "yhat": float(val)} for ts, val in zip(idx, yhat)]
        return jsonify({"ok": True, "code": code, "metrics": metrics, "forecast": forecast})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/v1/tune_train")
def tune_train():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict() or request.args.to_dict()
        code = str(payload.get("code", "")).strip()
        freq = str(payload.get("freq", "B")).strip()
        h = int(payload.get("horizon", 5))
        init_ratio = float(payload.get("init_ratio", 0.7))

        out = tune_sarimax_for_code(code=code, freq=freq, horizon=h, initial_train_ratio=init_ratio)
        if not out.get("ok"):
            return jsonify(out), 400

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
            }
        )

        # >>> Se o cliente pede HTML (HTMX ou Accept: text/html), retorna um bloco pronto pra injetar no DOM
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


        # >>> Caso contrário, API JSON “crua”
        return jsonify({
            "ok": True,
            "best": {"order": best["order"], "seasonal_order": best["seasonal_order"], "metrics": best["metrics"]},
            "artifacts": paths
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500




@app.get("/dashboard")
def dashboard():
    # página HTML
    return render_template("dashboard.html", title="Dashboard", subtitle="SGS → Lake → Feature Store → Modelo",
                           default_code="1")

@app.get("/v1/history")
def history():
    try:
        code = str(request.args.get("code", "")).strip()
        freq = str(request.args.get("freq", "B"))
        if not code:
            return jsonify({"error": "informe 'code'"}), 400

        df = load_series(code)
        if df is not None and not df.empty:
            s = df.set_index("ts")["value"].asfreq(freq, method="ffill").dropna()
            out = [{"ts": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]
            return jsonify(out)

        # FALLBACK: se não tem dados no warehouse, use a série do último modelo salvo
        model, _ = load_latest_model(code, freq)
        if model is not None:
            import pandas as pd
            y = pd.Series(model.data.endog, index=pd.to_datetime(model.data.row_labels), name="value")
            s = y.asfreq(freq, method="ffill").dropna()
            out = [{"ts": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]
            return jsonify(out)

        # se não tiver nada mesmo:
        return jsonify([])

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/v1/forecast_latest")
def forecast_latest():
    """
    Retorna forecast do último modelo salvo: /v1/forecast_latest?code=1&h=15&freq=B
    """
    try:
        code = str(request.args.get("code", "")).strip()
        h = int(request.args.get("h", 15))
        freq = str(request.args.get("freq", "B"))
        if not code or h <= 0:
            return jsonify({"error": "informe 'code' e 'h' > 0"}), 400

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
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/v1/last_model_info")
def last_model_info():
    """
    Info do último modelo salvo (metrics, order, seasonal, run dir): /v1/last_model_info?code=1&freq=B
    """
    try:
        code = str(request.args.get("code", "")).strip()
        freq = str(request.args.get("freq", "B"))
        if not code:
            return jsonify({"error": "informe 'code'"}), 400

        model, metrics = load_latest_model(code, freq)
        run = latest_model_run(code, freq)
        if model is None:
            return jsonify({"metrics": {}, "order": None, "seasonal_order": None, "run": run})

        order = getattr(model.model, "order", None)
        seasonal = getattr(model.model, "seasonal_order", None)
        return jsonify({"metrics": metrics or {}, "order": order, "seasonal_order": seasonal, "run": run})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.before_first_request
def bootstrap_if_empty():
    """
    Preenche a série '1' se o histórico estiver vazio (útil no primeiro deploy).
    É idempotente: se já existir dado, não faz nada.
    """
    try:
        df = load_series("1")
        if df is not None and not df.empty:
            print("[bootstrap] Série 1 já possui histórico, nada a fazer.", flush=True)
            return

        end = (date.today() - timedelta(days=1)).isoformat()  # evita pedir futuro ao BCB
        print(f"[bootstrap] Sem histórico. Coletando SGS 1 até {end}...", flush=True)

        # garante que a série existe na tabela 'series'
        upsert_series("1", source="SGS", name="USD/BRL PTAX venda", frequency="daily")

        # baixa e grava observações
        rows = fetch_sgs_series("1", "2010-01-01", end)
        n = insert_observations("1", rows)
        print(f"[bootstrap] Inseridas/atualizadas {n} observações da série 1.", flush=True)

    except Exception as e:
        print("[bootstrap] erro:", e, flush=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
