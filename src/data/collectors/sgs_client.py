import httpx
from datetime import datetime

BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

def _fmt(d):  # recebe 'YYYY-MM-DD' e vira 'DD/MM/YYYY'
    return datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")

def fetch_sgs_series(code: str, start: str, end: str):
    url = BASE.format(code=code)
    params = {
        "formato": "json",
        "dataInicial": _fmt(start),
        "dataFinal": _fmt(end),
    }
    try:
        r = httpx.get(url, params=params, timeout=30)
        if r.status_code == 404:
            # Sem dados no período: não falha o pipeline
            return []
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        # Logue e propague como lista vazia para não quebrar o fluxo de coleta
        print(f"SGS fetch error for {code}: {e}")
        return []

    rows = []
    for item in js:
        # API do BCB retorna {"data": "DD/MM/YYYY", "valor": "123,45"}
        ds = datetime.strptime(item["data"], "%d/%m/%Y").strftime("%Y-%m-%d")
        val = float(str(item["valor"]).replace(",", "."))
        rows.append({"ts": ds, "value": val})
    return rows

