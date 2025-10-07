"""
SGS (Sistema Gerenciador de Séries Temporais) Client
Fetches time series data from Banco Central do Brasil
"""
import requests
from datetime import datetime
from typing import List, Dict, Any


def fetch_sgs_series(code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Fetch time series data from BCB SGS API
    
    Args:
        code: Series code (e.g., "1" for SELIC)
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
    
    Returns:
        List of dictionaries with 'date' and 'value' keys
    """
    # Convert dates to DD/MM/YYYY format expected by BCB API
    start_date = datetime.strptime(start, "%Y-%m-%d").strftime("%d/%m/%Y")
    end_date = datetime.strptime(end, "%Y-%m-%d").strftime("%d/%m/%Y")
    
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    params = {
        "formato": "json",
        "dataInicial": start_date,
        "dataFinal": end_date
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Transform to expected format
        result = []
        for item in data:
            result.append({
                "date": datetime.strptime(item["data"], "%d/%m/%Y").strftime("%Y-%m-%d"),
                "value": float(item["valor"])
            })
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SGS series {code}: {e}")
        return []
    except (KeyError, ValueError) as e:
        print(f"Error parsing SGS response for series {code}: {e}")
        return []
