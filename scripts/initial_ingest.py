# scripts/initial_ingest.py (Crie este arquivo)

import sys
import os
# Ajusta o path para importar módulos do src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importe a função principal de coleta, se ela estiver no app.py ou em algum módulo de coleta
# Vou assumir que você tem uma função que pode ser chamada diretamente:
from src.api.app import collect_sgs # Ajuste se a função de coleta estiver em outro lugar

def run_initial_ingestion():
    """Roda a coleta de dados diretamente, sem a camada HTTP."""
    print("Iniciando ingestão de dados para seeding inicial...")
    
    # Simula os parâmetros do seu POST:
    codes = ["1"]
    start_date = "2023-01-01"
    end_date = "2025-10-01"
    metadata = {"1": {"name": "USD/BRL venda (SGS 1)", "frequency": "daily"}}
    
    # Chama a função principal de coleta
    try:
        # A função collect_sgs precisa aceitar esses parâmetros
        summary = collect_sgs(
            codes=codes,
            start=start_date,
            end=end_date,
            write_lake=True,
            metadata=metadata
        )
        print("Ingestão inicial concluída com sucesso.")
        print(f"Resumo: {summary}")
        return True
    except Exception as e:
        print(f"ERRO FATAL DURANTE A INGESTÃO INICIAL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_initial_ingestion()