"""
Data Warehouse operations for time series data
Handles database operations for series metadata and observations
"""
from typing import List, Dict, Any
from src.db import get_engine
from sqlalchemy import text
from ..db import get_engine 


def upsert_series(code: str, source: str, name: str, frequency: str) -> bool:
    """
    Insert or update series metadata
    
    Args:
        code: Series identifier code
        source: Data source (e.g., "SGS")
        name: Human-readable series name
        frequency: Data frequency (e.g., "daily", "monthly")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Create table if not exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS series (
                    code VARCHAR(50) PRIMARY KEY,
                    source VARCHAR(50),
                    name VARCHAR(255),
                    frequency VARCHAR(50),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Upsert series metadata
            conn.execute(text("""
                INSERT INTO series (code, source, name, frequency, updated_at)
                VALUES (:code, :source, :name, :frequency, CURRENT_TIMESTAMP)
                ON CONFLICT (code) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    frequency = EXCLUDED.frequency,
                    updated_at = CURRENT_TIMESTAMP
            """), {"code": code, "source": source, "name": name, "frequency": frequency})
            
            conn.commit()
        
        return True
    
    except Exception as e:
        print(f"Error upserting series {code}: {e}")
        return False


def insert_observations(code: str, observations: List[Dict[str, Any]]) -> int:
    """
    Insere observações no Postgres.
    Aceita linhas com 'ts' ou 'date' e grava na coluna 'ts' (DATE).
    PK: (code, ts)
    """
    if not observations:
        return 0

    # normaliza chaves -> ts/value
    norm = []
    for obs in observations:
        ts = obs.get("ts") or obs.get("date")
        val = obs.get("value")
        if not ts or val is None:
            continue
        norm.append({"code": code, "ts": ts, "value": val})

    if not norm:
        return 0

    engine = get_engine()
    try:
        with engine.begin() as conn:  # abre transação automaticamente (commit/rollback)
            # garante a tabela com a coluna 'ts'
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS observations (
                    code VARCHAR(50) NOT NULL,
                    ts   DATE        NOT NULL,
                    value NUMERIC    NOT NULL,
                    PRIMARY KEY (code, ts)
                )
            """))

            # upsert
            conn.execute(
                text("""
                    INSERT INTO observations (code, ts, value)
                    VALUES (:code, :ts, :value)
                    ON CONFLICT (code, ts)
                    DO UPDATE SET value = EXCLUDED.value
                """),
                norm
            )

        return len(norm)

    except Exception as e:
        print(f"Error inserting observations for {code}: {e}", flush=True)
        return 0