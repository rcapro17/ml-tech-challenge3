"""
Data Warehouse operations for time series data
Handles database operations for series metadata and observations
"""
from typing import List, Dict, Any
from src.db import get_engine
from sqlalchemy import text


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
    Insert time series observations into database
    
    Args:
        code: Series identifier code
        observations: List of dicts with 'date' and 'value' keys
    
    Returns:
        Number of rows inserted
    """
    if not observations:
        return 0
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Create table if not exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS observations (
                    code VARCHAR(50),
                    date DATE,
                    value NUMERIC,
                    PRIMARY KEY (code, date)
                )
            """))
            
            # Insert observations
            inserted = 0
            for obs in observations:
                try:
                    conn.execute(text("""
                        INSERT INTO observations (code, date, value)
                        VALUES (:code, :date, :value)
                        ON CONFLICT (code, date) 
                        DO UPDATE SET value = EXCLUDED.value
                    """), {"code": code, "date": obs["date"], "value": obs["value"]})
                    inserted += 1
                except Exception as e:
                    print(f"Error inserting observation for {code} on {obs['date']}: {e}")
            
            conn.commit()
        
        return inserted
    
    except Exception as e:
        print(f"Error inserting observations for {code}: {e}")
        return 0
