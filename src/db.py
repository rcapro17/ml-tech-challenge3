# src/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.config import settings   # ✅ absoluto, não ".config"

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        db_url_raw = settings.database_url
        db_url_sqla = db_url_raw.replace("postgresql://", "postgresql+psycopg://")
        _engine = create_engine(db_url_sqla, pool_pre_ping=True, future=True)
    return _engine

def ping_db() -> bool:
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
