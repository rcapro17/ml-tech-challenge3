# src/db.py
import os
from sqlalchemy import create_engine, text

def _build_db_url():
    # 1) Prioriza DATABASE_URL (formato SQLAlchemy)
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # 2) Compatibilidade com DATABASE_URL sem driver (postgresql://...)
    #    Força driver psycopg e ssl se for Render
    ext = os.getenv("EXTERNAL_DATABASE_URL") or os.getenv("DATABASE_URL_RAW")
    if ext and ext.startswith("postgresql://"):
        return ext.replace("postgresql://", "postgresql+psycopg://") + "?sslmode=require"
    # 3) Fallback local (apenas para dev)
    return "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres"

ENGINE = create_engine(_build_db_url(), pool_pre_ping=True, pool_recycle=1800)

def ping_db():
    try:
        with ENGINE.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# # src/db.py

# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import Engine
# from .config import settings

# _engine: Engine | None = None

# def get_engine() -> Engine:
#     global _engine
#     if _engine is None:
#         # 1. Obtém a URL crua: postgresql://...
#         db_url_raw = settings.database_url
        
#         # 2. Converte para o dialeto do SQLAlchemy/Psycopg: postgresql+psycopg://...
#         db_url_sqla = db_url_raw.replace("postgresql://", "postgresql+psycopg://")
        
#         _engine = create_engine(db_url_sqla, pool_pre_ping=True, future=True)
        
#     return _engine

# def ping_db() -> bool:
#     try:
#         eng = get_engine()
#         with eng.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         return True
#     except Exception:
#         return False