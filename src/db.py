from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .config import settings

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    return _engine

def ping_db() -> bool:
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

# import os
# from sqlalchemy import create_engine

# url = os.getenv("DATABASE_URL")
# if not url:
#     # monta com variáveis separadas
#     host = os.getenv("DB_HOST", "127.0.0.1")
#     port = os.getenv("DB_PORT", "5432")
#     db   = os.getenv("DB_NAME", "mltech")
#     user = os.getenv("DB_USER", "mluser")
#     pwd  = os.getenv("DB_PASSWORD", "mlpass")
#     ssl  = os.getenv("DB_SSLMODE")  # ex: "require" ou None

#     url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
#     if ssl:
#         url += f"?sslmode={ssl}"

# # cria o engine
# eng = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)

