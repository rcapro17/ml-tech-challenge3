import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Garanta que a raiz e a pasta "src" estão no sys.path
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Exponha o app WSGI para o Gunicorn
from src.api.app import app
