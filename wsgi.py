# wsgi.py (na raiz do repo)
import os, sys
BASE_DIR = os.path.dirname(__file__)
# garante que a raiz do projeto esteja no sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.api.app import app  # expõe "app" para o Gunicorn
