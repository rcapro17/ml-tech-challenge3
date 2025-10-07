#!/usr/bin/env bash
set -e
mkdir -p data/raw/source=SGS data/feature_store/source=SGS data/models/source=SGS
exec gunicorn -w 2 -k gthread --threads 4 --timeout 120 --bind 0.0.0.0:$PORT wsgi:app
