#!/bin/bash

# Este script roda o pipeline completo de seeding
set -euo pipefail

# --- 1. CRIAÇÃO DE TABELAS ---
echo "Executando inicialização do DB..."
python scripts/init_db.py

# --- 2. COLETA INICIAL (NOVO: Sem depender da API HTTP!) ---
echo "Executando Ingestão de Dados..."
python scripts/initial_ingest.py

# --- 3. TREINAMENTO INICIAL DO MODELO ---
echo "Executando treinamento inicial do modelo..."
python scripts/train_save.py --code 1 --freq B

echo "SETUP INICIAL CONCLUÍDO. PRONTO PARA INICIAR O SERVIDOR."