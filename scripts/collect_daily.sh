#!/usr/bin/env bash
set -euo pipefail
START=$(date -v-30d +%Y-%m-%d)
END=$(date +%Y-%m-%d)
CODES='["10844"]'
curl -s -X POST "http://localhost:8000/v1/collect/sgs" \
  -H "Content-Type: application/json" \
  -d "{
        \"codes\": $CODES,
        \"start\": \"$START\",
        \"end\":   \"$END\",
        \"write_lake\": true,
        \"metadata\": {\"10844\": {\"name\": \"USD/BRL PTAX venda\", \"frequency\": \"daily\"}}
      }" | tee -a scripts/collect_daily.log
echo
