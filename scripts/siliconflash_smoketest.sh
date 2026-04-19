#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
MODEL="${2:-omni9b-phase2prime}"

curl -s "$BASE_URL/v1/models" | uv run python -m json.tool >/dev/null
curl -s "$BASE_URL/health" | uv run python -m json.tool >/dev/null
curl -s "$BASE_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}], \"max_tokens\": 32, \"temperature\": 0}" \
  | uv run python -m json.tool
