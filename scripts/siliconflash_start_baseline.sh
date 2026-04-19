#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PATH="${OMLX_BASE_PATH:-$ROOT_DIR/.omlx-local}"
MODEL_DIR="${OMLX_MODEL_DIR:-$ROOT_DIR/models}"
PORT="${OMLX_PORT:-8000}"
HOST="${OMLX_HOST:-127.0.0.1}"
MAX_MODEL_MEMORY="${OMLX_MAX_MODEL_MEMORY:-48GB}"
MAX_PROCESS_MEMORY="${OMLX_MAX_PROCESS_MEMORY:-96GB}"
CACHE_DIR="${OMLX_PAGED_SSD_CACHE_DIR:-$ROOT_DIR/.omlx-local/cache}"
CACHE_MAX_SIZE="${OMLX_PAGED_SSD_CACHE_MAX_SIZE:-100GB}"
HOT_CACHE_MAX_SIZE="${OMLX_HOT_CACHE_MAX_SIZE:-0}"

mkdir -p "$BASE_PATH" "$CACHE_DIR"

exec uv run omlx serve \
  --base-path "$BASE_PATH" \
  --model-dir "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-memory "$MAX_MODEL_MEMORY" \
  --max-process-memory "$MAX_PROCESS_MEMORY" \
  --paged-ssd-cache-dir "$CACHE_DIR" \
  --paged-ssd-cache-max-size "$CACHE_MAX_SIZE" \
  --hot-cache-max-size "$HOT_CACHE_MAX_SIZE"
