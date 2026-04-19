#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PATH="${OMLX_BASE_PATH:-$ROOT_DIR/.omlx-qwen35-fast}"
MODEL_DIR="${OMLX_MODEL_DIR:-$ROOT_DIR/mlx_models}"
MODEL_ID="${OMLX_MODEL_ID:-Qwen3.5-9B-bf16}"
DRAFT_MODEL="${OMLX_DFLASH_DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}"
PORT="${OMLX_PORT:-8000}"
HOST="${OMLX_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
MAX_MODEL_MEMORY="${OMLX_MAX_MODEL_MEMORY:-48GB}"
MAX_PROCESS_MEMORY="${OMLX_MAX_PROCESS_MEMORY:-96GB}"
CACHE_DIR="${OMLX_PAGED_SSD_CACHE_DIR:-$BASE_PATH/cache-dflash-fast-v3}"
CACHE_MAX_SIZE="${OMLX_PAGED_SSD_CACHE_MAX_SIZE:-100GB}"
HOT_CACHE_MAX_SIZE="${OMLX_HOT_CACHE_MAX_SIZE:-0}"
INITIAL_CACHE_BLOCKS="${OMLX_INITIAL_CACHE_BLOCKS:-16}"
MAX_CONCURRENT_REQUESTS="${OMLX_MAX_CONCURRENT_REQUESTS:-1}"
START_TIMEOUT="${OMLX_START_TIMEOUT:-600}"
AUTO_WARMUP="${OMLX_AUTO_WARMUP:-1}"
ENABLE_THINKING="${OMLX_ENABLE_THINKING:-0}"
THINKING_BUDGET="${OMLX_THINKING_BUDGET:-}"
API_KEY="${OMLX_API_KEY:-}"

if [[ -n "${DFLASH_BLOCK_TOKENS:-}" ]]; then
  export DFLASH_BLOCK_TOKENS
elif [[ "$ENABLE_THINKING" == "1" ]]; then
  export DFLASH_BLOCK_TOKENS="11"
else
  export DFLASH_BLOCK_TOKENS="12"
fi

if [[ -n "${OMLX_WARMUP_MAX_TOKENS:-}" ]]; then
  WARMUP_MAX_TOKENS="$OMLX_WARMUP_MAX_TOKENS"
elif [[ "$ENABLE_THINKING" == "1" ]]; then
  WARMUP_MAX_TOKENS="16"
else
  WARMUP_MAX_TOKENS="$DFLASH_BLOCK_TOKENS"
fi
if [[ -n "${OMLX_WARMUP_IGNORE_EOS:-}" ]]; then
  WARMUP_IGNORE_EOS="$OMLX_WARMUP_IGNORE_EOS"
elif [[ "$ENABLE_THINKING" == "1" ]]; then
  WARMUP_IGNORE_EOS="0"
else
  WARMUP_IGNORE_EOS="1"
fi

export DFLASH_DDTREE_RUNTIME="${DFLASH_DDTREE_RUNTIME:-0}"
export DFLASH_BSTNXBT_RUNTIME="${DFLASH_BSTNXBT_RUNTIME:-1}"
export DFLASH_BSTNXBT_SPLIT_SDPA="${DFLASH_BSTNXBT_SPLIT_SDPA:-1}"
export DFLASH_BSTNXBT_RECURRENT_KERNELS="${DFLASH_BSTNXBT_RECURRENT_KERNELS:-1}"
export DFLASH_BSTNXBT_EXTERNAL_DRAFT="${DFLASH_BSTNXBT_EXTERNAL_DRAFT:-1}"

mkdir -p "$BASE_PATH" "$CACHE_DIR"

BASE_PATH="$BASE_PATH" MODEL_ID="$MODEL_ID" DRAFT_MODEL="$DRAFT_MODEL" ENABLE_THINKING="$ENABLE_THINKING" THINKING_BUDGET="$THINKING_BUDGET" uv run python - <<'PY'
import os
from pathlib import Path
from omlx.model_settings import ModelSettingsManager, apply_dflash_fast_model_settings

base = Path(os.environ["BASE_PATH"])
settings = ModelSettingsManager(base)
current = settings.get_settings(os.environ["MODEL_ID"])
thinking_budget = os.environ.get("THINKING_BUDGET", "").strip()
current = apply_dflash_fast_model_settings(
    current,
    draft_model=os.environ["DRAFT_MODEL"],
    enable_thinking=os.environ.get("ENABLE_THINKING", "0") == "1",
    thinking_budget=int(thinking_budget) if thinking_budget else None,
)
settings.set_settings(os.environ["MODEL_ID"], current)
PY

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

uv run python -m omlx.cli serve \
  --base-path "$BASE_PATH" \
  --model-dir "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-memory "$MAX_MODEL_MEMORY" \
  --max-process-memory "$MAX_PROCESS_MEMORY" \
  --paged-ssd-cache-dir "$CACHE_DIR" \
  --paged-ssd-cache-max-size "$CACHE_MAX_SIZE" \
  --hot-cache-max-size "$HOT_CACHE_MAX_SIZE" \
  --initial-cache-blocks "$INITIAL_CACHE_BLOCKS" \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" &
SERVER_PID=$!

for _ in $(seq 1 "$START_TIMEOUT"); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
    exit 1
  fi
done

if ! curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "Timed out waiting for server startup at $BASE_URL" >&2
  exit 1
fi

echo "oMLX Qwen3.5 fast server is up at $BASE_URL"
echo "Model: $MODEL_ID"
echo "Draft: $DRAFT_MODEL"
echo "Env: DFLASH_DDTREE_RUNTIME=$DFLASH_DDTREE_RUNTIME DFLASH_BSTNXBT_RUNTIME=$DFLASH_BSTNXBT_RUNTIME DFLASH_BSTNXBT_SPLIT_SDPA=$DFLASH_BSTNXBT_SPLIT_SDPA DFLASH_BSTNXBT_RECURRENT_KERNELS=$DFLASH_BSTNXBT_RECURRENT_KERNELS DFLASH_BSTNXBT_EXTERNAL_DRAFT=$DFLASH_BSTNXBT_EXTERNAL_DRAFT DFLASH_BLOCK_TOKENS=$DFLASH_BLOCK_TOKENS ENABLE_THINKING=$ENABLE_THINKING THINKING_BUDGET=${THINKING_BUDGET:-unset}"
echo "DFlash is enabled by model settings, so requests do not need to pass dflash=true unless you want to override per request."

if [[ "$AUTO_WARMUP" == "1" ]]; then
  echo "Warming the DFlash path for steady-state serving..."
  BASE_URL="$BASE_URL" MODEL_ID="$MODEL_ID" WARMUP_MAX_TOKENS="$WARMUP_MAX_TOKENS" WARMUP_IGNORE_EOS="$WARMUP_IGNORE_EOS" ENABLE_THINKING="$ENABLE_THINKING" THINKING_BUDGET="$THINKING_BUDGET" API_KEY="$API_KEY" uv run python - <<'PY'
import os
import requests

payload = {
    "model": os.environ["MODEL_ID"],
    "messages": [{
        "role": "user",
        "content": "warmup: exercise the configured DFlash generation path with a distinct prompt and reply briefly."
    }],
    "max_tokens": int(os.environ["WARMUP_MAX_TOKENS"]),
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "min_p": 0.0,
    "stream": False,
    "dflash": True,
    "chat_template_kwargs": {"enable_thinking": os.environ.get("ENABLE_THINKING", "0") == "1"},
}
if os.environ.get("WARMUP_IGNORE_EOS", "1") == "1":
    payload["ignore_eos"] = True
thinking_budget = os.environ.get("THINKING_BUDGET", "")
if thinking_budget:
    payload["thinking_budget"] = int(thinking_budget)
headers = {}
api_key = os.environ.get("API_KEY", "").strip()
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
requests.post(f"{os.environ['BASE_URL']}/v1/chat/completions", json=payload, headers=headers, timeout=1800).raise_for_status()
PY
  echo "Warmup complete."
fi

echo
echo "Example request:"
if [[ -n "$API_KEY" ]]; then
  echo "curl -s ${BASE_URL}/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -H 'Authorization: Bearer ${API_KEY}' \\
  -d '{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain speculative decoding on Apple Silicon.\"}],\"max_tokens\":256,\"temperature\":0,\"top_p\":1,\"top_k\":0,\"min_p\":0.0,\"stream\":false}$( [[ -n "$THINKING_BUDGET" ]] && printf ',\"thinking_budget\":%s' "$THINKING_BUDGET" )'"
else
  echo "curl -s ${BASE_URL}/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -d '{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain speculative decoding on Apple Silicon.\"}],\"max_tokens\":256,\"temperature\":0,\"top_p\":1,\"top_k\":0,\"min_p\":0.0,\"stream\":false}$( [[ -n "$THINKING_BUDGET" ]] && printf ',\"thinking_budget\":%s' "$THINKING_BUDGET" )'"
fi

wait "$SERVER_PID"
