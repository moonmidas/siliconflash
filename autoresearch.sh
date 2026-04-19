#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${AR_PORT:-8011}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
BASE_PATH="${AR_BASE_PATH:-$ROOT_DIR/.omlx-autoresearch}"
MODEL_DIR="${AR_MODEL_DIR:-$ROOT_DIR/models}"
MODEL_ID="${AR_MODEL_ID:-omni9b-phase2prime}"
CACHE_DIR="${AR_CACHE_DIR:-$BASE_PATH/cache}"
SERVER_LOG="${AR_SERVER_LOG:-$BASE_PATH/server.log}"
BENCH_JSON="${AR_BENCH_JSON:-$BASE_PATH/benchmark.json}"
STARTUP_FILE="${AR_STARTUP_FILE:-$BASE_PATH/startup_s.txt}"
WARMUP_FILE="${AR_WARMUP_FILE:-$BASE_PATH/warmup_s.txt}"
MAX_MODEL_MEMORY="${AR_MAX_MODEL_MEMORY:-48GB}"
MAX_PROCESS_MEMORY="${AR_MAX_PROCESS_MEMORY:-96GB}"
CACHE_MAX_SIZE="${AR_CACHE_MAX_SIZE:-100GB}"
HOT_CACHE_MAX_SIZE="${AR_HOT_CACHE_MAX_SIZE:-0}"
INITIAL_CACHE_BLOCKS="${AR_INITIAL_CACHE_BLOCKS:-16}"
MAX_CONCURRENT_REQUESTS="${AR_MAX_CONCURRENT_REQUESTS:-1}"
MAX_TOKENS="${AR_MAX_TOKENS:-256}"
START_TIMEOUT="${AR_START_TIMEOUT:-600}"
ENABLE_DFLASH="${AR_ENABLE_DFLASH:-0}"
BENCHMARK_DFLASH="${AR_BENCHMARK_DFLASH:-0}"
DFLASH_DRAFT_MODEL="${AR_DFLASH_DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}"
DFLASH_USE_MLX_NATIVE_DRAFTER="${AR_DFLASH_USE_MLX_NATIVE_DRAFTER:-1}"
IGNORE_EOS="${AR_IGNORE_EOS:-0}"
ENABLE_THINKING="${AR_ENABLE_THINKING:-0}"
THINKING_BUDGET="${AR_THINKING_BUDGET:-}"
SETTINGS_ENABLE_THINKING="${AR_SETTINGS_ENABLE_THINKING:-$ENABLE_THINKING}"
SETTINGS_THINKING_BUDGET="${AR_SETTINGS_THINKING_BUDGET:-$THINKING_BUDGET}"
BENCH_PROMPT="${AR_PROMPT:-You are helping with a performance engineering task. Explain, in detail, how speculative decoding can accelerate autoregressive generation on Apple Silicon.}"
WARMUP_PROMPT="${AR_WARMUP_PROMPT:-warmup: exercise the configured generation path with a distinct prompt and reply briefly.}"
WARMUP_MAX_TOKENS="${AR_WARMUP_MAX_TOKENS:-1}"
WARMUP_DFLASH="${AR_WARMUP_DFLASH:-$BENCHMARK_DFLASH}"
WARMUP_ENABLE_THINKING="${AR_WARMUP_ENABLE_THINKING:-$ENABLE_THINKING}"
WARMUP_REPEAT="${AR_WARMUP_REPEAT:-1}"
QUALITY_EVAL="${AR_QUALITY_EVAL:-1}"
QUALITY_EVAL_JSON="${AR_QUALITY_EVAL_JSON:-$BASE_PATH/quality_eval.json}"
WORKLOAD_ENV_FILE="${AR_WORKLOAD_ENV_FILE:-$BASE_PATH/workload.env}"

mkdir -p "$BASE_PATH"
if [[ -f "$WORKLOAD_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$WORKLOAD_ENV_FILE"
fi
BENCH_PROMPT="${BENCH_PROMPT:-${AR_PROMPT:-$BENCH_PROMPT}}"
DEFAULT_WARMUP_PROMPT="warmup: exercise the configured generation path with a distinct prompt and reply briefly."
WARMUP_PROMPT="${WARMUP_PROMPT:-${AR_WARMUP_PROMPT:-$WARMUP_PROMPT}}"
if [[ "$ENABLE_THINKING" == "1" && "$WARMUP_PROMPT" == "$DEFAULT_WARMUP_PROMPT" ]]; then
  WARMUP_PROMPT="Solve briefly and show your reasoning: What is 2 + 2?"
fi
if [[ "$WARMUP_DFLASH" == "1" && "$WARMUP_MAX_TOKENS" == "1" ]]; then
  if [[ -n "${DFLASH_BLOCK_TOKENS:-}" ]]; then
    WARMUP_MAX_TOKENS="$DFLASH_BLOCK_TOKENS"
  elif [[ "$ENABLE_THINKING" == "1" ]]; then
    WARMUP_MAX_TOKENS="16"
  else
    WARMUP_MAX_TOKENS="16"
  fi
fi
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"
rm -f "$SERVER_LOG" "$BENCH_JSON" "$STARTUP_FILE" "$WARMUP_FILE" "$QUALITY_EVAL_JSON"

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  BASE_PATH="$BASE_PATH" MODEL_ID="$MODEL_ID" DFLASH_DRAFT_MODEL="$DFLASH_DRAFT_MODEL" DFLASH_USE_MLX_NATIVE_DRAFTER="$DFLASH_USE_MLX_NATIVE_DRAFTER" SETTINGS_ENABLE_THINKING="$SETTINGS_ENABLE_THINKING" SETTINGS_THINKING_BUDGET="$SETTINGS_THINKING_BUDGET" uv run python - <<'PY'
import os
from pathlib import Path
from omlx.model_settings import ModelSettingsManager, apply_dflash_fast_model_settings

base = Path(os.environ["BASE_PATH"])
settings = ModelSettingsManager(base)
current = settings.get_settings(os.environ["MODEL_ID"])
thinking_budget = os.environ.get("SETTINGS_THINKING_BUDGET", "").strip()
current = apply_dflash_fast_model_settings(
    current,
    draft_model=os.environ["DFLASH_DRAFT_MODEL"],
    enable_thinking=os.environ.get("SETTINGS_ENABLE_THINKING", "0") == "1",
    thinking_budget=int(thinking_budget) if thinking_budget else None,
)
current.dflash_use_mlx_native_drafter = os.environ["DFLASH_USE_MLX_NATIVE_DRAFTER"] == "1"
settings.set_settings(os.environ["MODEL_ID"], current)
PY
else
  BASE_PATH="$BASE_PATH" MODEL_ID="$MODEL_ID" uv run python - <<'PY'
import os
from pathlib import Path
from omlx.model_settings import ModelSettingsManager

base = Path(os.environ["BASE_PATH"])
settings = ModelSettingsManager(base)
current = settings.get_settings(os.environ["MODEL_ID"])
current.dflash_enabled = False
current.dflash_auto_profile = False
settings.set_settings(os.environ["MODEL_ID"], current)
PY
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Fast pre-checks.
uv run python -m py_compile \
  omlx/engine/batched.py \
  omlx/model_settings.py \
  omlx/cli.py \
  omlx/server.py \
  omlx/dflash/*.py \
  scripts/siliconflash_benchmark.py \
  scripts/siliconflash_quality_eval.py

START_TS=$(uv run python - <<'PY'
import time
print(time.perf_counter())
PY
)

uv run python -m omlx.cli serve \
  --base-path "$BASE_PATH" \
  --model-dir "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-memory "$MAX_MODEL_MEMORY" \
  --max-process-memory "$MAX_PROCESS_MEMORY" \
  --paged-ssd-cache-max-size "$CACHE_MAX_SIZE" \
  --hot-cache-max-size "$HOT_CACHE_MAX_SIZE" \
  --initial-cache-blocks "$INITIAL_CACHE_BLOCKS" \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "$START_TIMEOUT"); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited unexpectedly. Last log lines:" >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done

if ! curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "Timed out waiting for server startup. Last log lines:" >&2
  tail -n 120 "$SERVER_LOG" >&2 || true
  exit 1
fi

STARTUP_S=$(uv run python - <<PY
import time
print(time.perf_counter() - float(${START_TS}))
PY
)
printf '%s\n' "$STARTUP_S" > "$STARTUP_FILE"

WARMUP_S=$(BASE_URL="$BASE_URL" MODEL_ID="$MODEL_ID" WARMUP_PROMPT="$WARMUP_PROMPT" WARMUP_MAX_TOKENS="$WARMUP_MAX_TOKENS" WARMUP_DFLASH="$WARMUP_DFLASH" IGNORE_EOS="$IGNORE_EOS" WARMUP_ENABLE_THINKING="$WARMUP_ENABLE_THINKING" THINKING_BUDGET="$THINKING_BUDGET" WARMUP_REPEAT="$WARMUP_REPEAT" uv run python - <<'PY'
import os
import time
import requests

payload = {
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": os.environ["WARMUP_PROMPT"]}],
    "max_tokens": int(os.environ["WARMUP_MAX_TOKENS"]),
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "min_p": 0.0,
    "stream": False,
    "dflash": os.environ.get("WARMUP_DFLASH", "0") == "1",
}
if os.environ.get("WARMUP_ENABLE_THINKING", "0") == "1":
    payload["chat_template_kwargs"] = {"enable_thinking": True}
if payload["dflash"] and os.environ.get("IGNORE_EOS", "0") == "1":
    payload["ignore_eos"] = True
thinking_budget = os.environ.get("THINKING_BUDGET", "")
if thinking_budget:
    payload["thinking_budget"] = int(thinking_budget)
repeats = max(1, int(os.environ.get("WARMUP_REPEAT", "1")))
start = time.perf_counter()
for _ in range(repeats):
    resp = requests.post(f"{os.environ['BASE_URL']}/v1/chat/completions", json=payload, timeout=1800)
    resp.raise_for_status()
print(time.perf_counter() - start)
PY
)
printf '%s\n' "$WARMUP_S" > "$WARMUP_FILE"

if [[ "$BENCHMARK_DFLASH" == "1" ]]; then
  if [[ "$IGNORE_EOS" == "1" ]]; then
    uv run python scripts/siliconflash_benchmark.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      --runs 1 \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$BENCH_PROMPT" \
      --dflash \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      --ignore-eos \
      > "$BENCH_JSON"
  else
    uv run python scripts/siliconflash_benchmark.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      --runs 1 \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$BENCH_PROMPT" \
      --dflash \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      > "$BENCH_JSON"
  fi
else
  if [[ "$IGNORE_EOS" == "1" ]]; then
    uv run python scripts/siliconflash_benchmark.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      --runs 1 \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$BENCH_PROMPT" \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      --ignore-eos \
      > "$BENCH_JSON"
  else
    uv run python scripts/siliconflash_benchmark.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      --runs 1 \
      --max-tokens "$MAX_TOKENS" \
      --prompt "$BENCH_PROMPT" \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      > "$BENCH_JSON"
  fi
fi

if [[ "$QUALITY_EVAL" == "1" ]]; then
  if [[ "$BENCHMARK_DFLASH" == "1" ]]; then
    uv run python scripts/siliconflash_quality_eval.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      --dflash \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      $( [[ "$IGNORE_EOS" == "1" ]] && printf '%s' '--ignore-eos ' )\
      --probe-prompt "$BENCH_PROMPT" \
      > "$QUALITY_EVAL_JSON"
  else
    uv run python scripts/siliconflash_quality_eval.py \
      --base-url "$BASE_URL" \
      --model "$MODEL_ID" \
      $( [[ "$ENABLE_THINKING" == "1" ]] && printf '%s' '--enable-thinking ' )\
      $( [[ -n "$THINKING_BUDGET" ]] && printf -- '--thinking-budget %q ' "$THINKING_BUDGET" )\
      $( [[ "$IGNORE_EOS" == "1" ]] && printf '%s' '--ignore-eos ' )\
      --probe-prompt "$BENCH_PROMPT" \
      > "$QUALITY_EVAL_JSON"
  fi
fi

BENCH_JSON="$BENCH_JSON" STARTUP_FILE="$STARTUP_FILE" WARMUP_FILE="$WARMUP_FILE" QUALITY_EVAL_JSON="$QUALITY_EVAL_JSON" uv run python - <<'PY'
import json
import os
from pathlib import Path

bench = json.loads(Path(os.environ["BENCH_JSON"]).read_text())
run = bench["results"][0]
startup_s = float(Path(os.environ["STARTUP_FILE"]).read_text())
warmup_s = float(Path(os.environ["WARMUP_FILE"]).read_text())
print(f"METRIC completion_tok_s={run['tok_s']}")
print(f"METRIC startup_s={startup_s}")
print(f"METRIC warmup_s={warmup_s}")
print(f"METRIC request_s={run['elapsed_s']}")
print(f"METRIC prompt_tokens={run['prompt_tokens']}")
print(f"METRIC completion_tokens={run['completion_tokens']}")

quality_path = Path(os.environ["QUALITY_EVAL_JSON"])
if quality_path.exists():
    quality = json.loads(quality_path.read_text())
    for key in (
        "case_count",
        "pass_count",
        "failed_cases",
        "pass_rate",
        "degenerate_cases",
        "repeat_ratio_mean",
        "repeat_ratio_max",
        "max_run_max",
        "mean_tok_s",
    ):
        value = quality.get(key)
        if isinstance(value, (int, float)):
            print(f"METRIC quality_eval_{key}={value}")

for key in (
    "dflash_draft_steps",
    "dflash_drafted_tokens",
    "dflash_accepted_tokens",
    "dflash_acceptance_rate",
    "dflash_commit_events",
    "dflash_mean_commit_tokens",
    "dflash_full_accept_steps",
    "dflash_full_accept_rate",
    "dflash_verify_passes",
    "dflash_target_forward_passes",
    "dflash_block_tokens_mean",
    "dflash_block_tokens_min",
    "dflash_block_tokens_max",
    "dflash_prefill_s",
    "dflash_draft_s",
    "dflash_verify_s",
    "dflash_eval_s",
    "dflash_total_s",
    "dflash_cache_restore_s",
    "dflash_cache_rollback_calls",
    "dflash_cache_trim_calls",
    "dflash_cache_trim_tokens",
    "dflash_cache_full_accept_clears",
    "dflash_split_full_attention_calls",
    "dflash_split_path_calls",
    "dflash_split_path_hit_rate",
    "dflash_split_exact_prefix_calls",
    "dflash_split_batched_2pass_calls",
    "dflash_split_batched_2pass_fallback_calls",
    "dflash_split_query_chunks",
    "dflash_async_submit_calls",
    "dflash_async_submit_to_consume_samples",
    "dflash_async_submit_to_consume_mean_s",
    "dflash_async_submit_to_consume_max_s",
    "dflash_async_submit_to_consume_min_s",
    "dflash_async_submit_unconsumed_steps",
    "dflash_draft_submit_steps",
    "dflash_draft_submit_mean_s",
    "dflash_draft_submit_min_s",
    "dflash_draft_submit_max_s",
    "dflash_draft_sync_eval_wait_steps",
    "dflash_draft_sync_eval_wait_mean_s",
    "dflash_draft_sync_eval_wait_min_s",
    "dflash_draft_sync_eval_wait_max_s",
    "dflash_verify_submit_steps",
    "dflash_verify_submit_mean_s",
    "dflash_verify_submit_min_s",
    "dflash_verify_submit_max_s",
    "dflash_verify_host_gap_steps",
    "dflash_verify_host_gap_mean_s",
    "dflash_verify_host_gap_min_s",
    "dflash_verify_host_gap_max_s",
    "dflash_verify_eval_wait_steps",
    "dflash_verify_eval_wait_mean_s",
    "dflash_verify_eval_wait_min_s",
    "dflash_verify_eval_wait_max_s",
    "dflash_verify_eval_wait_fused_steps",
    "dflash_verify_eval_wait_fused_mean_s",
    "dflash_verify_eval_wait_fused_min_s",
    "dflash_verify_eval_wait_fused_max_s",
    "dflash_verify_eval_wait_unfused_steps",
    "dflash_verify_eval_wait_unfused_mean_s",
    "dflash_verify_eval_wait_unfused_min_s",
    "dflash_verify_eval_wait_unfused_max_s",
    "dflash_verify_eval_wait_unfused_target_posterior_steps",
    "dflash_verify_eval_wait_unfused_target_posterior_mean_s",
    "dflash_verify_eval_wait_unfused_target_posterior_min_s",
    "dflash_verify_eval_wait_unfused_target_posterior_max_s",
    "dflash_verify_eval_wait_unfused_draft_logits_steps",
    "dflash_verify_eval_wait_unfused_draft_logits_mean_s",
    "dflash_verify_eval_wait_unfused_draft_logits_min_s",
    "dflash_verify_eval_wait_unfused_draft_logits_max_s",
    "dflash_verify_eval_fused_steps",
    "dflash_verify_eval_unfused_steps",
    "dflash_ddtree_enabled",
    "dflash_ddtree_native_runtime",
    "dflash_ddtree_tree_budget",
    "dflash_ddtree_cycles_completed",
    "dflash_ddtree_ddtree_cycles_completed",
    "dflash_ddtree_dflash_cycles_completed",
    "dflash_ddtree_dflash_accepted_from_draft",
    "dflash_ddtree_avg_acceptance",
    "dflash_ddtree_tokens_per_second",
    "dflash_ddtree_fast_path_ratio",
    "dflash_ddtree_fast_path_count",
    "dflash_ddtree_slow_path_count",
    "dflash_ddtree_tree_aware_commit_count",
    "dflash_ddtree_tree_aware_linear",
    "dflash_ddtree_exact_commit",
    "dflash_ddtree_dflash_controller_enabled",
    "dflash_ddtree_dflash_controller_probe_count",
    "dflash_ddtree_dflash_controller_switch_count",
    "dflash_ddtree_elapsed_s",
    "dflash_ddtree_prefill_s",
    "dflash_ddtree_tree_build_s",
    "dflash_ddtree_tree_verify_s",
    "dflash_ddtree_tree_verify_linear_s",
    "dflash_ddtree_tree_verify_attention_s",
    "dflash_ddtree_commit_s",
    "dflash_ddtree_dflash_draft_s",
    "dflash_ddtree_dflash_verify_s",
    "dflash_ddtree_dflash_replay_s",
    "dflash_ddtree_dflash_commit_s",
    "dflash_thermal_sidecar_enabled",
    "dflash_thermal_sidecar_sample_interval_s",
    "dflash_thermal_sidecar_sample_timeout_s",
    "dflash_thermal_sidecar_samples",
    "dflash_thermal_sidecar_failures",
    "dflash_thermal_sidecar_last_sample_age_s",
    "dflash_thermal_sidecar_thermal_warning_samples",
    "dflash_thermal_sidecar_performance_warning_samples",
    "dflash_thermal_sidecar_cpu_power_status_samples",
    "dflash_thermal_sidecar_thermal_warning_level_max",
    "dflash_thermal_sidecar_performance_warning_level_max",
    "dflash_thermal_sidecar_cpu_power_status_max",
    "dflash_thermal_sidecar_cpu_speed_limit_samples",
    "dflash_thermal_sidecar_cpu_speed_limit_mean_pct",
    "dflash_thermal_sidecar_cpu_speed_limit_min_pct",
    "dflash_thermal_sidecar_cpu_speed_limit_max_pct",
    "dflash_thermal_sidecar_gpu_speed_limit_samples",
    "dflash_thermal_sidecar_gpu_speed_limit_mean_pct",
    "dflash_thermal_sidecar_gpu_speed_limit_min_pct",
    "dflash_thermal_sidecar_gpu_speed_limit_max_pct",
    "dflash_thermal_sidecar_cpu_scheduler_limit_samples",
    "dflash_thermal_sidecar_cpu_scheduler_limit_mean_pct",
    "dflash_thermal_sidecar_cpu_scheduler_limit_min_pct",
    "dflash_thermal_sidecar_cpu_scheduler_limit_max_pct",
    "dflash_thermal_sidecar_cpu_available_samples",
    "dflash_thermal_sidecar_cpu_available_mean",
    "dflash_thermal_sidecar_cpu_available_min",
    "dflash_thermal_sidecar_cpu_available_max",
    "dflash_mx_active_mem_mean_bytes",
    "dflash_mx_active_mem_max_bytes",
    "dflash_mx_cache_mem_mean_bytes",
    "dflash_mx_cache_mem_max_bytes",
    "dflash_mx_peak_mem_max_bytes",
    "dflash_mx_recommended_working_set_bytes",
    "dflash_mx_peak_over_recommended_ratio",
    "dflash_mx_peak_over_recommended_events",
    "dflash_collapse_watchdog_enabled",
    "dflash_collapse_spike_events",
    "dflash_collapse_severe_spike_events",
    "dflash_collapse_safe_mode_active",
    "dflash_collapse_safe_mode_activations",
    "dflash_collapse_safe_mode_step",
    "dflash_collapse_safe_mode_block_tokens",
    "dflash_collapse_async_drain_calls",
    "dflash_collapse_async_drain_s",
    "dflash_collapse_clear_cache_calls",
    "dflash_collapse_safe_sync_calls",
    "dflash_collapse_safe_sync_s",
    "dflash_collapse_eval_step_mean_s",
    "dflash_collapse_eval_step_max_s",
    "dflash_collapse_eval_step_min_s",
    "dflash_clear_cache_calls",
    "dflash_adaptive_growth_events",
    "dflash_adaptive_shrink_events",
):
    value = run.get(key)
    if isinstance(value, bool):
        continue
    if isinstance(value, (int, float)):
        print(f"METRIC {key}={value}")
PY
