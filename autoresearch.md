# Autoresearch: SiliconFlash OmniCoder DFlash bootstrap

## Objective
Optimize native DFlash integration inside oMLX for Apple Silicon, targeting **OmniCoder-9B bf16** as the primary model. The immediate workload is a reproducible local oMLX server startup, a tiny warmup request that forces lazy model load without warming the benchmark prefix, then one deterministic **256-token** chat-completion benchmark request against `models/omni9b-phase2prime/`. This remains the least pathological workload we have found for exposing generation-path changes before generator modifications.

## Metrics
- **Primary**: `completion_tok_s` (tokens/s, higher is better) — steady-state completion throughput for the benchmark request after model warmup
- **Secondary**: `startup_s`, `warmup_s`, `request_s`, `prompt_tokens`, `completion_tokens` — monitors server startup cost, model-load warmup cost, and request shape

## How to Run
`./autoresearch.sh` — starts an isolated local oMLX server, waits for health, sends one tiny model-load warmup request with a distinct prompt, then runs one deterministic benchmark request and prints structured `METRIC ...` lines.

## Files in Scope
- `omlx/engine/batched.py` — main text-generation engine; likely DFlash entry point
- `omlx/model_settings.py` — per-model feature flags/settings; likely place for DFlash config fields
- `omlx/cli.py` — CLI surface if new user-facing flags are needed
- `omlx/server.py` — server bootstrap and engine pool integration
- `omlx/cache/paged_ssd_cache.py` — SSD-backed KV cache behavior that must be preserved
- `omlx/cache/tiered_manager.py` — hot/cold cache orchestration
- `omlx/dflash/__init__.py` — package export surface
- `omlx/dflash/config.py` — DFlash config object
- `omlx/dflash/interfaces.py` — typed contracts for draft/verify/context metrics
- `omlx/dflash/context_fusion.py` — target hidden-state extraction and fusion seam
- `omlx/dflash/drafter.py` — block-diffusion drafter runtime seam
- `omlx/dflash/verify.py` — exact-match acceptance logic and target verify seam
- `omlx/dflash/verify_kernel.py` — Python wrapper for future Metal verify kernel
- `omlx/dflash/verify_kernel.metal` — future custom Metal kernel source
- `omlx/dflash/runtime.py` — eventual orchestration layer
- `omlx/dflash/metrics.py` — runtime metrics helpers
- `scripts/siliconflash_benchmark.py` — benchmark request driver
- `scripts/siliconflash_start_baseline.sh` — manual local server launcher
- `scripts/siliconflash_smoketest.sh` — manual smoke test
- `docs/siliconflash-workspace.md` — workspace notes and operator instructions
- `autoresearch.sh` — benchmark harness used by the experiment loop
- `autoresearch.md` — experiment context and historical notes

## Off Limits
- `models/` — copied local model artifacts; benchmark inputs only
- packaging/app release behavior unrelated to DFlash
- unrelated model families or non-LLM engines
- external dependencies unless absolutely necessary

## Constraints
- Preserve oMLX paged SSD KV cache semantics
- Preserve OpenAI/Anthropic-compatible APIs
- Primary target model is local `models/omni9b-phase2prime/`
- Day-1 focus: bf16 and single-request path first
- Use `uv` for Python commands
- Keep benchmark deterministic (`temperature=0`)
- Favor minimal, reviewable changes over speculative rewrites

## What's Been Tried
- Bootstrapped this repo as an isolated copy of upstream oMLX
- Copied local OmniCoder-9B bf16 model into `models/omni9b-phase2prime/`
- Added a non-invasive `omlx/dflash/` scaffolding package to define integration boundaries
- Added baseline helper scripts for local start, smoke test, and benchmark
- Identified `ModelSettings` + `BatchedEngine.start()` as the cleanest initial seam because upstream already has experimental `SpecPrefill` plumbing
- Found a real single-request win by setting `max-concurrent-requests=1` in the harness for the Day-1 workload
- Found a stronger win by raising `BatchedEngine` default `stream_interval` from 1 to 8
- After switching to a warmed-model benchmark, discovered that tiny differences around the 64-token workload remain noisy enough to obscure small engine wins; moved the baseline to 256 generated tokens for better signal
- Kept a non-streaming fast path in `EngineCore`/`Scheduler`: `generate()` requests are marked `stream_output=False`, skip per-token detokenization/new_text construction, and only construct scheduler outputs on finish
- Re-tested reduced event-loop yields after the non-streaming finish-only path landed; unlike the earlier attempt, this now helps and is kept with an 8-step cadence
- Attempted to reduce noise with a median-of-3 measured workload, but repeated benchmark requests inside one server run became progressively slower and made the workload pathological; reverted to the single-request warmed-model benchmark
- Rejected `stream_interval=4`, `stream_interval=16`, and reverting `stream_interval` back to 1
- Rejected a non-aggregating collector fast path for non-streaming generate()
- Rejected disabling `RequestStreamState` for non-streaming requests (broke response delivery)
- Rejected collector bypass / direct final-output handoff after a strong first sample failed to reproduce
- Rejected smaller trims like avoiding final `output_token_ids` copies, avoiding final `new_token_ids` allocation, and replacing collector draining with a single read
- Rejected an idle-engine direct synchronous generation path that bypassed the background async loop
- Explicitly setting `top_p=1.0` in the benchmark did not improve throughput over the current baseline
- Probed mlx-lm's built-in speculative decoding (`draft_model` in `stream_generate`) as a possible shortcut, but OmniCoder/Qwen3.5 currently fails that path with `Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'})`, so a native oMLX/MLX integration is still required
- Realized the original benchmark was dominated by lazy model load on the first request, so the workload was redefined around a warmed model with a cold benchmark prefix
