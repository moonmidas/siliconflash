# SiliconFlash Workspace Bootstrap

This repository is now an isolated working copy of upstream `jundot/omlx`, prepared for native DFlash research on Apple Silicon.

## What was set up

- Upstream oMLX source copied into this folder
- Local OmniCoder bf16 model copied to:
  - `models/omni9b-phase2prime/`
- Initial SiliconFlash scaffolding added under:
  - `omlx/dflash/`
- Baseline helper scripts added under:
  - `scripts/siliconflash_start_baseline.sh`
  - `scripts/siliconflash_smoketest.sh`
  - `scripts/siliconflash_benchmark.py`

## Current status

This is intentionally **autoresearch-ready, not DFlash-complete**.

The new `omlx/dflash/` package defines the boundaries for:
- target hidden-state extraction and fusion
- drafter orchestration
- exact-match verification
- custom Metal kernel insertion
- runtime metrics for acceptance-rate benchmarking

It does **not** change oMLX serving behavior yet.

## Relevant upstream integration points

- CLI: `omlx/cli.py`
- server bootstrap: `omlx/server.py`
- per-model settings: `omlx/model_settings.py`
- batched text engine: `omlx/engine/batched.py`
- paged SSD cache manager: `omlx/cache/paged_ssd_cache.py`
- tiered cache manager: `omlx/cache/tiered_manager.py`

Notable finding: upstream already contains an experimental speculative path called `SpecPrefill`, loaded via `ModelSettings.specprefill_*` and wired in `omlx/engine/batched.py`. That makes `model_settings.py` and `BatchedEngine.start()` the cleanest initial seams for a native DFlash path.

## Baseline run flow

### 1. Install dependencies

```bash
uv sync --dev
```

### 2. Start oMLX locally against this repo's model folder

```bash
bash scripts/siliconflash_start_baseline.sh
```

### 3. Smoke test

```bash
bash scripts/siliconflash_smoketest.sh
```

### 4. Baseline benchmark

```bash
uv run python scripts/siliconflash_benchmark.py \
  --model omni9b-phase2prime \
  --runs 3 \
  --max-tokens 256
```

## Suggested Day-1 research target

- target model: `models/omni9b-phase2prime/`
- precision: bf16
- execution mode: single-request baseline first
- objective: establish stable baseline tok/s before generator patching

## Next implementation step

Patch the real oMLX generation path so DFlash becomes a first-class optional algorithm alongside the existing batched engine, while preserving:
- paged SSD KV cache
- continuous batching contracts
- OpenAI / Anthropic APIs
- admin / menu-bar app compatibility
