# SiliconFlash: DFlash Bring-up Research for oMLX

> **Research snapshot branch** focused on making DFlash work in oMLX on Apple Silicon.
>
> This repository is **not** presented as an official oMLX product release page.

## What this repo is

A reproducible research branch containing:
- oMLX code changes related to DFlash integration and runtime behavior
- the full autoresearch log of experiments (wins, regressions, crashes)
- benchmark harness/scripts used to produce the results

## What this repo is not

- a model host (no model weights included)
- a polished product README for upstream oMLX
- a benchmark cherry-pick (negative results are preserved)

---

## TL;DR findings

### Throughput outcome (current best known, throughput-first)

- **DFlash config:** `185.83 tok/s` (run **3234**, commit `03ebbdd`)
- **Same-window explicit no-DFlash control:** `89.54 tok/s` (run **3235**)
- **Gain:** **~2.08x** (`+107.5%`)

### Quality-guarded high-throughput lane

- **DFlash config:** `184.27 tok/s` (run **3355**, commit `4679277`)
- Quality eval profile at that point: `pass_rate=0.75`, `degenerate_cases=0`
- Same-window controls in that segment stayed around high-80 tok/s

> Full details are in `autoresearch.jsonl`.

---

## Best-known configurations

### 1) Throughput-first (max observed tok/s)

```bash
export DFLASH_MIRROR_SD_RUNTIME=1
export DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD=0
export DFLASH_BLOCK_TOKENS=10
export DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=0
export DFLASH_DRAFT_SINK=48
```

### 2) Quality-guarded high-throughput lane

```bash
export DFLASH_MIRROR_SD_RUNTIME=1
export DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD=0
export DFLASH_BLOCK_TOKENS=11
export DFLASH_VERIFY_LEN=9
export DFLASH_VERIFY_LEN_WARMUP_STEPS=8
export DFLASH_VERIFY_LEN_WARMUP_CAP=2
export DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=0
export DFLASH_DRAFT_SINK=64
```

---

## Model references used in this research

- **Target model:** `Qwen3.6-35B-A3B-MLX-8bit`
- **Draft model:** `z-lab/Qwen3.6-35B-A3B-DFlash`

Use local model paths through env vars (no machine-specific absolute paths):

```bash
export AR_MODEL_DIR=/path/to/your/mlx-models
export AR_MODEL_ID=Qwen3.6-35B-A3B-MLX-8bit
export AR_DFLASH_DRAFT_MODEL=z-lab/Qwen3.6-35B-A3B-DFlash
```

---

## Reproducing the benchmark harness

```bash
./autoresearch.sh
```

The harness emits structured `METRIC ...` lines including primary throughput and auxiliary quality/runtime telemetry.

Key files:
- `autoresearch.sh` — benchmark + warmup + quality-eval orchestration
- `scripts/siliconflash_benchmark.py` — benchmark request driver
- `scripts/siliconflash_quality_eval.py` — fixed quality-eval suite

---

## Where to inspect the research trail

- `autoresearch.jsonl` — full experiment history (kept/discarded/crash + notes)
- `autoresearch.md` — objective, constraints, workload definition
- `autoresearch.ideas.md` — deferred promising ideas

---

## Code areas with core DFlash work

- `omlx/dflash/` — runtime, verification, metrics, integration paths
- `omlx/engine/batched.py` / `omlx/scheduler.py` / `omlx/engine_core.py` — generation-path and scheduling optimizations
- `omlx/model_settings.py` — DFlash-related model settings and tuning knobs
- `omlx/server.py` / `omlx/cli.py` — server-level integration surfaces

---

## Repository hygiene for public sharing

This branch intentionally excludes machine-local artifacts and model data via `.gitignore`, including:
- `/.omlx-*/`
- `/models/`
- `/mlx_models/`
- `/.pi/`
- `/external/dflash-mlx-bstnxbt/`

---

## Upstream

This work is based on the oMLX codebase. For upstream project documentation and product usage, see the original oMLX repository.
