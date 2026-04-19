# DFlash Out-of-Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the built-in throughput benchmark exercise the safe non-streaming DFlash path when available, so interface benchmarking reflects real tuned server behavior instead of silently bypassing DFlash.

**Architecture:** Keep the existing streaming benchmark path for standard PP/TG and batching measurements. Add an automatic single-request mode selector in the admin benchmark backend that switches compatible single-request runs to non-streaming `engine.generate(..., dflash=True)`, returns only the metrics that are honestly measurable there, and renders unavailable fields as `-` in the UI.

**Tech Stack:** Python, FastAPI/oMLX admin backend, Alpine dashboard UI, pytest.

---

### Task 1: Add failing benchmark-selection tests

**Files:**
- Modify: `tests/test_benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for single-request mode auto-selection**
- [ ] **Step 2: Run only the new benchmark tests and verify they fail for the missing helper/behavior**
- [ ] **Step 3: Add the minimal backend helper(s) and non-streaming benchmark branch in `omlx/admin/benchmark.py`**
- [ ] **Step 4: Re-run the targeted benchmark tests and verify they pass**

### Task 2: Render unavailable metrics honestly

**Files:**
- Modify: `omlx/admin/static/js/dashboard.js`
- Modify: `omlx/admin/templates/dashboard/_bench.html`
- Test: `tests/test_benchmark.py` (backend contract), manual UI verification

- [ ] **Step 1: Update the dashboard formatting path to render nullable metrics as `-`**
- [ ] **Step 2: Keep existing standard benchmark rows unchanged**
- [ ] **Step 3: Re-run targeted backend tests and any relevant smoke checks**

### Task 3: Preserve benchmark upload/summary behavior

**Files:**
- Modify: `omlx/admin/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Ensure upload/summary code tolerates non-standard single-request rows without fake values**
- [ ] **Step 2: Add/update tests only if needed by the changed contract**
- [ ] **Step 3: Run the full benchmark test file**

### Task 4: Verify the slice end-to-end

**Files:**
- Modify: none or touched files above
- Test: `tests/test_benchmark.py`, targeted related tests

- [ ] **Step 1: Run `uv run pytest -q tests/test_benchmark.py`**
- [ ] **Step 2: Run a broader targeted suite covering touched files**
- [ ] **Step 3: Summarize what this phase fixes and what is still intentionally out of scope (streaming/thinking/multi-request DFlash support)**
