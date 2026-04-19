# DFlash Compatibility Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DFlash behavior understandable in normal app/API usage by exposing whether it ran, and why it fell back when it did not.

**Architecture:** Add DFlash decision metadata at the engine layer, attach it to generation outputs, surface it through API usage fields, and improve runtime gate reasons so fallbacks are actionable rather than generic.

**Tech Stack:** Python, FastAPI, Pydantic, pytest.

---

### Task 1: Add failing API-contract tests

**Files:**
- Modify: `tests/test_stream_usage.py`
- Modify: `tests/integration/test_server_endpoints.py`

- [ ] **Step 1: Add failing tests for new DFlash usage fields on the Usage model**
- [ ] **Step 2: Add a failing integration test proving `/v1/chat/completions` returns DFlash decision metadata when present on the engine output**
- [ ] **Step 3: Run only the new tests and verify they fail for the missing fields/behavior**

### Task 2: Improve runtime gate reasons

**Files:**
- Modify: `omlx/dflash/runtime.py`
- Modify: `omlx/dflash/config.py` (only if needed)
- Test: reuse targeted tests or add unit tests later if necessary

- [ ] **Step 1: Replace generic unsupported reasons with explicit messages for streaming, batching, sampling, and stop-string limitations**
- [ ] **Step 2: Keep request semantics unchanged**

### Task 3: Propagate DFlash decision metadata through engine outputs

**Files:**
- Modify: `omlx/engine/base.py`
- Modify: `omlx/engine/batched.py`
- Modify: `tests/integration/test_server_endpoints.py` mock output helpers if needed

- [ ] **Step 1: Extend `GenerationOutput` with optional backend metadata**
- [ ] **Step 2: Populate DFlash decision metadata in both `generate()` and `stream_generate()`**
- [ ] **Step 3: Preserve existing generation behavior while adding metadata only**

### Task 4: Surface DFlash metadata in API usage

**Files:**
- Modify: `omlx/api/openai_models.py`
- Modify: `omlx/server.py`
- Modify: `tests/test_stream_usage.py`
- Modify: `tests/integration/test_server_endpoints.py`

- [ ] **Step 1: Extend `Usage` with optional DFlash fields**
- [ ] **Step 2: Include them in non-streaming chat/completion responses when available**
- [ ] **Step 3: Include them in streaming usage chunks when available**
- [ ] **Step 4: Verify `exclude_none` behavior remains clean**

### Task 5: Verify the slice

**Files:**
- Modify: touched files above
- Test: targeted suites

- [ ] **Step 1: Run `uv run pytest -q tests/test_stream_usage.py tests/integration/test_server_endpoints.py -k dflash`**
- [ ] **Step 2: Run a broader targeted suite covering touched files**
- [ ] **Step 3: Summarize what is now visible and what still requires later phases (true streaming, thinking, sampling, multi-request support)**
