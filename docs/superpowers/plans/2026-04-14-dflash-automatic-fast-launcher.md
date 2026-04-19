# DFlash Automatic Fast Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/siliconflash_start_qwen35_fast.sh` the canonical automatic-DFlash launcher so omitted sampling params resolve to DFlash-safe defaults across normal API/chat/harness flows, while `scripts/siliconflash_start_baseline.sh` remains stock/reference.

**Architecture:** Add a narrow fast-launch profile signal that is written by the fast launcher, then teach server-side sampling resolution to apply DFlash-safe defaults only when that profile is active and only when the client omitted sampling params. Preserve explicit client sampling intent, preserve honest fallback metadata, and validate the behavior through API, streaming chat, and benchmark-style request shapes.

**Tech Stack:** Bash launchers, FastAPI server, Python dataclasses/settings, pytest, integration tests, repo-local benchmark/live smoke commands.

---

## File Structure

**Create:**
- `docs/superpowers/plans/2026-04-14-dflash-automatic-fast-launcher.md` — this plan

**Modify:**
- `scripts/siliconflash_start_qwen35_fast.sh` — declare/enable the fast-launch automatic-DFlash profile consistently
- `scripts/siliconflash_start_baseline.sh` — keep stock/reference behavior explicit and unmodified by the fast profile
- `omlx/model_settings.py` — extend reusable fast-profile settings helper(s) if needed
- `omlx/server.py` — implement narrow omitted-param resolution for the fast-launch profile only
- `tests/test_server.py` — unit coverage for omitted-param resolution vs explicit-param preservation
- `tests/integration/test_server_endpoints.py` — endpoint-level behavior checks for usage metadata and request-shape outcomes
- `tests/test_model_settings.py` — assert launcher/profile settings are written consistently

**Maybe modify (only if tests show a gap):**
- `omlx/api/openai_models.py` — only if additional usage-field plumbing is needed
- `omlx/admin/templates/chat.html` or related UI files — only if the fast-launch visibility gap is specifically in chat/admin surfaces

---

### Task 1: Lock down the fast-launch profile contract in tests

**Files:**
- Modify: `tests/test_server.py`
- Modify: `tests/test_model_settings.py`
- Reference: `omlx/server.py`, `omlx/model_settings.py`

- [ ] **Step 1: Add a failing unit test for omitted sampling params under the fast-launch profile**

Add a test in `tests/test_server.py` that sets up server state/settings to represent the fast-launch profile and calls `get_sampling_params(...)` with omitted request sampling params.

Expected assertions:
- temperature resolves to `0.0`
- top_p resolves to `1.0`
- top_k resolves to `0`
- min_p resolves to `0.0`
- explicit request omission is treated differently from explicit request values

- [ ] **Step 2: Add a failing unit test proving explicit client params are preserved**

Add a second test in `tests/test_server.py` that exercises the same fast-launch profile, but passes explicit non-greedy request params.

Expected assertions:
- explicit request `temperature`, `top_p`, `min_p`, or similar values are preserved
- the fast-launch omitted-param policy does not override explicit client intent

- [ ] **Step 3: Add a failing settings test for the fast launcher profile marker/settings contract**

In `tests/test_model_settings.py`, add a test that verifies the fast-launch helper/path writes the intended DFlash-safe model settings contract consistently for the target model.

Minimum assertions:
- `dflash_enabled == True`
- `dflash_draft_model` is set
- `dflash_use_mlx_native_drafter == True`
- `temperature == 0.0`
- `top_p == 1.0`
- `top_k == 0`
- `min_p == 0.0`
- `force_sampling == False`

- [ ] **Step 4: Run the focused failing tests**

Run:
```bash
uv run pytest -q tests/test_server.py tests/test_model_settings.py -k 'dflash or sampling or fast'
```

Expected:
- new tests fail for the right reason
- no unrelated broad failures block understanding

- [ ] **Step 5: Commit the failing-test checkpoint**

```bash
git add tests/test_server.py tests/test_model_settings.py
git commit -m "test: cover automatic dflash fast-launch defaults"
```

---

### Task 2: Implement profile-scoped omitted-param resolution in the server

**Files:**
- Modify: `omlx/server.py`
- Modify: `omlx/model_settings.py`
- Reference: `scripts/siliconflash_start_qwen35_fast.sh`

- [ ] **Step 1: Identify the narrow signal that activates automatic DFlash mode**

Use the existing fast-launcher/model-settings path rather than inventing a broad global switch. Prefer a profile-scoped signal that can be derived from launcher-written state and is not active for the baseline launcher.

Implementation note:
- keep this signal narrow and explicit
- do not apply the behavior globally to all DFlash-enabled servers unless the fast profile is clearly active

- [ ] **Step 2: Implement omitted-param coercion only for omitted params under the fast profile**

In `omlx/server.py` `get_sampling_params(...)`, add logic so that:
- omitted request sampling params resolve to DFlash-safe defaults only when the fast-launch profile is active
- explicit client sampling params still win
- baseline/non-fast paths keep current stock behavior

- [ ] **Step 3: Keep the implementation minimal and local**

Do not add broad request rewriting deeper in the generation stack if `get_sampling_params(...)` can handle the contract cleanly.

- [ ] **Step 4: Re-run the focused unit tests**

Run:
```bash
uv run pytest -q tests/test_server.py tests/test_model_settings.py -k 'dflash or sampling or fast'
```

Expected:
- new tests pass
- existing related tests remain green

- [ ] **Step 5: Commit the minimal implementation**

```bash
git add omlx/server.py omlx/model_settings.py tests/test_server.py tests/test_model_settings.py
git commit -m "feat: apply automatic dflash defaults for fast launcher"
```

---

### Task 3: Make the fast launcher write the contract clearly and keep baseline clean

**Files:**
- Modify: `scripts/siliconflash_start_qwen35_fast.sh`
- Modify: `scripts/siliconflash_start_baseline.sh`
- Reference: `omlx/model_settings.py`

- [ ] **Step 1: Add a failing launcher/settings test if needed**

If the current tests do not already pin launcher behavior sufficiently, add or extend tests to assert:
- fast launcher writes/activates the fast DFlash profile contract
- baseline launcher does not opt into it

Prefer focused tests over brittle full-script execution.

- [ ] **Step 2: Refactor the fast launcher to use the shared fast-settings helper consistently**

Update `scripts/siliconflash_start_qwen35_fast.sh` so the launcher writes the same DFlash-safe model settings contract used by tests and server expectations.

Key constraints:
- keep DFlash enabled and current validated backend choices intact
- keep thinking/profile settings intact
- do not add broad global behavior beyond the fast-launch path

- [ ] **Step 3: Make baseline launcher explicitly remain stock/reference**

If needed, add a small comment or guard in `scripts/siliconflash_start_baseline.sh` so its role stays clear and it does not accidentally opt into fast-profile behavior.

- [ ] **Step 4: Syntax-check launchers and run focused tests**

Run:
```bash
bash -n scripts/siliconflash_start_qwen35_fast.sh
bash -n scripts/siliconflash_start_baseline.sh
uv run pytest -q tests/test_model_settings.py tests/test_server.py -k 'dflash or sampling or fast'
```

Expected:
- shell syntax checks pass
- focused tests pass

- [ ] **Step 5: Commit launcher/profile wiring**

```bash
git add scripts/siliconflash_start_qwen35_fast.sh scripts/siliconflash_start_baseline.sh omlx/model_settings.py tests/test_model_settings.py tests/test_server.py
git commit -m "feat: make fast launcher the automatic dflash profile"
```

---

### Task 4: Verify API and streaming chat surfaces behave correctly

**Files:**
- Modify: `tests/integration/test_server_endpoints.py`
- Maybe modify: `omlx/server.py`
- Maybe modify: `omlx/api/openai_models.py`

- [ ] **Step 1: Add a failing integration test for plain API omitted-param requests**

Add/extend an integration test in `tests/integration/test_server_endpoints.py` that simulates the fast-launch profile and sends a plain OpenAI-compatible request with omitted sampling params.

Expected assertions:
- request succeeds
- usage metadata includes DFlash fields
- the mock/recorded path shows the effective sampling shape is DFlash-safe

- [ ] **Step 2: Add a failing integration test for explicit non-greedy fallback visibility**

Add a second test that sends explicit non-greedy request params under the fast-launch profile.

Expected assertions:
- request succeeds
- DFlash does not have to run
- usage metadata exposes `dflash_used == false` and a non-empty reason

- [ ] **Step 3: Add a failing streaming/chat-style regression test if missing**

Add/extend a streaming or chat-style endpoint test that ensures the fast profile does not regress the previously fixed reasoning/content behavior on plain streamed chat requests.

Expected assertions:
- stream completes cleanly
- no garbled reasoning/content structure
- DFlash metadata remains visible where exposed

- [ ] **Step 4: Implement any endpoint-layer fixes needed to satisfy the tests**

Only make endpoint changes that are necessary for:
- omitted-param fast-profile consistency
- explicit fallback visibility
- preserved reasoning/chat behavior

Avoid broad API-wide coercion.

- [ ] **Step 5: Run the focused integration suite**

Run:
```bash
uv run pytest -q tests/integration/test_server_endpoints.py tests/test_server.py tests/test_model_settings.py -k 'dflash or sampling or stream or chat'
```

Expected:
- all touched endpoint/server/settings tests pass

- [ ] **Step 6: Commit the surface-consistency pass**

```bash
git add omlx/server.py omlx/api/openai_models.py tests/integration/test_server_endpoints.py tests/test_server.py tests/test_model_settings.py
git commit -m "feat: preserve automatic dflash behavior across api and chat surfaces"
```

---

### Task 5: Live verification through the real fast launcher

**Files:**
- Modify only if verification exposes a real bug in touched files
- Reference: `scripts/siliconflash_start_qwen35_fast.sh`, `scripts/siliconflash_start_baseline.sh`

- [ ] **Step 1: Start the fast launcher locally**

Run:
```bash
OMLX_PORT=8022 scripts/siliconflash_start_qwen35_fast.sh
```

Expected:
- server starts
- warmup completes
- no auth/warmup regression

- [ ] **Step 2: Verify plain omitted-param OpenAI-style request uses DFlash**

Send a plain request with omitted sampling params (only model/messages/max_tokens/stream as needed).

Expected:
- request succeeds
- usage shows `dflash_requested=true`
- usage shows `dflash_used=true`

- [ ] **Step 3: Verify a plain streamed chat-style request behaves correctly**

Send a streamed chat-style request without per-request DFlash glue.

Expected:
- coherent streamed output
- no garbled reasoning/content behavior
- DFlash usage visible/consistent

- [ ] **Step 4: Verify explicit non-greedy request falls back honestly**

Send a request with explicit non-greedy params such as `temperature=0.7`.

Expected:
- request still succeeds
- usage shows `dflash_used=false`
- usage includes a human-meaningful fallback reason

- [ ] **Step 5: Run the baseline launcher spot-check**

Run:
```bash
OMLX_PORT=8023 scripts/siliconflash_start_baseline.sh
```

Expected:
- no fast-profile automatic behavior leaks into baseline mode

- [ ] **Step 6: Commit any necessary live-fix follow-up**

```bash
git add scripts/siliconflash_start_qwen35_fast.sh scripts/siliconflash_start_baseline.sh omlx/server.py omlx/model_settings.py tests/test_server.py tests/integration/test_server_endpoints.py tests/test_model_settings.py
git commit -m "fix: harden automatic dflash fast-launch behavior"
```

Only do this if live verification required code changes.

---

### Task 6: Re-verify the honest benchmark lane

**Files:**
- Modify only if a real regression is found in already-touched files
- Reference: `.omlx-autoresearch/workload.env`, `autoresearch.sh`

- [ ] **Step 1: Restore the authoritative benchmark workload**

Before benchmarking, ensure the repo-local autoresearch files are back on the intended authoritative 1024-token thinking workload rather than a long-context exploratory lane.

- [ ] **Step 2: Run one honest benchmark using the existing harness**

Run:
```bash
./autoresearch.sh
```

Expected:
- benchmark completes on the real server path
- no cheating/semantic drift
- result is comparable to the established authoritative lane

- [ ] **Step 3: Judge the benchmark honestly**

If the automatic-fast-launch productization changes improve the primary metric, keep them.
If they are neutral or worse but functionally required, document that tradeoff clearly before deciding whether to keep or split the change further.

- [ ] **Step 4: Record learnings in autoresearch notes**

Update `autoresearch.ideas.md` with:
- what automatic-fast-launch behavior now works
- what still falls back and why
- any remaining cross-surface gaps

- [ ] **Step 5: Final commit if appropriate**

```bash
git add autoresearch.ideas.md .omlx-autoresearch/workload.env scripts/siliconflash_start_qwen35_fast.sh scripts/siliconflash_start_baseline.sh omlx/server.py omlx/model_settings.py tests/test_server.py tests/integration/test_server_endpoints.py tests/test_model_settings.py
git commit -m "feat: make fast launcher use dflash automatically across normal surfaces"
```

Only commit if the final integrated state is the one to keep.

---

## Verification Checklist

Run these before claiming the work is done:

```bash
bash -n scripts/siliconflash_start_qwen35_fast.sh
bash -n scripts/siliconflash_start_baseline.sh
uv run pytest -q tests/test_model_settings.py tests/test_server.py tests/integration/test_server_endpoints.py -k 'dflash or sampling or stream or chat'
OMLX_PORT=8022 scripts/siliconflash_start_qwen35_fast.sh
# then issue real omitted-param / explicit-non-greedy / streaming requests
./autoresearch.sh
```

## Notes for the Implementer

- Keep the omitted-param coercion **profile-scoped**. Do not reintroduce the earlier broad server-wide coercion that hurt the hot path.
- Respect explicit client sampling params.
- Prefer small changes in `get_sampling_params(...)` and launcher/model-settings wiring over deeper request rewriting.
- If a surface does not expose DFlash metadata yet, add the minimum visibility needed; do not expand unrelated schemas without reason.
- If the integrated benchmark regresses, split functionality-preserving fixes from performance-sensitive defaulting changes rather than forcing them to land together.
