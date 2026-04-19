# DFlash Automatic Fast Launcher Design

## Goal

Make `scripts/siliconflash_start_qwen35_fast.sh` the canonical **automatic DFlash** launcher so that normal oMLX usage through this repo’s fast server path “just works” across API clients, harnesses, agents, and chat without per-request `dflash=true` or other request-level glue.

`scripts/siliconflash_start_baseline.sh` remains the honest stock/reference launcher.

## Problem Statement

Today, DFlash can work well on the validated server path, but ordinary request flows can still silently fall off DFlash because:

- omitted sampling params may resolve to non-greedy defaults
- different entrypoints do not all inherit the same DFlash-safe defaults
- launcher/model-settings behavior is not yet a complete cross-surface contract
- users and harnesses often cannot tell whether DFlash actually ran unless they inspect detailed metadata

The user’s desired contract is simpler:

> If the server is started with the fast launcher, it should use DFlash automatically for normal compatible requests everywhere possible.

## Non-Goals

- changing stock oMLX behavior globally for all launch modes
- forcing DFlash onto explicitly incompatible requests
- hiding fallbacks or cheating benchmark semantics
- claiming true continuous speculative batching where only interleaving exists

## Constraints

- Preserve paged SSD KV cache semantics
- Preserve API compatibility
- Do not silently change explicit client intent
- Do not overfit or cheat the benchmark lane
- Keep `scripts/siliconflash_start_baseline.sh` as the stock/reference path
- Keep the fast path honest: if DFlash does not run, surfaces should say why

## Approved Design

### 1. Launcher Roles

#### Baseline launcher
`script/siliconflash_start_baseline.sh` remains the stock/reference launcher.

Properties:
- no automatic DFlash-oriented coercion
- useful for comparison, debugging, and honest stock baselines

#### Fast launcher
`scripts/siliconflash_start_qwen35_fast.sh` becomes the canonical automatic-DFlash launcher.

Properties:
- enables the validated DFlash backend/profile
- configures model settings and server behavior so normal omitted-param requests land on the DFlash-safe path
- is the recommended entrypoint for API harnesses, agents, chat, and other normal usage when the user wants DFlash

### 2. Request Contract Under the Fast Launcher

When the server is started via the fast launcher, the system should apply a **profile-scoped automatic DFlash policy**.

#### Omitted sampling params
If a request omits sampling params, the fast-launch profile should resolve them to DFlash-safe greedy defaults:

- `temperature = 0.0`
- `top_p = 1.0`
- `top_k = 0`
- `min_p = 0.0`
- `force_sampling = false`

This should make ordinary callers “just work” without requiring request-level tuning.

#### Explicit sampling params
If a client explicitly sends sampling params:
- respect them
- do not silently rewrite them to greedy
- if they are incompatible with DFlash, fall back honestly

This preserves client intent and avoids surprising harnesses or chat clients.

### 3. Fallback Policy

Requests that are not DFlash-compatible should:
- still complete successfully where normal generation supports them
- fall back cleanly
- expose why DFlash did not run

No silent pretending. No hidden coercion of explicit incompatible client parameters.

### 4. Visibility Contract

As much as each surface allows, the fast-launch path should expose:
- `dflash_requested`
- `dflash_used`
- `dflash_reason`
- `dflash_backend`

This matters for:
- OpenAI-compatible API clients
- harnesses and agents
- chat and admin/debug surfaces

The key requirement is that if DFlash does not run, users should not have to guess why.

### 5. Scope Boundary

This automatic behavior must be **profile-scoped**, not global.

That means:
- baseline launcher: stock/reference semantics
- fast launcher: automatic DFlash semantics for omitted params

This avoids the risk of broad global server coercion while still meeting the user’s operational goal.

## Why This Design

Three broad approaches were considered:

### A. Launcher-only defaults
Pros:
- small blast radius
- easy to reason about

Cons:
- can still be fragile if some request paths bypass launcher-written assumptions

### B. Global server coercion
Pros:
- maximum automation

Cons:
- too broad
- higher semantic risk
- prior experiments suggested broad server-side coercion can hurt the hot path

### C. Profile-scoped hybrid (**chosen**)
Pros:
- matches the user’s desired contract exactly
- keeps stock launcher clean
- gives consistent automatic behavior across API/chat/harnesses
- lower risk than global coercion

Cons:
- requires careful implementation across launcher, model settings, and request resolution

## Surfaces Covered by the Design

Under the fast launcher, the intended automatic-DFlash behavior should apply to:
- normal API requests with omitted sampling params
- harnesses targeting the API
- agents targeting the API
- built-in chat using the same model/profile
- benchmark requests run against the launched server

With the important caveat that **explicit incompatible parameters still win** and may trigger honest fallback.

## Compatibility Model

### Compatible requests today
Generally:
- greedy / greedy-equivalent request shapes
- validated bstnxbt MLX path
- existing thinking-capable fast path when request semantics stay compatible

### Likely incompatible requests today
Generally:
- explicitly non-greedy sampling
- some stop-string-heavy requests
- request shapes outside the currently supported DFlash runtime contract
- broader batching/concurrency shapes that exceed today’s honest support model

These should not be forced onto DFlash invisibly.

## Success Criteria

Starting the server with `scripts/siliconflash_start_qwen35_fast.sh` should mean:

1. Plain OpenAI-compatible API requests with omitted sampling params use DFlash automatically when otherwise compatible, and report `dflash_used=true`.
2. External harnesses/agents pointed at that server also use DFlash automatically without per-request `dflash=true`.
3. Plain streamed chat-style requests work through the same profile without garbled or misleading reasoning/content behavior.
4. Explicit non-greedy requests are respected, fall back honestly when incompatible with DFlash, and report `dflash_used=false` with a visible reason.
5. If DFlash does not run for any supported surface, the surface exposes an honest reason.
6. `scripts/siliconflash_start_baseline.sh` remains the stock/reference path.

## Implementation Outline

Implementation should likely be split into these areas:

1. **Fast-launch profile definition**
   - make the launcher write the intended DFlash-safe settings contract consistently

2. **Omitted-param resolution**
   - add a narrow, profile-scoped rule so omitted params resolve to DFlash-safe defaults only under the fast launcher/profile

3. **Surface consistency**
   - ensure API/chat/harness-visible surfaces use the same effective request contract

4. **Fallback visibility and debugging**
   - confirm all relevant surfaces expose DFlash decision metadata

5. **Verification**
   - live validation through launcher + API + chat-style request shapes
   - ensure the authoritative benchmark lane remains honest

## Risks

- accidentally broadening behavior outside the fast-launch profile
- silently changing explicit client semantics
- regressing the authoritative benchmark while fixing defaults
- fixing one surface while leaving another to silently fall off DFlash

## Mitigations

- keep the behavior strictly scoped to the fast-launch profile
- never override explicit sampling params
- validate API, chat, and benchmark shapes separately
- preserve visible fallback metadata everywhere practical

## Open Questions Resolved During Brainstorming

- Primary focus order: **defaults-first**, then **API/harness**, then **chat/UI**
- Desired operational contract: **if started via the fast launcher, it should just work automatically with DFlash**
- Baseline vs fast launcher roles: **baseline stays stock; fast becomes canonical DFlash launcher**
- Handling of explicit incompatible params: **respect them and fall back honestly**
- Handling of omitted sampling params under the fast launcher: **treat them as DFlash-safe defaults**

## Final Recommendation

Proceed with a **profile-scoped automatic DFlash launcher design**:

- `scripts/siliconflash_start_baseline.sh` stays stock/reference
- `scripts/siliconflash_start_qwen35_fast.sh` becomes the automatic-DFlash entrypoint
- omitted sampling params under the fast launcher resolve to DFlash-safe defaults
- explicit client params are respected
- incompatible explicit requests fall back honestly with visible reasons

This is the best match for the user’s stated goal of “just work automatically” without taking on the larger risk of changing global oMLX semantics.
