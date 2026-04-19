# DFlash Compatibility Expansion Design

**Goal:** Evolve DFlash from a narrow proven fast path into something that behaves predictably in real apps: visible fallbacks, true streaming support, thinking-aware support, broader sampling compatibility, and eventual multi-request operation.

## Program structure

### Phase A — Fallback visibility and diagnostics
Add machine-readable DFlash decision metadata to normal API responses so clients can tell:
- whether DFlash was requested
- whether it was actually used
- which backend was selected
- why it fell back when it did not run

This phase is foundational because it makes every later rollout observable.

### Phase B — Safe-default routing
Use the diagnostics from Phase A to tighten “default when safe” behavior:
- only auto-use DFlash when the request shape is known-supported
- otherwise fall back cleanly with explicit metadata
- avoid silent slow-path behavior

### Phase C — True streaming DFlash
Introduce a streaming-capable DFlash path that preserves SSE semantics and does not require buffering the whole response before clients see content.

### Phase D — Thinking + broader sampling compatibility
Expand support beyond strict greedy/no-thinking requests in measured, correctness-preserving slices:
- thinking-enabled prompt/runtime shapes
- stop strings / stop tokens interactions
- selective non-greedy compatibility where semantics are well-defined

### Phase E — Multi-request / continuous batching
Lift the single-request gate only after correctness and isolation are proven under concurrent request pressure.

## Immediate implementation choice
Start with **Phase A**.

It has the best leverage:
- fixes the current “why is my app slow?” usability problem
- makes UI/app benchmarking interpretable
- provides the reason labels needed to drive future benchmark and admin UX improvements

## Phase A design

### API surface
Extend OpenAI-style `usage` with optional oMLX-specific fields:
- `dflash_requested: bool | None`
- `dflash_used: bool | None`
- `dflash_reason: str | None`
- `dflash_backend: str | None`

These are extensions, like the existing timing fields.

### Engine/runtime contract
Add per-request backend metadata to `GenerationOutput`.

Batched engine will populate DFlash decision metadata for both:
- non-streaming `generate()`
- streaming `stream_generate()`

This metadata will reflect actual routing decisions, including fallback reasons from the runtime gates.

### Gate reason quality
Replace vague gate failures with actionable reasons, e.g.:
- streaming not yet supported
- continuous batching not yet supported
- greedy decoding only
- stop strings not yet supported
- runtime not ready / draft load failure

## Success criteria
- Normal `/v1/chat/completions` responses expose whether DFlash was used.
- Streaming usage chunks can expose the same metadata when usage is requested.
- Unsupported shapes become diagnosable instead of mysteriously slow.
- No request semantics change yet; this phase is visibility-first.

## Out of scope for this phase
- Implementing true streaming DFlash
- changing thinking behavior
- enabling non-greedy DFlash
- enabling multi-request DFlash
