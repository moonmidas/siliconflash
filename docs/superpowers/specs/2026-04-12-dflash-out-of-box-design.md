# DFlash Out-of-Box Behavior Design

**Problem:** The tuned Qwen3.5 DFlash path is real, but it is not consistently reflected in the built-in oMLX interface, default request shapes, or benchmark surfaces. Today the admin throughput benchmark measures the streaming engine path, which bypasses the tuned non-streaming DFlash route entirely.

## Scope

This design splits the work into three phases:

1. **Benchmark the actual interface modes**
   - Make built-in benchmarking capable of exercising the same safe non-streaming DFlash path used by real fast server measurements.
   - Keep the existing streaming/continuous-batching benchmark path for standard PP/TG comparisons.
   - Clearly distinguish standard streaming benchmark results from non-streaming DFlash-style results.

2. **Make the fast path default when safe**
   - Preserve today’s request-level DFlash auto-use semantics when model settings enable it.
   - Improve visibility when requests fall off the DFlash-compatible path, instead of silently looking slow.

3. **Support broader real app shapes**
   - Add explicit validation/benchmark coverage for streaming, thinking-enabled requests, broader sampling settings, varied prompt/decode mixes, and multi-request behavior.
   - Only promote defaults when they are measured honestly and do not change request semantics.

## Phase 1 architecture

The highest-value immediate fix is in `omlx/admin/benchmark.py`.

- The current single-request benchmark uses `engine.stream_generate(...)` for everything.
- The tuned DFlash route lives in `engine.generate(...)` with `stream=False` and DFlash request support checks.
- Phase 1 will add an automatic single-request benchmark mode selector:
  - **standard**: keep current streaming benchmark path
  - **nonstream_dflash**: when a loaded engine has a supported DFlash runtime, benchmark single-request generation via non-streaming `engine.generate(..., dflash=True)`
- Batch tests remain unchanged because the current DFlash contract is still single-request only.

## Metrics and honesty rules

For non-streaming DFlash-mode single-request benchmarks:
- report `gen_tps`, `e2e_latency_s`, `total_throughput`, token counts, and peak memory honestly
- do **not** fabricate TTFT or prompt-processing TPS when the underlying path does not expose first-token timing
- render unavailable metrics as `-` in the UI/text export
- do not silently force `ignore_eos=true` or disable thinking unless the benchmark explicitly says so

## Files

- `omlx/admin/benchmark.py`
  - add single-request mode selection and non-streaming DFlash benchmark path
  - keep batch benchmark behavior unchanged
- `tests/test_benchmark.py`
  - add regression tests for the auto-selection logic and non-streaming single-request metrics
- `omlx/admin/static/js/dashboard.js`
  - render unavailable single-request metrics safely
- `omlx/admin/templates/dashboard/_bench.html`
  - optionally show mode metadata/note if needed by the rendered result set

## Risks

- Non-streaming benchmark results are not directly comparable to streaming TTFT/TPOT metrics.
- Community benchmark upload should not assume all single-request results carry full standard metrics.
- This phase improves correctness of the benchmark surface; it does not yet make streaming/thinking hit the tuned fast path.

## Success criteria

- The built-in throughput benchmark no longer bypasses the tuned DFlash path for compatible single-request runs.
- Standard batch benchmarking still works unchanged.
- The UI renders mixed standard/DFlash benchmark rows without crashes or fake values.
- The change is covered by unit tests.
