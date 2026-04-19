# DFlash mirror robustness batch (2026-04-16)

## Scope
- Honest server-path runs only (`DFLASH_BSTNXBT_EMIT_TELEMETRY=1 ./autoresearch.sh`)
- Authoritative lane: mirror runtime (`mirror_sd_mlx`)
- Window: runs `#2192..#2222` (30 mirror samples + 1 immediate control parity check)
- Code: unchanged baseline `a0f1200`

## Transition parity check
- Fresh mirror collapse observed at run `#2195`: `dflash_eval_s=20.7412`, `completion_tok_s=38.68`
- Immediate control parity run `#2196`: `dflash_eval_s=33.0597`, `completion_tok_s=23.61`
- Result: collapse is shared-path, not mirror-only.

## Mirror batch summary (30 runs)
- `dflash_eval_s`:
  - min: **8.7935**
  - p5: **8.8506**
  - p10: **8.8834**
  - median: **9.4748**
  - p90: **14.2189**
  - p95: **17.8467**
  - max: **29.8828**
  - mean: **11.1884**
- `completion_tok_s`:
  - min: **27.2345**
  - p5: **47.9604**
  - p10: **67.4331**
  - median: **100.5953**
  - p90: **106.2062**
  - max: **107.9462**
  - mean: **90.9308**

## Collapse and recovery behavior
- Collapse runs: `#2195`, `#2197`, `#2198`, `#2199`
- Collapse frequency: **4/30 = 13.3%**
- Severe spike frequency: **4/30 = 13.3%**
- Time-to-recover to `eval_s <= 9.0` after collapse: about **22–26 runs** in this batch.

## Comparison with previous 30-run window (`#2162..#2191`)
- Previous window had **no collapse runs**.
- Previous `dflash_eval_s`: min **8.9215**, median **8.9327**, p90 **9.0359**, max **9.1735**.
- New batch shows wider tails and explicit collapse/recovery cycle, while still returning to near-floor by the end.

## Takeaway
Watchdog containment works (collapse is detected and bounded), but deep shared collapse events still occur and recovery is long. The next practical step is root-cause telemetry on shared runtime pressure (queue/backpressure/memory/allocator/thermal-adjacent proxies), then re-run a fixed batch to verify collapse frequency and tail metrics improve.

---

## Follow-up v2 batch (root telemetry enabled)

### Scope
- Added root telemetry fields in runtime + API + harness (cache-restore/trim cost, split-path hit-rate, MX memory proxies, async-drain/safe-sync timings)
- Window: mirror samples `#2224..#2254` (30 samples) + immediate control parity `#2232`
- Code base for v2: `79733ef` lineage

### Transition parity check
- Severe mirror transition at `#2231`: `dflash_eval_s=51.6814`, `completion_tok_s=15.66`, safe-mode step `2`
- Immediate control parity `#2232`: `dflash_eval_s=53.3561`, `completion_tok_s=15.23`, safe-mode step `2`
- Result: collapse remains shared-path.

### Mirror v2 summary (30 runs)
- `dflash_eval_s`:
  - min: **8.7392**
  - p5: **8.7610**
  - p10: **8.7751**
  - median: **9.2423**
  - p90: **12.8265**
  - p95: **17.1463**
  - max: **51.6814**
  - mean: **11.3019**
- `completion_tok_s`:
  - min: **15.6559**
  - p5: **46.5385**
  - p10: **62.8717**
  - median: **102.3968**
  - p90: **107.9763**
  - max: **108.1589**
  - mean: **94.0201**

### Collapse and recovery behavior (v2)
- Collapse runs: `#2231`, `#2233`, `#2234`, `#2235`
- Collapse frequency: **4/30 = 13.3%** (unchanged vs pre-telemetry batch)
- Severe spike frequency: **4/30 = 13.3%**
- First return to `eval_s <= 9.0` after initial collapse occurred in **~7 runs** (faster than prior batch’s long trough cycle).

### Root telemetry signal readout
- In healthy vs collapsed samples, **split-path hit-rate stayed near ~5%**.
- `mx_peak_over_recommended_ratio` stayed near **~0.175** in both healthy and collapsed runs (no obvious working-set saturation signal).
- Collapse episodes were still dominated by **`dflash_eval_s` / `dflash_draft_s` inflation**, not acceptance collapse.

### Updated takeaway
Root telemetry is now in place and reproducible in honest runs. It improves observability but does **not** yet reduce collapse frequency (still 4/30). The strongest current evidence is that collapse is a shared runtime-state issue not explained by simple split-path activation rate or MX working-set headroom alone.

---

## Follow-up v3 probe (thermal/power sidecar)

### Scope
- Added optional coarse sidecar sampling via `pmset -g therm` (1s cadence): `DFLASH_BSTNXBT_THERMAL_SIDECAR=1`
- Window: runs `#2272..#2279` (mirror + immediate control parity)
- Includes healthy floor samples, fresh mirror collapse transition, immediate control parity collapse, deep trough, and rebound samples.

### Transition parity check
- Mirror transition collapse: `#2275` (`dflash_eval_s=14.0449`, `completion_tok_s=60.77`)
- Immediate control parity: `#2276` (`dflash_eval_s=28.2174`, `completion_tok_s=28.74`)
- Result: collapse remained shared-path in this sidecar-enabled block.

### Sidecar signal readout
- Across runs `#2272..#2279`:
  - `dflash_thermal_sidecar_thermal_warning_samples`: **0**
  - `dflash_thermal_sidecar_performance_warning_samples`: **0**
  - `dflash_thermal_sidecar_cpu_power_status_samples`: **0**
  - `dflash_thermal_sidecar_cpu_speed_limit_samples`: **0**
  - `dflash_thermal_sidecar_gpu_speed_limit_samples`: **0**
  - `dflash_thermal_sidecar_cpu_scheduler_limit_samples`: **0**
- Sidecar collected samples reliably (`samples` non-zero, `failures=0`) but did not expose throttle-warning or speed-limit signals through `pmset -g therm` in either healthy or collapsed phases.

### Updated takeaway (v3)
The thermal sidecar is now wired and validated, but this probe did **not** show thermal/performance warning evidence that tracks collapse onset. Given shared mirror/control collapse persisted while sidecar warnings stayed flat, thermal throttling is currently a weaker explanation than core runtime-state inflation in draft/eval paths.

---

## Follow-up v4 probe (per-step host-time decomposition)

### Scope
- Added per-step decomposition telemetry for draft/verify scheduling and wait behavior:
  - `draft_submit_*`
  - `draft_sync_eval_wait_*`
  - `verify_submit_*`
  - `verify_host_gap_*`
  - `verify_eval_wait_*`
  - `verify_eval_fused_steps`, `verify_eval_unfused_steps`
- Window sampled: runs `#2282..#2289` with sidecar enabled, including healthy mirror, fresh mirror collapse, immediate control parity collapse, rebound samples, and return to healthy fused profile.

### Key sampled observations
- Healthy mirror baseline (`#2282`, reconfirmed again at `#2289`):
  - `verify_eval_wait_mean_s ≈ 0.091–0.092s`
  - fused/unfused = `96/0`
- Collapsed mirror samples (`#2283,#2285,#2286,#2287,#2288` average):
  - `verify_eval_wait_mean_s ≈ 0.161s`
  - fused/unfused average ≈ `53/48`
  - `draft_sync_eval_wait_mean_s ≈ 0.050s` (non-zero once unfused path appears)
- Immediate control parity collapse (`#2284`):
  - `verify_eval_wait_mean_s ≈ 0.432s`
  - fused/unfused = `15/90`
  - `draft_sync_eval_wait_mean_s ≈ 0.124s`

### Host-submit vs wait-path contrast
- Mirror healthy → mirror collapsed:
  - `verify_submit_mean_s`: ~`1.29ms` → ~`1.45ms` (modest)
  - `verify_host_gap_mean_s`: ~`6.4µs` → ~`7.2µs` (tiny)
  - `verify_eval_wait_mean_s`: ~`0.091s` → ~`0.161s` (large)
- Interpretation: sampled collapse inflation is dominated by eval-wait growth and fused→unfused regime shifts, not by large host-gap growth.

### Updated takeaway (v4)
This decomposition weakens the “host scheduling gap” explanation and strengthens a wait-path/regime-shift explanation: collapse aligns with heavy unfused fallback and much higher per-step eval waits (plus non-zero draft sync waits), while host submit/gap telemetry changes only modestly.
