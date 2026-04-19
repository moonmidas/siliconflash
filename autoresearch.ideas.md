# Autoresearch ideas backlog (pruned 2026-04-19)

## Authoritative target (do not drift)
- Honest warmed **official Qwen3.5-9B-bf16** server path.
- **Thinking enabled**, math prompt, `max_tokens=1024`, normal EOS (`ignore_eos=false`).
- DFlash lane: mirror SD + bstnxbt runtime.
- Judge wins only on full `./autoresearch.sh` runs.

## Current kept baseline
- Mirror lane best keep: **108.86 tok/s** (`b304c02`).
- Active winning shape:
  - `DFLASH_MIRROR_SD_RUNTIME=1`
  - `DFLASH_MIRROR_SD_EXIT_LAYER=14`
  - `DFLASH_BLOCK_TOKENS=15`
  - `DFLASH_BSTNXBT_CONTEXT_ONLY_DRAFT_CACHE=1`
  - `DFLASH_DRAFT_SINK=48`
  - `DFLASH_BSTNXBT_ASYNC_DRAFT_EVAL=1`
  - `DFLASH_BSTNXBT_FUSED_DRAFT_VERIFY_EVAL=1`
- DDTree keeps in-tree (still experimental/non-default):
  - `f115bca` (`DFLASH_DDTREE_TOPK_UNNORMALIZED=1`)
  - `66ab83f` (default DDTree budget `13`)
  - `26f9501` (`DFLASH_DDTREE_TOPK_CAP` env knob, default off)

## Qwen3.6 diagnostic lane (non-authoritative)
- Harness honesty fix landed: benchmark payload now always sends an explicit `dflash` boolean, and `autoresearch.sh` forcibly sets `model_settings.dflash_enabled=false` whenever `ENABLE_DFLASH=0`.
- Corrected no-DFlash control band (same explanatory prompt, natural EOS, 1024 tokens) now repeatedly sits around **~89.3–89.9 tok/s** in recovered windows (still watch for occasional abrupt dips).
- Earlier mirror-diagnostic shape (kept for history) clustered around high-86/87:
  - `DFLASH_MIRROR_SD_RUNTIME=1`
  - `DFLASH_BSTNXBT_RUNTIME=1`
  - `DFLASH_BSTNXBT_EXTERNAL_DRAFT=1`
  - `DFLASH_BSTNXBT_RECURRENT_KERNELS=1`
  - `DFLASH_BLOCK_TOKENS=7`
  - `DFLASH_DRAFT_SINK=48`
  - historical keeps: **~86.88, ~87.30, ~87.00, ~87.20 tok/s**.
- Current dominant qwen3.6 mode (new):
  - `DFLASH_MIRROR_SD_RUNTIME=1`
  - `DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD=0`
  - `DFLASH_BSTNXBT_RUNTIME=1`
  - `DFLASH_BSTNXBT_EXTERNAL_DRAFT=1`
  - `DFLASH_BSTNXBT_RECURRENT_KERNELS=1`
  - `DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=0`
  - `DFLASH_BLOCK_TOKENS=10`
  - `DFLASH_DRAFT_SINK=48`
  - repeated keeps (throughput-only): **~185.3–185.8 tok/s**, with matched explicit controls around **~89.1–89.5 tok/s**.
  - **Quality gate pending**: recent spot-checks show degenerate outputs at this top-speed shape; do not promote as a final winner until coherence is restored.
- Context-length characterization (length-anchored synthetic prompt, 1024 completion tokens):
  - ~980 prompt tokens: DFlash **~174.3–176.2 tok/s** vs explicit no-DFlash control **~80.3 tok/s**.
  - ~1908 prompt tokens: DFlash **~120.4 tok/s** vs explicit no-DFlash control **~76.9 tok/s**.
  - ~3793 prompt tokens: DFlash **~123.3–124.3 tok/s** vs explicit no-DFlash control **~70.1 tok/s**.
  - ~7633 prompt tokens: DFlash **~88.0–89.8 tok/s** vs explicit no-DFlash control **~60.5 tok/s**.
  - DFlash-over-control separation remains positive across tested context buckets (**+~27 to +~96 tok/s**), while absolute DFlash tok/s declines as prompt context grows.
- New telemetry acceptance-vs-context sweep (synthetic alpha-context prompts, 1024 completions) mapped two distinct regimes:
  - `SPECULATIVE_LINEAR_CACHE=0` fast lane: acceptance stayed relatively high/non-monotonic (**0.679 → 0.662 → 0.664 → 0.628 → 0.670** across ~45/~1k/~2k/~4k/~8k prompt tokens) while tok/s fell (**181.3 → 159.7 → 141.7 → 110.8 → 80.2**) and prefill grew sharply (**0.09s → 0.65s → 1.25s → 2.58s → 5.69s**).
  - `SPECULATIVE_LINEAR_CACHE=1` reference (`IGNORE_EOS=1` for fixed-length comparability): acceptance was much lower and collapsed hard past ~1k (**0.359 → 0.355 → 0.0606 → 0.0607 → 0.0859**) with tok/s (**55.7 → 64.2 → 20.0 → 20.4 → 21.1**) and mean commit depth shrinking to ~1.6 in the collapse pocket.
  - Working interpretation: in the fast lane, >1k slowdown appears dominated by context-scaling compute (especially prefill) plus moderate acceptance drift; the severe acceptance cliff is currently most pronounced in the speculative-linear-cache reference lane on this prompt family.
- Quality warning (short-prompt lane): recent DFlash spot-checks reproduced a degenerate preview pattern (`# Speculative Decoding: A high-performance.` followed by repetitive `.` scaffolding) while matched explicit controls remained coherent prose. The pattern persists with both mirror wrapper on (`mirror_sd_mlx`) and off (`bstnxbt_mlx`), so treat this as a shared fast-path correctness blocker until root-caused.
- Fixed quality eval suite is now wired into `./autoresearch.sh` (`scripts/siliconflash_quality_eval.py`, metrics `quality_eval_*`). In same-window comparisons, control repeatedly scores clean (`pass_rate=1.0`, `failed_cases=0`, `degenerate_cases=0`) while fast DFlash configs in `SPECULATIVE_LINEAR_CACHE=0` remain around `pass_rate=0.5` with persistent failures (`arith_addition` wrong value/partial value + degenerate coherence probe), even across recurrent-kernel, packed-QKV, and mirror-exit-layer retunes.
- Emerging quality/perf boundary in current fast path: `DFLASH_BLOCK_TOKENS=7/8` produced coherent previews at ~147 tok/s, while `9/10` yielded higher throughput (~174 to ~183+) but degraded into repetitive-token loops; prioritize fixing this boundary over further max-speed tuning.
- Mitigation evolution: initial point `BLOCK_TOKENS=10` + `VERIFY_LEN=8` reached ~164.6–165.9 tok/s; raising to `BLOCK_TOKENS=11` with the same cap recovered much more throughput, and sink retunes (`SINK=40/32`) lifted that shelf to ~178.6–179.1 tok/s. New warmup-limited verify controls then improved the speed/quality compromise further: `VERIFY_LEN=9` + `WARMUP_STEPS=8` + `WARMUP_CAP=2` reached ~181.9 with improved eval profile (`pass_rate=0.75`, `degenerate_cases=0`), and sink sweeps in this sub-regime pushed throughput to ~183.8 (`SINK=56`) and ~184.3 (`SINK=64`) while keeping the same 0.75 pass profile.
- Under this mitigation family, `BLOCK_TOKENS=9` remained slower (~162.1), and `BLOCK_TOKENS=12` regressed sharply (~147.1), so keep block=11 as the current tuned anchor.
- Verify-cap boundary at `BLOCK_TOKENS=11`: uncapped `VERIFY_LEN=9` previously reintroduced repetitive-token loops (~173.7), while `VERIFY_LEN=8` held the strongest pre-warmup mitigated shelf (~178+), `7` was slower (~155.0), and `1/2` restored full suite pass but collapsed throughput (~42.0/~66.9). `3/4` remained non-winning/pathological in prior mapping.
- New keep (2026-04-19): added warmup-limited verify support in bstnxbt runtime (`DFLASH_VERIFY_LEN_WARMUP_STEPS`, `DFLASH_VERIFY_LEN_WARMUP_CAP`). With `BLOCK=11`, `VERIFY_LEN=9`, `WARMUP_STEPS=8`, `WARMUP_CAP=2`, short-lane throughput recovered to ~181.9 tok/s while quality improved versus uncapped fast path (`quality_eval_pass_rate=0.75`, `degenerate_cases=0`; arithmetic fixed, coherence probe remains the only failing case).
- New code-level keeps (2026-04-19):
  - `RecurrentRollbackCache` now defaults to reference snapshots instead of per-step array copies (`DFLASH_BSTNXBT_ROLLBACK_SNAPSHOT_COPY=1` restores old behavior).
  - Recurrent conv-state rebuild now uses a direct trailing-qkv slice fast path when `accepted_steps >= conv_kernel_size-1`, avoiding concat work in common rollback cases.
  - Added Qwen3Next target projection packing gate in oMLX bstnxbt runtime (`DFLASH_BSTNXBT_PACK_TARGET=1`, `DFLASH_BSTNXBT_PACK_ATTENTION=1`): packed QKV path is integrated into hooked full-attention forward and works even when split-SDPA is off.
  - Keep status for packed-QKV is tentative/noise-limited (first keep ~86.00), and needs re-validation once the lane exits collapse windows.
  - Ported verify-path commit slicing from `dflash-mlx`: compute acceptance before hidden eval, slice `committed_hidden` to `commit_count` pre-eval, and reuse that slice directly for next-step `target_hidden` (keeps at ~86.48/~86.08/~86.24; same-window controls ~85.77/~85.98/~85.35, so edge is positive-but-noise-limited).
  - Replicated upstream `dflash-mlx` hook layering for packed attention: installed dedicated packed-attention call hook and made split-SDPA hook early-return to original call when split is off (keeps at ~86.52/~86.40; first paired control ~85.50, +~1.02 tok/s before next global collapse).
  - Ported upstream verify-block controls: added `DFLASH_VERIFY_LEN` cap support and optional `DFLASH_BSTNXBT_VERIFY_CHUNK_TOKENS` chunked verify plumbing in bstnxbt runtime (default run ~86.05); mirror-mode safety hardening auto-disables verify chunking outside default target-forward mode.
  - Rollback hardening for potential chunked verify path: `RecurrentRollbackCache.record_tape()` now accumulates per-chunk tape/k/g/qkv slices while armed (instead of overwriting) so rollback can replay full verified span.
  - Ported upstream cache policy for chunked verify into oMLX (`make_target_cache(enable_speculative_linear_cache=...)`) and added explicit override knob `DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE` (auto/0/1). In this qwen3.6 lane, forcing non-speculative linear cache (`=0`, or prior `VERIFY_CHUNK_TOKENS>=block`) repeatedly unlocked a new high-throughput shelf (~142.5–148.9 tok/s) with large same-window separation vs explicit no-DFlash controls (~89.3–89.7), but recent short-prompt quality checks indicate a potential output-degeneration regression tied to this fast path that must be resolved before promotion.
  - Added a major routing step in mirror runtime: env-gated switch to shared default target-forward path (`DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD=1`) instead of always using mirror override; DFlash samples can run high (~86.96/~87.94) but same-window explicit controls have ranged from ~86.04 up to ~89.92, so throughput edge is currently inconclusive/noise-limited.
  - Added adaptive block-token controller scaffolding in bstnxbt runtime (`DFLASH_BSTNXBT_ADAPTIVE_BLOCK_TOKENS`) with warmup/window/cooldown and bidirectional shrink/grow logic, plus reusable per-step block token buffer allocation; default-off path can still hit high-87, but first adaptive-on probe (`min=6,max=7`) regressed (~83.36).
  - Routing variant check: disabling mirror runtime wrapper (`DFLASH_MIRROR_SD_RUNTIME=0`) while keeping bstnxbt active can hit high-88 (~87.77/~88.40), but immediate explicit controls in the same windows were higher (~89.65/~89.92), so this is currently parity/control-favored.
  - Non-stream execute micro-optimizations landed in runtime: avoid per-commit full `output_token_ids` copies when no callback (`copy_output_token_ids=False`), add a no-yield execute mode (`emit_commits=False` + StopIteration final tokens), and add execute-mode device output buffering (commit tokens staged on MX and converted to Python once at end). DFlash stayed in high-87 (~87.64/~87.78) but same-window controls still ran faster (~89.50/~89.42).
- Natural-EOS control still intermittently emits early-stop 17-token samples; these are non-comparable and must be discarded (do not update baseline from them).
- Fixed-length forcing (`IGNORE_EOS=1`) and alternate “robust prompt” parity attempts are closed as non-representative (pathological tails around ~25 tok/s and ~16 tok/s).
- Historical note (older mirror/block-7 lane): nearby sink sweeps `DFLASH_DRAFT_SINK=40` and `56` were catastrophic (~45–51 tok/s). This does **not** apply to the current `BLOCK=11, VERIFY_LEN=8` mitigation lane, where `SINK=32/40` are currently strongest.
- DDTree status unchanged: non-Mirror `ddtree_mlx` remains non-viable, and Mirror+DDTree remains below corrected no-DFlash control in matched windows.
- Stability note (2026-04-18 → 2026-04-19): corrected-harness lane remains bursty with severe collapse waves (as low as ~30–36 tok/s plus one ~10.86 outlier), partial recoveries into high-86/87 pockets, and recurrent re-collapses (recent mirror ~27.30/~46.43). After the latest collapse, controls rebounded from ~51.95 through ~60/~66/~71/~76/~79/~81 and then plateaued around ~82–85.4 without clearing the >=85.5 re-entry gate; the lane then hit another severe collapse pocket (DFlash ~26 → ~42 → ~54 → ~59 and explicit no-DFlash control ~42.9 in `#3037–#3053`) before rebounding again toward mid-80s (`#3058–#3070`) with persistent sub-baseline noise-scale movement (~84.7–85.1, plus one block-size retune regression). Same-window parity checks then showed mixed recovered behavior: DFlash reached ~86.01/~86.19/~86.42 while immediate explicit controls ranged ~84.55 to ~86.13 (`#3091–#3096`), so parity flipped between ~+1.46 DFlash edge and near-zero (~+0.06) in adjacent samples. The lane then re-collapsed (`#3097–#3098`: DFlash ~81.59, explicit control ~66.14), followed by control-led recovery (`#3099–#3106`: ~73.5 → ~80.8 → ~82.6 → ~83.4 → ~84.4 → ~85.2) while adjacent DFlash probes stayed much lower (~77.44 and telemetry ~78.75), producing large temporary control-over-DFlash gaps (~+7.00 and ~+6.49). Post-slicing/hook-layering/verify-port windows (`#3109–#3195`) mostly stayed in high-87/88 with controls often higher. Major regime shift then landed in `#3196–#3236`: non-speculative linear cache mode (`DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=0`) repeatedly reached ~142.5–185.8 tok/s while explicit controls stayed ~89.0–89.7 (+~53 to +~96). Block-size sweeps improved strongly through 8→9→10, while 11 and 12 regressed; sink remained near-parity across 40/48/56; and mirror default-forward routing (`DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD` 0/1) is currently low-signal near-parity in this regime (historical edge for `=0`, but recent short-lane retest was ~182.88 with `=1` vs ~183.21 baseline). This is the first robust same-window DFlash separation in this lane, but still needs correctness/quality validation before broad promotion.

## Current regime snapshot
- Regime is historically bursty; recent windows alternate between near-floor control and abrupt downshifts.
- In the last healthy pocket controls repeatedly returned to ~108.6–108.8 (eval ~8.71–8.73) while DDTree stayed ~103.0–103.6, but the segment then re-collapsed: DDTree baseline dropped to ~63.9 (eval ~16.0, tree_build ~3.24) and control fell to ~68.7 then ~77.0 then ~85.3 with elevated fused verify waits (~0.147 → ~0.131 → ~0.117s), indicating a global instability wave rather than DDTree-local effects. The window later recovered to ~108.4–108.7 (DDTree still ~103.4), but later DDTree experiments (aggressive controller and then always-split exact-KV threshold) re-triggered additional collapse waves in control (e.g., ~66/76/70/89/62 and later ~79/59/63/74/77), with recurring `safe_mode=1` or slow fused-only shelves. Latest monitors have re-established a healthy pocket again (multiple controls ~108.58–108.83, eval ~8.71–8.73, fused-only `safe_mode=0`), while DDTree continues around ~103.3–103.5 in the same window.
- Interpretation: DDTree remains roughly **~5 tok/s** behind control in healthy windows, but current attribution is fragile because global collapse/recovery waves can invalidate A/B conclusions for long stretches.
- Synchronized DDTree verify profiling (`DDTREE_PROFILE_VERIFY=1`) repeatedly shows verify dominating DDTree cost in this lane (earlier `tree_verify_s` ~7.72s with ~5.81s linear / ~1.90s attention; later ~8.00s with ~6.02s linear / ~1.97s attention). Detail mode (`DDTREE_PROFILE_VERIFY=detail`) still inflates absolute times (latest ~12.02s total, ~8.33s linear, ~2.78s attention; prior ~12.20/~8.45/~2.82) but preserves the same dominance pattern, so pure tree-build tuning is unlikely to close the full gap.

## Stability gate before any DDTree promotion
Require all of the following before trusting DDTree A/B conclusions:
1. Control lane fused-only and `safe_mode=0`.
2. `dflash_eval_s` repeatedly near **~8.8–9.0** (not just one spike sample).
3. Immediate parity control run after DDTree candidate.

If gate is not met: monitor only; do not promote DDTree changes.

## Active queue (highest value)
1. Validate correctness/quality of the fast path with the fixed eval suite: warmup-limited verify removed one failure class (arith fixed; no degenerate cases) but one coherence case still fails (`pass_rate=0.75` vs control `1.0`).
2. Treat `BLOCK_TOKENS=11` + `DFLASH_VERIFY_LEN=8` + `DFLASH_DRAFT_SINK=32` as the current throughput anchor in the mitigated family (~179.1 tok/s), but do **not** treat it as quality-safe until eval pass rate improves.
3. Prioritize code-level root-cause work in `SPECULATIVE_LINEAR_CACHE=0` path (acceptance/commit/cache semantics), especially the remaining coherence-probe failure after warmup-verify guard (`VERIFY_LEN=9`, warmup 8/2). New per-case diagnostics show this failure is a low-token-count malformed continuation (`token_count~16`, long single-token span) with moderate acceptance (`~0.55`), not a classic repeat-loop signature.
4. Keep strict attribution triplets for every change: **DFlash candidate → explicit no-DFlash control → DFlash repeat**.
5. Keep telemetry mostly off for speed comparisons; use targeted telemetry snapshots only when diagnosing regressions/safe-mode behavior.
6. Keep DDTree work deferred during unstable windows; only resume DDTree triplets after control remains stable in matched windows and the correctness gate is satisfied.

## Promising but deferred (only after floor gate)
- Warmup-verify follow-up: periodic verify-resync pulses are still speculative; first probe (`RESYNC_INTERVAL=16`, `RESYNC_CAP=2`) on top of `VERIFY_LEN=9` + warmup 8/2 was non-winning (~177.0, no quality gain). Repeat-token rescue variant (`REPEAT_RUN_TRIGGER=12`, `REPEAT_RUN_STEPS=8`, `REPEAT_RUN_CAP=2`) was catastrophic (~76.3, no quality gain). If revisited, require telemetry-guided activation-rate checks and much looser/safer parameters.
- Verify-path reductions that preserve correctness (especially linear/attention work inside tree verify while keeping kernels enabled).
- Tree-build top-k alternatives remain secondary and must clear >noise-level gains.
- Narrow sync-placement trims around DDTree verify/commit that preserve correctness.
- Lightweight allocation reuse in DDTree hot path (only if verified no correctness drift).
- Upstream-focused verify-kernel work: reduce `tree_verify_linear` dominance (latest ~8.30s) by caching parent/index tensors and/or fusing linear projection + recurrence setup inside `ddtree_mlx.verify` kernel path (likely requires changes in `/tmp/ddtree-mlx`, not just oMLX wrapper).
- Qwen3.6 parity audit track: reproduce reference benchmark conditions (prompt mix + longer generations like 2k/8k) and compare oMLX runtime phase timings against `humanrouter/ddtree-mlx` to isolate where non-Mirror path diverges.
- Adaptive block-token controller for bstnxbt runtime (acceptance-streak shrink/grow) is now implemented but unproven: first qwen3.6 trial with `min=6,max=7` regressed (~83.36); only revisit with telemetry-guided threshold tuning and strict matched controls.
- Routing follow-up (mirror-default-forward vs non-mirror bstnxbt): treat as structural simplification/stability work; neither path has shown durable throughput edge against explicit controls yet.
- Naive adaptive-shrink-only controller (`DFLASH_BSTNXBT_ADAPTIVE_SHRINK=1`) produced a catastrophic sample (~18.8 tok/s) during a concurrent collapse wave (post-revert still ~41.4), so treat this variant as high-risk/inconclusive and do not retune in unstable windows.

## Do-not-retry / stale paths

### Mirror lane
- `DFLASH_MIRROR_SD_EXIT_LAYER != 14` (broadly non-winning; several severe cliffs).
- Qwen3.6 note: unsetting `DFLASH_MIRROR_SD_EXIT_LAYER` during the latest collapse wave produced a catastrophic ~11 tok/s sample, but immediate exit-layer-14 rerun also collapsed (~46), so attribution is currently inconclusive; only retest exit-layer variants in a healthy control window.
- `DFLASH_BLOCK_TOKENS` retunes away from `15` for 1024 thinking lane (non-winning; high values can cliff badly).
- `DFLASH_BSTNXBT_CONTEXT_ONLY_DRAFT_CACHE=0`.
- `DFLASH_DRAFT_SINK` retunes away from `48` in this lane (notably `32` severe cliff).
- **Qwen3.6 diagnostic**: for mirror-only block-7 path, sink sweeps `{40,56}` are catastrophic (~45–51 tok/s); keep `DFLASH_DRAFT_SINK=48` fixed.
- **Qwen3.6 diagnostic**: mirror-era block retune `DFLASH_BLOCK_TOKENS=8` (with sink 48) was catastrophic (~47.9 tok/s) under old speculative rollback path; this no longer applies to the new non-spec linear cache mode, where block sizes 8→10 improved and block=10 is currently best (~184.9).
- **Qwen3.6 diagnostic**: block-size behavior depends on verify-cap regime. In uncapped fast mode, quality breaks at higher blocks; in capped mitigation (`VERIFY_LEN=8`), `BLOCK_TOKENS=11` outperforms 9/10/12 for throughput.
- **Qwen3.6 diagnostic**: sink behavior also depends on regime. In the capped mitigation lane (`BLOCK=11, VERIFY_LEN=8`), `SINK=32/40` outperformed `48/56` in recent checks.
- **Qwen3.6 diagnostic**: with `SPECULATIVE_LINEAR_CACHE=0`, toggling `DFLASH_BSTNXBT_RECURRENT_KERNELS` remains low-impact for throughput and did not change quality-eval drift profile (still `pass_rate=0.5`, `degenerate_cases=1` in tuned lane); treat this toggle as non-root-cause.
- **Qwen3.6 diagnostic**: rollback snapshot copy fallback (`DFLASH_BSTNXBT_ROLLBACK_SNAPSHOT_COPY=1`) was parity/noise in the short-prompt high-throughput lane (~182.60 vs ~183.21 baseline); keep default reference-snapshot behavior.
- **Qwen3.6 diagnostic**: `DFLASH_BSTNXBT_CONTEXT_ONLY_DRAFT_CACHE` toggle is currently low-signal/stale under `DFLASH_BSTNXBT_EXTERNAL_DRAFT=1` because the external draft loader already installs `ContextOnlyDraftKVCache`; do not spend more tuning budget on this toggle in external-draft mode.
- **Qwen3.6 diagnostic**: `DFLASH_BSTNXBT_ASYNC_DRAFT_EVAL=1` + `DFLASH_BSTNXBT_FUSED_DRAFT_VERIFY_EVAL=1` regressed (~76.1 tok/s); keep both off.
- **Qwen3.6 diagnostic**: legacy path `DFLASH_BSTNXBT_SLICE_COMMITTED_HIDDEN_BEFORE_EVAL=0` remains non-winning (recently ~177.44 vs ~183.21 baseline) and did not resolve the current degenerate-output pattern; keep default slicing enabled.
- **Qwen3.6 diagnostic**: verify-length controls are highly non-monotonic. Recent short-lane checks at block=11 show: plain `VERIFY_LEN=8` ~178 with failing eval; plain `9` quality-unstable; `1/2` quality-clean but very slow (~42/~67); `3/4` non-winning/pathological. Warmup-limited verify is now the strongest compromise: `VERIFY_LEN=9` + `WARMUP_STEPS=8` + `WARMUP_CAP=2` reached ~181.9 with improved suite profile (`pass_rate=0.75`, `degenerate_cases=0`) but still one coherence failure. Nearby warmup retunes (`steps=4/10/16`, `cap=1`) were non-winning for quality stability, and loosening to `cap=3` regressed quality back to `pass_rate=0.5`.
- **Qwen3.6 diagnostic**: with warmup-guarded verify9, lowering block to 10/9 was non-winning (~174.6/~169.2, no quality gain); keep block=11 in this sub-regime.
- **Qwen3.6 diagnostic**: in the warmup-guarded verify9 sub-regime, sink retunes trended upward (`SINK=40` -> ~182.9, `56` -> ~183.8, `64` -> ~184.3) without changing the remaining quality failure class.
- **Qwen3.6 diagnostic**: earlier verify-chunk crashes were tied to speculative linear cache policy; with `DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=0` the path is now stable and very fast. Do not pair verify chunking with speculative linear cache enabled (`DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=1`) in this lane.
- **Qwen3.6 diagnostic**: forcing `DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE=1` is massively regressive on throughput (recently ~69.49 tok/s in short lane), but serves as a useful correctness reference because preview output remained coherent versus the degenerate pattern seen in `=0` fast path.
- **Qwen3.6 diagnostic**: completed fixed-length context sweep for `SPECULATIVE_LINEAR_CACHE=1` (`IGNORE_EOS=1`) shows severe acceptance collapse beyond ~1k on the synthetic alpha-context workload (`acceptance ~0.355 @1k -> ~0.061 @2k/4k; mean_commit ~1.6`), with throughput collapsing into ~20 tok/s. Treat this as a mapped reference behavior (not an optimization direction) and avoid repeating this broad sweep unless workload assumptions change.
- **Qwen3.6 diagnostic**: `DFLASH_BSTNXBT_SPLIT_SDPA=1` was non-winning in current packed-QKV lane (~84.89), and lowering split activation threshold (`DFLASH_BSTNXBT_SPLIT_EXACT_KV_THRESHOLD=256`) remained non-winning (~84.92).
- **Qwen3.6 diagnostic**: MoE SwitchGLU packing probe (`DFLASH_BSTNXBT_PACK_SWITCH_MLP=1`) regressed (~84.7 tok/s in matched packed-QKV window); close this path until a cleaner implementation is available.
- **Qwen3.6 diagnostic**: `DFLASH_BSTNXBT_PACK_MLP=1` (shared-expert/Qwen3NextMLP packing) was non-winning (~85.03 vs nearby packed-QKV baseline).
- **Qwen3.6 diagnostic**: non-stream output-id copy-elision path (`DFLASH_BSTNXBT_NO_COPY_OUTPUT_IDS=1`) was non-winning (~84.83).
- **Qwen3.6 diagnostic**: capture-state list micro-optimization (replace dict capture with ordered list capture in mirror/default forward paths) was non-winning (~84.95).
- **Qwen3.6 diagnostic**: block-size retune `DFLASH_BLOCK_TOKENS=6` regressed sharply (~80.74); keep block=7 fixed.
- **Qwen3.6 diagnostic**: newly added adaptive block controller is currently non-winning with tested conservative bounds (`DFLASH_BSTNXBT_ADAPTIVE_BLOCK_TOKENS=1`, min=6 max=7 -> ~83.36); keep adaptive mode off by default.
- **Qwen3.6 diagnostic**: execute-path output-id copy-elision / no-yield / device-buffer optimizations in non-stream mode improved DFlash absolute speed into high-87, but did not establish DFlash edge in matched windows (~87.64–87.78 DFlash vs ~89.42–89.50 controls); treat as structural cleanup, not a throughput winner.
- **Qwen3.6 diagnostic**: exact-small-proj pad sweeps via `DFLASH_BSTNXBT_EXACT_SMALL_PROJ_PAD_M` (`8`, `12` vs default `16`) were parity/noise-level (~87.54–87.70) and non-promotable.
- **Qwen3.6 diagnostic**: mirror runtime wrapper toggle (`DFLASH_MIRROR_SD_RUNTIME` 0 vs 1) remains near-parity/low-signal across both throughput-max and quality-safe regimes (e.g., ~177.30 with `0` vs ~178.27–178.34 with `1` in `BLOCK=11, VERIFY_LEN=8` lane); do not spend tuning budget on this toggle unless behavior diverges again.
- **Qwen3.6 diagnostic**: mirror exit-layer retune (`DFLASH_MIRROR_SD_EXIT_LAYER=20`) raised throughput slightly (~179.39) but did not change quality-eval failures (`pass_rate=0.5`), so exit-layer sweeps are not currently solving drift.
- **Qwen3.6 diagnostic**: mirror default-forward sub-toggle (`DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD` 0 vs 1) remains near-parity/noise across regimes (e.g., quality-safe lane ~178.33 with `=1` vs ~178.34 with `=0`); keep `=0` as default and stop spending budget on this toggle.
- **Qwen3.6 diagnostic**: packed-QKV toggle-off probe (`DFLASH_BSTNXBT_PACK_TARGET=0`, `DFLASH_BSTNXBT_PACK_ATTENTION=0`) remains parity/noise in throughput and did not improve quality-eval drift (still `pass_rate=0.5`, `degenerate_cases=1`), so it is non-promotable and non-root-cause.
- Disable core wins: `DFLASH_BSTNXBT_SPLIT_SDPA=0`, `DFLASH_BSTNXBT_RECURRENT_KERNELS=0`; also keep `DFLASH_BSTNXBT_EXTERNAL_DRAFT=1` (recent `=0` isolation was catastrophic at ~32.7 tok/s and did not resolve degraded-output symptoms).
- Control-lane `DFLASH_BSTNXBT_SPLIT_CHUNK_SIZE` retunes (`12`, `16`, `24`, `32`) were near-best but non-winning (e.g., `12`/`32` ~108.72, `16` ~108.80/~108.74, `24` ~108.68) vs immediate default parity in the same windows.
- `DFLASH_USE_MLX_NATIVE_DRAFTER=0` (catastrophic).
- Ungated shared bstnxbt GQA head-alignment patch (forcing `g`/`beta`/tape-`k` head expansion to `Hv` in core runtime+kernels) regressed authoritative control in a direct check (~105.3 tok/s vs healthy ~108+ band); only revisit as a strict model-gated Qwen3.6 fix path.

### DDTree lane
- **Qwen3.6 diagnostic**: pure `ddtree_mlx` path without Mirror flags is currently non-viable (severe collapses at both 64 and 1024 tokens); do not spend more tuning budget here until runtime-path mismatch is identified.
- **Qwen3.6 diagnostic**: non-Mirror verify-bind tuning is stale (`DFLASH_DDTREE_NATIVE_VERIFY_BIND_MODE=none` still collapsed; acceptance ~1 token/cycle and tree_build ballooned).
- **Qwen3.6 diagnostic**: forcing `DDTREE_EXACT_COMMIT=1` with tree-kernel flags (`DDTREE_TREE_AWARE_LINEAR=1`, `DDTREE_TREE_KERNEL=1`, `DDTREE_TREE_CONV_KERNEL=1`) is stale/non-winning on this integration (~15.5 tok/s @1024).
- **Qwen3.6 diagnostic**: `DFLASH_DRAFT_SINK=64` + `DFLASH_DRAFT_WINDOW=1024` did not materially improve Mirror+DDTree budget-4 in oMLX (~80.0 tok/s @1024); low-priority for retests.
- **Qwen3.6 diagnostic**: budget sweeps above 8 are currently stale/non-winning in oMLX (budget=10/13 regressed sharply; budget=8 performed best among tested values).
- **Qwen3.6 diagnostic**: `DFLASH_DDTREE_TOPK_SKIP_SORT=1` at budget=8 gave only a modest uplift (~+0.86 tok/s telemetry run) and remains non-promotable.
- **Qwen3.6 diagnostic**: `DFLASH_DDTREE_TOPK_CAP` retune (`cap=6` with budget=8) was strongly regressive (~60 tok/s @1024); close cap-family sweeps.
- **Qwen3.6 diagnostic**: reducing `DFLASH_BLOCK_TOKENS` to 8 with budget-8 skip-sort regressed (~73.3 tok/s @1024) and lowered fast-path quality; keep default block sizing.
- `DDTREE_DFLASH_CONTROLLER=1` is non-winning; aggressive controller tuning (`WARMUP=0`, `INTERVAL=1`, `MARGIN=1.00`, `MIN_PROBES=1`) was catastrophic (~31.6 tok/s, switched almost entirely to dflash cycles with acceptance collapse).
- DDTree budget retunes `{8,9,10,11,12,14,15}` as lane default (non-winning or severe cliffs vs 13; 8/9/10 are catastrophic in this lane).
- `DFLASH_DDTREE_TOPK_SKIP_SORT=1`.
- `DFLASH_DDTREE_TOPK_CAP in {8,9,10,11,12}` on budget-13 path.
- `DFLASH_DDTREE_DEPTH_CAP=11` (severe regression: ~93.5 tok/s, lower acceptance, higher tree-build time).
- `DFLASH_DDTREE_TOPK_UNNORMALIZED=0`.
- `DDTREE_TREE_AWARE_LINEAR=0` (catastrophic).
- `DDTREE_EXACT_COMMIT=1` (catastrophic commit overhead).
- `DDTREE_TREE_KERNEL=0` (severe regression; kernel path is active and critical).
- `DDTREE_TREE_CONV_KERNEL=0` (severe regression; parent-aware conv kernel is active and critical).
- `DDTREE_EXACT_TREE_ATTENTION=1` (severe regression in this 1024-token lane: ~80.8 tok/s, lower acceptance, higher tree-build time).
- `DDTREE_EXACT_TREE_ATTENTION=auto` with practical thresholds (`MIN_PREFIX=512` or `1024`) was non-winning/regressive vs matched DDTree baselines.
- DDTree block retunes `{11,12,14,16}` under budget-13 unnormalized path were non-winning/flat; explicit `DFLASH_BLOCK_TOKENS=13` probe was effectively flat/no-op (native path already behaves like block 13).
- Prior per-cycle top-k config/cast micro-trim refactor in `ddtree_runtime.py` (catastrophic sample).
- Tree-aware-linear bookkeeping micro-trims (including lazy DFS materialization/comparison changes) repeatedly regressed in matched DDTree baselines (~103.0–103.6 → ~102.4–102.5 tok/s).
- Tree-aware fast/slow classification skip in default `exact_commit=0` mode plus another redundant-cast guard was effectively noise-level (~103.37 → ~103.39 tok/s) and non-promotable.
- `DFLASH_DDTREE_NATIVE_DISABLE_VERIFY_DETAIL_HOOKS=1` (short-circuit verify detail-hook helper calls) yielded only a small/noise-level uplift (~103.32 → ~103.46) and is non-promotable.
- Tree-aware commit hidden-slicing micro-trim (avoid full-tree hidden concat before accepted-path gather) was neutral/noise-level (~103.44 → ~103.44) and non-promotable.
- Identity-prefix commit hidden slice (skip gather allocation when accepted indices are prefix-contiguous) regressed slightly in matched healthy window (~103.31 → ~103.19); non-winning.
- Scoped tree-aware SSM-mask skip patch (monkey-patched `create_ssm_mask` off only during tree-aware verify) was flat/slightly worse (~103.35 → ~103.32); non-winning.
- Budget-depth trim in `_build_tree_from_mlx_logits` (`logits[:budget]`) produced only noise-level movement (~102.98 → ~103.10 tok/s, tree_build ~1.938s → ~1.931s); low-priority/non-robust.
- Redundant float32-cast guard micro-trim in `_build_tree_from_mlx_logits` regressed slightly (~103.37 → ~103.25 tok/s) despite tiny tree_build drop; non-winning.
- Positive-tail `argpartition` top-k selection variant (avoiding negation) was effectively flat/noise-level (~103.37 → ~103.39 tok/s); non-robust.
- Draft-dtype top-k selection variant (skip full-logits float32 cast in unnormalized mode) was also noise-level (~103.20 → ~103.25 tok/s) despite a tiny tree_build drop; non-winning.
- Removing explicit `mx.eval(top_token_ids, top_scores)` before NumPy conversion regressed (~103.31 → ~103.11 tok/s) and increased tree_build time; non-winning.
- `DFLASH_DDTREE_NATIVE_VERIFY_BIND_MODE` sweeps (`all`, `model_only`, `none`) were effectively noise-level (~102.5–103.4 tok/s) with no robust uplift; non-winning.
- Current control floor in this segment repeatedly revisits ~108.0–108.8 tok/s (`dflash_eval_s` ~8.72–8.79), while DDTree remains in ~101–103 tok/s band; gap remains structural in matched healthy windows.
- `DFLASH_DDTREE_NATIVE_SPLIT_SDPA=1` is non-winning across tested chunk sizes in this lane: default (`8`) regressed (~103.3 → ~102.2), chunk `4` and `32` were non-winning/flat, and chunk `16` (alone or with `DFLASH_DDTREE_NATIVE_VERIFY_BIND_MODE=none`) failed robust recheck (~103.34 baseline → ~103.05).
- Tree-verify exact-KV threshold overrides are now closed as low-value/non-promotable: high-threshold no-split runs (`DFLASH_DDTREE_NATIVE_EXACT_KV_THRESHOLD` at `262144` and rerun at `65536`) repeatedly gave only small uplifts (~+0.24 to ~+0.38 tok/s vs ~103.3 baseline), while low threshold/always-split (`=0`) was catastrophic (~37.6 tok/s, eval/tree-build blow-up).
- DDTree `DFLASH_VERIFY_LEN` caps are non-winning in this lane: `12` and `10` were flat/slightly worse (~103.42 baseline → ~103.28/~103.35) and `8` regressed (~103.42 → ~102.64).
- `DFLASH_DDTREE_NATIVE_USE_EXTERNAL_CACHE=1` regressed against matched DDTree baseline (~102.37 → ~102.09 tok/s) with higher verify time; non-winning.
- `DFLASH_DDTREE_NATIVE_INSTALL_HOOKS=1` was effectively noise-level (~101.65 → ~101.74 tok/s) and did not reduce verify cost; non-winning.
- Recent DDTree baselines in the recovered control pocket drifted down to ~100.98–102.66 (vs control ~106.7–108.8), reinforcing DDTree volatility and lane-local gap persistence.
- DDTree runs without `DFLASH_DDTREE_PATH=/tmp/ddtree-mlx` in this workspace (falls back/unavailable; invalid for DDTree attribution).

### Collapse mitigation probes already closed
- Aggressive always-on collapse-threshold packs (catastrophic in transition windows).
- `DFLASH_BSTNXBT_CLEAR_CACHE_EVERY_STEPS=8` (non-winning).
- Async backlog as primary cause (unsupported by telemetry).
- Thermal throttling as primary cause (unsupported by sidecar telemetry on sampled waves).
- Recent low DFlash telemetry sample (`#3105`) had `safe_mode=0` and zero collapse-spike events, so watchdog/safe-mode toggles are not a sufficient explanation for current DFlash-only dips.
- Single-sample control rebounds in the mid-85 range are not sufficient to reopen mirror parity (latest streak `#3024–#3032` stayed below strict >=85.5 x2 gate).
- Packed-target toggle is not a sufficient collapse explanation by itself: disabling `DFLASH_BSTNXBT_PACK_TARGET` during the latest collapse remained in the same low-59 shelf (`#3042`).
- Catastrophic adaptive-shrink trial (~18.8) was immediately followed by a post-revert ~41.4 run (`#3048`), reinforcing that this window was a regime crash and not clean A/B attribution territory.
- Explicit no-DFlash control sample also collapsed (~42.9, `#3053`), confirming the latest slowdown wave is global and not DFlash-only.

## Guardrails
- Do not overfit to collapse/recovery windows.
- No benchmark cheating: keep thinking enabled and normal EOS semantics on the authoritative lane.
- Use telemetry for attribution (`dflash_eval_s`, fused/unfused wait buckets, safe-mode fields), but keep/discard on honest primary metric behavior in stable windows.
- For DFlash-first control parity, require explicit request-level `dflash=false` (now enforced in harness) and discard early-stop short runs.
- Operational sanity check: before any "control" attribution sample, ensure `.omlx-autoresearch/workload.env` actually has `ENABLE_DFLASH=0`, `BENCHMARK_DFLASH=0`, `WARMUP_DFLASH=0`; stale DFlash defaults can silently contaminate control monitors.
- Harness operation note: this runner only accepts plain `./autoresearch.sh` commands in `run_experiment`; switch control/DFlash modes via `workload.env`, not inline env prefixes.
