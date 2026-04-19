from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from . import bstnxbt_runtime as bst
from .mirror_sd_target import mirror_target_forward_with_hidden_states

_DDTREE_GENERATE_ONCE: Callable[..., dict[str, Any]] | None = None
_DDTREE_IMPORT_ERROR: Exception | None = None
_DDTREE_NATIVE_COMPONENTS: dict[str, Any] | None = None
_DDTREE_NATIVE_IMPORT_ERROR: Exception | None = None


_FALSEY_ENV_VALUES = {"", "0", "false", "no", "off"}


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in _FALSEY_ENV_VALUES


def _add_to_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prepare_optional_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    # ddtree-mlx currently imports dflash_mlx helper modules.
    external_dflash = repo_root / "external" / "dflash-mlx-bstnxbt"
    if external_dflash.exists():
        _add_to_sys_path(external_dflash)

    candidates: list[Path] = []
    env_path = os.environ.get("DFLASH_DDTREE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(repo_root / "external" / "ddtree-mlx")

    for candidate in candidates:
        if (candidate / "ddtree_mlx").exists():
            _add_to_sys_path(candidate)
            break


def _use_native_ddtree_runtime() -> bool:
    # Option 2 (native refactor) is gated for controlled rollout.
    return _env_enabled("DFLASH_DDTREE_NATIVE_RUNTIME", "0")


def _resolve_native_ddtree_components() -> tuple[dict[str, Any] | None, Exception | None]:
    global _DDTREE_NATIVE_COMPONENTS, _DDTREE_NATIVE_IMPORT_ERROR

    if isinstance(_DDTREE_NATIVE_COMPONENTS, dict):
        return _DDTREE_NATIVE_COMPONENTS, _DDTREE_NATIVE_IMPORT_ERROR

    try:
        _prepare_optional_import_paths()

        from ddtree_mlx.tree import (  # type: ignore
            build_ddtree_tree_from_topk,
            follow_verified_tree,
        )
        from ddtree_mlx.compile import compile_tree  # type: ignore
        import ddtree_mlx.verify as ddtree_verify  # type: ignore
        from ddtree_mlx.cache import fast_path_commit, tree_aware_path_commit  # type: ignore

        # Bind verify helper refs to oMLX helpers. Modes:
        # - all (default): bind model + lm_head + split sdpa helper
        # - model_only: bind model + lm_head only
        # - target_only: bind target-model unwrap helper only
        # - none: keep upstream dflash_mlx helper bindings
        bind_mode = os.environ.get("DFLASH_DDTREE_NATIVE_VERIFY_BIND_MODE", "target_only").strip().lower()
        if bind_mode in _FALSEY_ENV_VALUES:
            bind_mode = "none"
        if bind_mode in ("all", "model_only", "target_only"):
            ddtree_verify._target_text_model = bst._target_text_model
        if bind_mode in ("all", "model_only"):
            ddtree_verify._lm_head_logits = bst._lm_head_logits
        if bind_mode == "all":
            ddtree_verify._split_sdpa_output = bst._split_sdpa_output
            ddtree_verify._HYBRID_SDPA_EXACT_KV_THRESHOLD = bst._HYBRID_SDPA_EXACT_KV_THRESHOLD

        _DDTREE_NATIVE_COMPONENTS = {
            "build_ddtree_tree_from_topk": build_ddtree_tree_from_topk,
            "follow_verified_tree": follow_verified_tree,
            "compile_tree": compile_tree,
            "tree_verify_forward": ddtree_verify.tree_verify_forward,
            "fast_path_commit": fast_path_commit,
            "tree_aware_path_commit": tree_aware_path_commit,
        }
        _DDTREE_NATIVE_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - exercised by availability checks
        _DDTREE_NATIVE_COMPONENTS = None
        _DDTREE_NATIVE_IMPORT_ERROR = exc

    return _DDTREE_NATIVE_COMPONENTS, _DDTREE_NATIVE_IMPORT_ERROR


def _resolve_generate_ddtree_once() -> tuple[Callable[..., dict[str, Any]] | None, Exception | None]:
    global _DDTREE_GENERATE_ONCE, _DDTREE_IMPORT_ERROR

    if callable(_DDTREE_GENERATE_ONCE):
        return _DDTREE_GENERATE_ONCE, _DDTREE_IMPORT_ERROR

    try:
        _prepare_optional_import_paths()
        from ddtree_mlx.runtime import generate_ddtree_once  # type: ignore

        _DDTREE_GENERATE_ONCE = generate_ddtree_once
        _DDTREE_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - exercised by availability checks
        _DDTREE_GENERATE_ONCE = None
        _DDTREE_IMPORT_ERROR = exc

    return _DDTREE_GENERATE_ONCE, _DDTREE_IMPORT_ERROR


def ddtree_runtime_availability_reason() -> str | None:
    if _use_native_ddtree_runtime():
        components, import_error = _resolve_native_ddtree_components()
        if isinstance(components, dict):
            return None
        detail = str(import_error) if import_error is not None else "unknown import error"
        return (
            "ddtree_mlx native backend unavailable "
            f"({detail}); install/clone ddtree-mlx and dflash-mlx-bstnxbt "
            "(or set DFLASH_DDTREE_PATH)"
        )

    generate_once, import_error = _resolve_generate_ddtree_once()
    if callable(generate_once):
        return None

    detail = str(import_error) if import_error is not None else "unknown import error"
    return (
        "ddtree_mlx backend unavailable "
        f"({detail}); install ddtree-mlx (pip install ddtree-mlx) "
        "or clone it to external/ddtree-mlx (or set DFLASH_DDTREE_PATH)"
    )


def _tree_budget() -> int:
    # Empirical default tuned for this oMLX lane: 13 currently outperforms 12
    # in matched healthy-window DDTree runs while avoiding low-budget cliffs.
    raw = os.environ.get("DFLASH_DDTREE_BUDGET", os.environ.get("DDTREE_BUDGET", "13"))
    try:
        return max(1, int(raw))
    except Exception:
        return 13


def _resolve_block_size(draft_model: Any) -> int:
    model_block_size = int(getattr(draft_model, "block_size", 16) or 16)
    override_raw = os.environ.get("DFLASH_BLOCK_TOKENS", "").strip()
    if not override_raw:
        override_raw = os.environ.get("DDTREE_BLOCK_SIZE", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(model_block_size, override))
    return max(1, model_block_size)


def _summary_us_to_s(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, 0.0)
    try:
        return float(value) / 1_000_000.0
    except Exception:
        return 0.0


def _phase_us_to_s(summary: dict[str, Any], phase_key: str) -> float:
    phase = summary.get("phase_timings_us")
    if not isinstance(phase, dict):
        return 0.0
    value = phase.get(phase_key, 0.0)
    try:
        return float(value) / 1_000_000.0
    except Exception:
        return 0.0


def _populate_telemetry(telemetry_out: dict[str, Any], summary: dict[str, Any]) -> None:
    elapsed_s = _summary_us_to_s(summary, "elapsed_us")
    prefill_s = _summary_us_to_s(summary, "prefill_us")

    telemetry_out.update(
        {
            # Keep DFlash-style summary fields populated for server metrics.
            "prefill_s": prefill_s,
            "draft_s": _phase_us_to_s(summary, "draft"),
            "verify_s": _phase_us_to_s(summary, "tree_verify") + _phase_us_to_s(summary, "dflash_verify"),
            "eval_s": elapsed_s,
            "total_s": elapsed_s,
            # DDTree diagnostics.
            "ddtree_enabled": 1,
            "ddtree_native_runtime": int(summary.get("native_runtime", 0) or 0),
            "ddtree_tree_budget": int(summary.get("tree_budget", 0) or 0),
            "ddtree_cycles_completed": int(summary.get("cycles_completed", 0) or 0),
            "ddtree_ddtree_cycles_completed": int(summary.get("ddtree_cycles_completed", 0) or 0),
            "ddtree_dflash_cycles_completed": int(summary.get("dflash_cycles_completed", 0) or 0),
            "ddtree_dflash_accepted_from_draft": int(summary.get("dflash_accepted_from_draft", 0) or 0),
            "ddtree_avg_acceptance": float(summary.get("avg_acceptance", 0.0) or 0.0),
            "ddtree_tokens_per_second": float(summary.get("tokens_per_second", 0.0) or 0.0),
            "ddtree_fast_path_ratio": float(summary.get("fast_path_ratio", 0.0) or 0.0),
            "ddtree_fast_path_count": int(summary.get("fast_path_count", 0) or 0),
            "ddtree_slow_path_count": int(summary.get("slow_path_count", 0) or 0),
            "ddtree_tree_aware_commit_count": int(summary.get("tree_aware_commit_count", 0) or 0),
            "ddtree_tree_aware_linear": int(1 if summary.get("tree_aware_linear") else 0),
            "ddtree_exact_commit": int(1 if summary.get("exact_commit") else 0),
            "ddtree_dflash_controller_enabled": int(1 if summary.get("dflash_controller_enabled") else 0),
            "ddtree_dflash_controller_probe_count": int(summary.get("dflash_controller_probe_count", 0) or 0),
            "ddtree_dflash_controller_switch_count": int(summary.get("dflash_controller_switch_count", 0) or 0),
            "ddtree_elapsed_s": elapsed_s,
            "ddtree_prefill_s": prefill_s,
            "ddtree_tree_build_s": _phase_us_to_s(summary, "tree_build"),
            "ddtree_tree_verify_s": _phase_us_to_s(summary, "tree_verify"),
            "ddtree_tree_verify_linear_s": _phase_us_to_s(summary, "tree_verify_linear"),
            "ddtree_tree_verify_attention_s": _phase_us_to_s(summary, "tree_verify_attention"),
            "ddtree_commit_s": _phase_us_to_s(summary, "commit"),
            "ddtree_dflash_draft_s": _phase_us_to_s(summary, "dflash_draft"),
            "ddtree_dflash_verify_s": _phase_us_to_s(summary, "dflash_verify"),
            "ddtree_dflash_replay_s": _phase_us_to_s(summary, "dflash_replay"),
            "ddtree_dflash_commit_s": _phase_us_to_s(summary, "dflash_commit"),
        }
    )


def _target_forward_fn() -> Any:
    if os.environ.get("DFLASH_MIRROR_SD_RUNTIME") == "1":
        return mirror_target_forward_with_hidden_states
    return bst.target_forward_with_hidden_states


def _eval_logits_and_captured(logits: mx.array, captured: dict[int, mx.array] | list[mx.array] | None) -> None:
    if isinstance(captured, dict) and captured:
        mx.eval(logits, *captured.values())
        return
    if isinstance(captured, list) and captured:
        mx.eval(logits, *captured)
        return
    mx.eval(logits)


def _tree_token_id(tree: Any, root_token: int, tree_index: int) -> int:
    if tree_index == 0:
        return int(root_token)
    return int(tree.node_token_ids[tree_index - 1])


def _tree_token_ids(tree: Any, root_token: int, indices: list[int]) -> list[int]:
    return [_tree_token_id(tree, root_token, idx) for idx in indices]


def _build_tree_from_mlx_logits(
    draft_logits: mx.array,
    *,
    budget: int,
    build_ddtree_tree_from_topk: Callable[..., Any],
) -> Any:
    import numpy as np

    if budget <= 0 or int(draft_logits.shape[0]) == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    topk = min(int(budget), int(draft_logits.shape[-1]))
    topk_cap_raw = os.environ.get("DFLASH_DDTREE_TOPK_CAP", "").strip()
    if topk_cap_raw:
        try:
            topk_cap = int(topk_cap_raw)
        except ValueError:
            topk_cap = 0
        if topk_cap > 0:
            topk = max(1, min(topk, topk_cap))

    logits = draft_logits.astype(mx.float32)
    top_indices = mx.argpartition(-logits, kth=topk - 1, axis=-1)[:, :topk]
    top_logits = mx.take_along_axis(logits, top_indices, axis=-1)

    if _env_enabled("DFLASH_DDTREE_TOPK_SKIP_SORT", "0"):
        # Experimental: skip in-topk sorting to reduce tree-build overhead.
        # This may reduce tree quality because sibling order is no longer strict
        # descending probability, so keep env-gated and off by default.
        top_token_ids = top_indices
    else:
        sort_order = mx.argsort(-top_logits, axis=-1)
        top_token_ids = mx.take_along_axis(top_indices, sort_order, axis=-1)
        top_logits = mx.take_along_axis(top_logits, sort_order, axis=-1)

    # Optional fast path: skip full-vocab logsumexp normalization.
    # Scores remain row-shift invariant for branch ordering and can reduce tree-build cost.
    if _env_enabled("DFLASH_DDTREE_TOPK_UNNORMALIZED", "0"):
        row_max = mx.max(top_logits, axis=-1, keepdims=True)
        top_scores = top_logits - row_max
    else:
        top_scores = top_logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    mx.eval(top_token_ids, top_scores)

    return build_ddtree_tree_from_topk(
        np.array(top_token_ids, copy=False),
        np.array(top_scores, copy=False),
        budget=budget,
    )


def _walk_dfs_exact_prefix(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
    dfs_order: list[int],
) -> tuple[list[int], int | None, int]:
    accepted_indices = [0]
    current_index = 0

    while True:
        next_token = int(posterior_tokens[current_index])
        child_index = child_maps[current_index].get(next_token)
        if child_index is None:
            return accepted_indices, next_token, len(accepted_indices)

        next_pos = len(accepted_indices)
        if next_pos < len(dfs_order) and int(dfs_order[next_pos]) == child_index:
            accepted_indices.append(child_index)
            current_index = child_index
            continue

        accepted_indices.append(child_index)
        return accepted_indices, None, next_pos


def _arm_target_rollback_with_prefix(cache_entries: list[Any], prefix_len: int) -> None:
    for cache_entry in cache_entries:
        arm_rollback = getattr(cache_entry, "arm_rollback", None)
        if not callable(arm_rollback):
            continue
        try:
            arm_rollback(prefix_len=int(prefix_len))
        except TypeError:
            arm_rollback()


def _resolve_verify_len_cap(block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _restore_target_cache_after_acceptance_ns(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int,
) -> int:
    start_ns = time.perf_counter_ns()
    bst._restore_target_cache_after_acceptance(
        cache_entries,
        target_len=target_len,
        acceptance_length=acceptance_length,
        drafted_tokens=drafted_tokens,
    )
    return time.perf_counter_ns() - start_ns


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: int | None,
    capture_layer_ids: set[int],
    target_forward_with_hidden_states: Any,
) -> tuple[mx.array, dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        logits, captured = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        return logits, captured

    logits_chunks: list[mx.array] = []
    captured_chunks: list[dict[int, mx.array]] = []

    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_captured = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(chunk_logits)
        captured_chunks.append(chunk_captured)

    merged_captured: dict[int, mx.array] = {}
    for layer_id in capture_layer_ids:
        merged_captured[layer_id] = mx.concatenate(
            [chunk[layer_id] for chunk in captured_chunks],
            axis=1,
        )

    return mx.concatenate(logits_chunks, axis=1), merged_captured


def _execute_ddtree_native_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    drafter_model: Any,
    prompt: str,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None,
) -> dict[str, Any]:
    components, import_error = _resolve_native_ddtree_components()
    if not isinstance(components, dict):
        detail = str(import_error) if import_error is not None else "unknown import error"
        raise RuntimeError(
            "ddtree_mlx native runtime is unavailable "
            f"({detail}); install/clone ddtree-mlx and dflash-mlx-bstnxbt or set DFLASH_DDTREE_PATH"
        )

    build_ddtree_tree_from_topk = components["build_ddtree_tree_from_topk"]
    follow_verified_tree = components["follow_verified_tree"]
    compile_tree = components["compile_tree"]
    tree_verify_forward = components["tree_verify_forward"]
    fast_path_commit = components["fast_path_commit"]
    tree_aware_path_commit = components["tree_aware_path_commit"]

    if _env_enabled("DFLASH_DDTREE_NATIVE_INSTALL_HOOKS", "0"):
        bst.install_target_speculative_hooks(target_model)
    if _env_enabled("DFLASH_DDTREE_NATIVE_SPLIT_SDPA", "0"):
        bst.configure_full_attention_split(
            target_model,
            enabled=True,
            chunk_size=int(os.environ.get("DFLASH_BSTNXBT_SPLIT_CHUNK_SIZE", "8")),
        )

    target_forward_with_hidden_states = _target_forward_fn()

    draft_model = getattr(drafter_model, "model", drafter_model)
    target_layer_ids = list(getattr(draft_model, "target_layer_ids", []) or [])
    if not target_layer_ids:
        raise ValueError("ddtree_mlx native backend requires draft target_layer_ids")

    capture_layer_ids = {int(layer_id) + 1 for layer_id in target_layer_ids}

    prompt_tokens = list(tokenizer.encode(prompt))
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    prompt_len = len(prompt_tokens)

    stop_token_ids = list(stop_token_ids or [])
    stop_token_set = set(int(token_id) for token_id in stop_token_ids)

    if _env_enabled("DFLASH_DDTREE_NATIVE_USE_EXTERNAL_CACHE", "0"):
        import dflash_mlx.runtime as external_runtime  # type: ignore

        target_cache = external_runtime.make_target_cache(
            target_model,
            enable_speculative_linear_cache=True,
        )
    else:
        target_cache = bst.make_target_cache(target_model)
    draft_cache = draft_model.make_cache()

    start_ns = time.perf_counter_ns()
    prefill_start_ns = time.perf_counter_ns()
    prefill_logits, prefill_hidden = target_forward_with_hidden_states(
        target_model,
        input_ids=prompt_array,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    _eval_logits_and_captured(prefill_logits, prefill_hidden)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns

    suppress_mask = bst.build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
    staged_first = bst.argmax_tokens_with_mask(prefill_logits[:, -1, :], suppress_mask).reshape(-1)
    target_hidden = bst.extract_context_feature_from_dict(prefill_hidden, target_layer_ids)

    block_size = _resolve_block_size(draft_model)
    verify_len_cap = _resolve_verify_len_cap(block_size)
    async_draft_eval = os.environ.get("DFLASH_BSTNXBT_ASYNC_DRAFT_EVAL", "0") == "1"
    fused_draft_verify_eval = os.environ.get("DFLASH_BSTNXBT_FUSED_DRAFT_VERIFY_EVAL", "0") == "1"

    generated_tokens: list[int] = []
    start = prompt_len
    cycles_completed = 0
    acceptance_history: list[int] = []
    fast_path_count = 0
    slow_path_count = 0
    ddtree_cycles_completed = 0
    dflash_cycles_completed = 0
    dflash_accepted_from_draft = 0

    draft_ns = 0
    dflash_draft_ns = 0
    dflash_verify_ns = 0
    dflash_replay_ns = 0
    dflash_commit_ns = 0
    tree_build_ns = 0
    tree_verify_ns = 0
    commit_ns = 0
    verify_linear_ns = 0
    verify_attention_ns = 0
    verify_detail_ns: dict[str, int] = {}

    profile_verify_value = os.environ.get("DDTREE_PROFILE_VERIFY", "").lower()
    profile_verify = profile_verify_value not in ("", "0", "false")
    profile_detail_value = os.environ.get("DDTREE_PROFILE_DETAIL", "").lower()
    profile_detail = profile_verify_value in ("detail", "full", "2") or profile_detail_value not in (
        "",
        "0",
        "false",
    )

    tree_aware_linear = _env_enabled("DDTREE_TREE_AWARE_LINEAR", "1")
    tree_aware_commit_count = 0
    exact_commit = _env_enabled("DDTREE_EXACT_COMMIT", "0")

    controller_enabled = _env_enabled("DDTREE_DFLASH_CONTROLLER", "0")
    controller_warmup = int(os.environ.get("DDTREE_CONTROLLER_WARMUP", "16"))
    controller_interval = max(1, int(os.environ.get("DDTREE_CONTROLLER_INTERVAL", "8")))
    controller_margin = float(os.environ.get("DDTREE_CONTROLLER_MARGIN", "1.20"))
    controller_min_probes = max(1, int(os.environ.get("DDTREE_CONTROLLER_MIN_PROBES", "3")))
    controller_mode = "ddtree"
    controller_switch_count = 0
    controller_probe_count = 0
    controller_last_probe_cycle = -1
    ddtree_cycle_tps: list[float] = []
    dflash_cycle_tps: list[float] = []

    def _run_dflash_cycle(block_len: int) -> tuple[int, bool, float]:
        nonlocal target_hidden, staged_first, start
        nonlocal dflash_draft_ns, dflash_verify_ns, dflash_replay_ns, dflash_commit_ns
        nonlocal dflash_cycles_completed, dflash_accepted_from_draft, cycles_completed

        cycle_start_ns = time.perf_counter_ns()
        block_token_ids = mx.full((block_len,), draft_model.mask_token_id, dtype=mx.uint32)
        block_token_ids[0] = staged_first[0] if staged_first.ndim > 0 else staged_first

        draft_logits = None
        if block_len > 1:
            draft_start_ns = time.perf_counter_ns()
            noise_embedding = bst._target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = bst._lm_head_logits(target_model, draft_hidden[:, 1:, :])
            if async_draft_eval:
                mx.async_eval(draft_logits)
            if not fused_draft_verify_eval:
                mx.eval(draft_logits)
            drafted = bst.argmax_tokens_with_mask(draft_logits, suppress_mask).squeeze(0)
            block_token_ids[1:block_len] = drafted
            dflash_draft_ns += time.perf_counter_ns() - draft_start_ns

        verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
        _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

        verify_start_ns = time.perf_counter_ns()
        verify_logits, verify_hidden_raw = _verify_target_block(
            target_model=target_model,
            verify_ids=verify_token_ids[None],
            target_cache=target_cache,
            verify_chunk_tokens=None,
            capture_layer_ids=capture_layer_ids,
            target_forward_with_hidden_states=target_forward_with_hidden_states,
        )
        dflash_verify_ns += time.perf_counter_ns() - verify_start_ns

        posterior = bst.argmax_tokens_with_mask(verify_logits[0], suppress_mask)
        acceptance_len = int(bst._match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item())
        commit_count = 1 + acceptance_len

        committed_segment = verify_token_ids[:commit_count]
        committed_hidden = bst.extract_context_feature_from_dict(verify_hidden_raw, target_layer_ids)[:, :commit_count, :]
        if fused_draft_verify_eval and draft_logits is not None:
            mx.eval(draft_logits, posterior, committed_hidden)
        else:
            mx.eval(committed_hidden, posterior)

        committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
        emitted_ids = committed_ids
        stop_hit = False
        for pos, token_id in enumerate(committed_ids):
            if token_id in stop_token_set:
                emitted_ids = committed_ids[: pos + 1]
                stop_hit = True
                break
        generated_tokens.extend(emitted_ids)

        commit_start_ns = time.perf_counter_ns()
        start += commit_count
        target_hidden = committed_hidden
        replay_ns = _restore_target_cache_after_acceptance_ns(
            target_cache,
            target_len=start,
            acceptance_length=acceptance_len,
            drafted_tokens=block_len - 1,
        )
        dflash_replay_ns += replay_ns
        dflash_commit_ns += time.perf_counter_ns() - commit_start_ns

        staged_first = posterior[acceptance_len : acceptance_len + 1]
        acceptance_history.append(commit_count)
        dflash_accepted_from_draft += acceptance_len
        dflash_cycles_completed += 1
        cycles_completed += 1

        cycle_ns = time.perf_counter_ns() - cycle_start_ns
        cycle_tps = commit_count / (cycle_ns / 1e9) if cycle_ns > 0 else 0.0
        return commit_count, stop_hit, cycle_tps

    tree_budget = _tree_budget()

    while len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        block_len = max(1, min(block_size, remaining))
        cycle_start_ns = time.perf_counter_ns()

        controller_probe = (
            controller_enabled
            and controller_mode == "ddtree"
            and ddtree_cycles_completed >= controller_warmup
            and ddtree_cycles_completed != controller_last_probe_cycle
            and (ddtree_cycles_completed - controller_warmup) % controller_interval == 0
        )
        if controller_mode == "dflash" or controller_probe:
            _, stop_hit, cycle_tps = _run_dflash_cycle(block_len)
            dflash_cycle_tps.append(cycle_tps)
            if controller_probe:
                controller_last_probe_cycle = ddtree_cycles_completed
                controller_probe_count += 1
                recent = ddtree_cycle_tps[-controller_interval:]
                recent_ddtree_tps = (sum(recent) / len(recent)) if recent else 0.0
                recent_dflash = dflash_cycle_tps[-controller_min_probes:]
                recent_dflash_tps = (
                    sum(recent_dflash) / len(recent_dflash)
                    if len(recent_dflash) >= controller_min_probes
                    else 0.0
                )
                all_dflash_probe_tps = (
                    sum(dflash_cycle_tps) / len(dflash_cycle_tps)
                    if len(dflash_cycle_tps) >= controller_min_probes
                    else 0.0
                )
                all_ddtree_tps = (sum(ddtree_cycle_tps) / len(ddtree_cycle_tps)) if ddtree_cycle_tps else 0.0
                if (
                    recent_ddtree_tps > 0
                    and all_ddtree_tps > 0
                    and recent_dflash_tps > recent_ddtree_tps * controller_margin
                    and all_dflash_probe_tps > all_ddtree_tps * controller_margin
                ):
                    controller_mode = "dflash"
                    controller_switch_count += 1
            if stop_hit:
                break
            continue

        draft_start_ns = time.perf_counter_ns()
        block_token_ids = mx.full((block_len,), draft_model.mask_token_id, dtype=mx.uint32)
        block_token_ids[0] = staged_first[0] if staged_first.ndim > 0 else staged_first

        if block_len > 1:
            noise_embedding = bst._target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = bst._lm_head_logits(target_model, draft_hidden[:, 1:, :])
            # Let top-k path perform selective eval.
        else:
            draft_logits = None
        draft_ns += time.perf_counter_ns() - draft_start_ns

        if draft_logits is None or block_len <= 1:
            generated_tokens.append(int(staged_first.item()))
            commit_start_ns = time.perf_counter_ns()
            staged_input = staged_first[None] if staged_first.ndim == 1 else staged_first.reshape(1, 1)
            fwd_logits, fwd_hidden = target_forward_with_hidden_states(
                target_model,
                input_ids=staged_input,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(fwd_logits, fwd_hidden)
            target_hidden = bst.extract_context_feature_from_dict(fwd_hidden, target_layer_ids)
            staged_first = bst.argmax_tokens_with_mask(fwd_logits[:, -1, :], suppress_mask).reshape(-1)
            start += 1
            commit_ns += time.perf_counter_ns() - commit_start_ns
            if generated_tokens[-1] in stop_token_set:
                break
            continue

        build_start_ns = time.perf_counter_ns()
        draft_logits_2d = draft_logits[0].astype(mx.float32)
        if suppress_mask is not None:
            floor = mx.array(-1e9, dtype=draft_logits_2d.dtype)
            draft_logits_2d = mx.where(suppress_mask, floor, draft_logits_2d)
        tree = _build_tree_from_mlx_logits(
            draft_logits_2d,
            budget=tree_budget,
            build_ddtree_tree_from_topk=build_ddtree_tree_from_topk,
        )
        root_token = int(staged_first.item())
        compiled = compile_tree(tree, root_token, prefix_len=start)
        dfs_order_list = [int(index) for index in compiled.dfs_order.tolist()]
        tree_build_ns += time.perf_counter_ns() - build_start_ns

        if not tree_aware_linear:
            _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

        verify_start_ns = time.perf_counter_ns()
        verify_profile = {"_detail": profile_detail} if profile_verify else None
        tree_cache_state: dict[str, Any] | None = {} if tree_aware_linear else None
        verify_logits, verify_hidden = tree_verify_forward(
            target_model,
            compiled_tree=compiled,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            profile_timings=verify_profile,
            tree_aware_linear=tree_aware_linear,
            tree_cache_state=tree_cache_state,
        )
        tree_verify_ns += time.perf_counter_ns() - verify_start_ns

        if verify_profile is not None:
            verify_linear_ns += int(verify_profile.get("linear_ns", 0) or 0)
            verify_attention_ns += int(verify_profile.get("attention_ns", 0) or 0)
            for key, value in verify_profile.items():
                if key.startswith("_") or key in ("linear_ns", "attention_ns"):
                    continue
                if isinstance(value, int):
                    verify_detail_ns[key] = verify_detail_ns.get(key, 0) + value

        posterior = bst.argmax_tokens_with_mask(verify_logits[0], suppress_mask)
        posterior_list = [int(token_id) for token_id in posterior.tolist()]

        if tree_aware_linear:
            accepted_indices, bonus_token = follow_verified_tree(tree.child_maps, posterior_list)
            exact_prefix_len = len(accepted_indices)
        else:
            accepted_indices, bonus_token, exact_prefix_len = _walk_dfs_exact_prefix(
                tree.child_maps,
                posterior_list,
                dfs_order_list,
            )

        commit_start_ns = time.perf_counter_ns()
        all_hidden = bst.extract_context_feature_from_dict(verify_hidden, target_layer_ids)
        use_fast_path = (
            accepted_indices == dfs_order_list[: len(accepted_indices)]
            if tree_aware_linear
            else exact_prefix_len == len(accepted_indices)
        )

        if tree_aware_linear and exact_commit:
            tree_aware_commit_count += 1
            for cache_entry in target_cache:
                if hasattr(cache_entry, "offset") and not hasattr(cache_entry, "rollback"):
                    cache_entry.offset = start
            _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

            accepted_token_ids_commit = _tree_token_ids(tree, root_token, accepted_indices)
            commit_ids_mx = mx.array(accepted_token_ids_commit, dtype=mx.uint32)[None]
            commit_logits, commit_hidden_raw = target_forward_with_hidden_states(
                target_model,
                input_ids=commit_ids_mx,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(commit_logits, commit_hidden_raw)
            committed_hidden = bst.extract_context_feature_from_dict(commit_hidden_raw, target_layer_ids)
            bonus_token = int(bst.argmax_tokens_with_mask(commit_logits[:, -1, :], suppress_mask).item())
            if use_fast_path:
                fast_path_count += 1
            else:
                slow_path_count += 1
        elif tree_aware_linear:
            tree_aware_commit_count += 1
            tree_aware_path_commit(
                target_cache,
                prefix_len=start,
                accepted_indices=accepted_indices,
                tree_cache_state=tree_cache_state or {},
            )
            accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = all_hidden[:, accepted_idx_array, :]
            if use_fast_path:
                fast_path_count += 1
            else:
                slow_path_count += 1
        elif use_fast_path:
            fast_path_count += 1
            fast_path_commit(
                target_cache,
                prefix_len=start,
                n_accepted=len(accepted_indices),
            )
            accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = all_hidden[:, accepted_idx_array, :]
        else:
            slow_path_count += 1
            fast_path_commit(
                target_cache,
                prefix_len=start,
                n_accepted=exact_prefix_len,
            )
            prefix_indices = accepted_indices[:exact_prefix_len]
            prefix_idx_array = mx.array(prefix_indices, dtype=mx.int32)
            hidden_chunks = [all_hidden[:, prefix_idx_array, :]]

            suffix_indices = accepted_indices[exact_prefix_len:]
            suffix_ids = _tree_token_ids(tree, root_token, suffix_indices)
            suffix_ids_mx = mx.array(suffix_ids, dtype=mx.uint32)[None]
            suffix_logits, suffix_hidden_raw = target_forward_with_hidden_states(
                target_model,
                input_ids=suffix_ids_mx,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(suffix_logits, suffix_hidden_raw)
            hidden_chunks.append(bst.extract_context_feature_from_dict(suffix_hidden_raw, target_layer_ids))
            current_index = accepted_indices[-1]
            next_token = int(bst.argmax_tokens_with_mask(suffix_logits[:, -1, :], suppress_mask).item())

            while next_token in tree.child_maps[current_index] and len(accepted_indices) < block_len:
                current_index = tree.child_maps[current_index][next_token]
                accepted_indices.append(current_index)
                token_ids_mx = mx.array([[next_token]], dtype=mx.uint32)
                suffix_logits, suffix_hidden_raw = target_forward_with_hidden_states(
                    target_model,
                    input_ids=token_ids_mx,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                )
                _eval_logits_and_captured(suffix_logits, suffix_hidden_raw)
                hidden_chunks.append(
                    bst.extract_context_feature_from_dict(suffix_hidden_raw, target_layer_ids)
                )
                next_token = int(bst.argmax_tokens_with_mask(suffix_logits[:, -1, :], suppress_mask).item())

            bonus_token = next_token
            committed_hidden = (
                mx.concatenate(hidden_chunks, axis=1)
                if len(hidden_chunks) > 1
                else hidden_chunks[0]
            )

        commit_ns += time.perf_counter_ns() - commit_start_ns

        accepted_token_ids_list = _tree_token_ids(tree, root_token, accepted_indices)
        n_accepted = len(accepted_indices)
        acceptance_history.append(n_accepted)
        emitted_token_ids = accepted_token_ids_list
        stop_hit = False
        for pos, token_id in enumerate(accepted_token_ids_list):
            if token_id in stop_token_set:
                emitted_token_ids = accepted_token_ids_list[: pos + 1]
                stop_hit = True
                break

        generated_tokens.extend(emitted_token_ids)
        start += n_accepted
        target_hidden = committed_hidden
        staged_first = mx.array([int(bonus_token)], dtype=mx.uint32)
        cycles_completed += 1
        ddtree_cycles_completed += 1

        cycle_ns = time.perf_counter_ns() - cycle_start_ns
        if cycle_ns > 0:
            ddtree_cycle_tps.append(n_accepted / (cycle_ns / 1e9))

        if stop_hit:
            break

    generated_tokens = generated_tokens[:max_new_tokens]
    while generated_tokens and generated_tokens[-1] in stop_token_set:
        generated_tokens.pop()

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    generation_tokens = len(generated_tokens)
    phase_timings_us: dict[str, Any] = {
        "prefill": prefill_ns / 1_000.0,
        "draft": draft_ns / 1_000.0,
        "dflash_draft": dflash_draft_ns / 1_000.0,
        "dflash_verify": dflash_verify_ns / 1_000.0,
        "dflash_replay": dflash_replay_ns / 1_000.0,
        "dflash_commit": dflash_commit_ns / 1_000.0,
        "tree_build": tree_build_ns / 1_000.0,
        "tree_verify": tree_verify_ns / 1_000.0,
        "commit": commit_ns / 1_000.0,
    }
    if profile_verify:
        phase_timings_us["tree_verify_linear"] = verify_linear_ns / 1_000.0
        phase_timings_us["tree_verify_attention"] = verify_attention_ns / 1_000.0
    if verify_detail_ns:
        phase_timings_us["tree_verify_detail"] = {
            key: value / 1_000.0
            for key, value in sorted(verify_detail_ns.items())
        }

    return {
        "native_runtime": 1,
        "generated_token_ids": generated_tokens,
        "generation_tokens": generation_tokens,
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "tokens_per_second": generation_tokens / (elapsed_us / 1e6) if elapsed_us > 0 else 0.0,
        "cycles_completed": cycles_completed,
        "ddtree_cycles_completed": ddtree_cycles_completed,
        "dflash_cycles_completed": dflash_cycles_completed,
        "dflash_accepted_from_draft": dflash_accepted_from_draft,
        "acceptance_history": acceptance_history,
        "avg_acceptance": (
            sum(acceptance_history) / len(acceptance_history)
            if acceptance_history
            else 0.0
        ),
        "fast_path_count": fast_path_count,
        "slow_path_count": slow_path_count,
        "tree_aware_commit_count": tree_aware_commit_count,
        "tree_aware_linear": tree_aware_linear,
        "exact_commit": exact_commit,
        "dflash_controller_enabled": controller_enabled,
        "dflash_controller_mode": controller_mode,
        "dflash_controller_probe_count": controller_probe_count,
        "dflash_controller_switch_count": controller_switch_count,
        "dflash_controller_min_probes": controller_min_probes,
        "ddtree_cycle_tps_avg": (
            sum(ddtree_cycle_tps) / len(ddtree_cycle_tps)
            if ddtree_cycle_tps
            else 0.0
        ),
        "dflash_cycle_tps_avg": (
            sum(dflash_cycle_tps) / len(dflash_cycle_tps)
            if dflash_cycle_tps
            else 0.0
        ),
        "fast_path_ratio": (
            fast_path_count / (fast_path_count + slow_path_count)
            if (fast_path_count + slow_path_count) > 0
            else 0.0
        ),
        "phase_timings_us": phase_timings_us,
        "tree_budget": tree_budget,
    }


def _maybe_patch_mirror_target_forward() -> None:
    if os.environ.get("DFLASH_MIRROR_SD_RUNTIME") != "1":
        return
    try:
        import dflash_mlx.runtime as dflash_runtime  # type: ignore

        dflash_runtime.target_forward_with_hidden_states = mirror_target_forward_with_hidden_states
    except Exception:
        # Best effort for legacy wrapper runtime.
        return


def execute_ddtree_mlx_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    drafter_model: Any,
    prompt: str,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    telemetry_out: dict[str, Any] | None = None,
) -> list[int]:
    del enable_thinking, thinking_budget

    if _use_native_ddtree_runtime():
        summary = _execute_ddtree_native_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            drafter_model=drafter_model,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
        )
    else:
        generate_once, import_error = _resolve_generate_ddtree_once()
        if not callable(generate_once):
            detail = str(import_error) if import_error is not None else "unknown import error"
            raise RuntimeError(
                "ddtree_mlx backend is unavailable "
                f"({detail}); install ddtree-mlx or set DFLASH_DDTREE_PATH"
            )

        _maybe_patch_mirror_target_forward()

        prompt_tokens = list(tokenizer.encode(prompt))
        draft_model = getattr(drafter_model, "model", drafter_model)
        effective_block_size = _resolve_block_size(draft_model)
        original_block_size = getattr(draft_model, "block_size", None)
        if original_block_size is not None:
            draft_model.block_size = effective_block_size
        try:
            summary = generate_once(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=int(max_new_tokens),
                tree_budget=_tree_budget(),
                stop_token_ids=stop_token_ids,
                suppress_token_ids=suppress_token_ids,
            )
        finally:
            if original_block_size is not None:
                draft_model.block_size = original_block_size

    if telemetry_out is not None and isinstance(summary, dict):
        _populate_telemetry(telemetry_out, summary)

    generated = summary.get("generated_token_ids", []) if isinstance(summary, dict) else []
    return [int(token_id) for token_id in generated]
