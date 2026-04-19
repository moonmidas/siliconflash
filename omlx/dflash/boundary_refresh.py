from __future__ import annotations

import os
from typing import Any, Iterable

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen35
from mlx_lm.models.base import create_attention_mask

from .sparse_refresh import apply_linear_ssm_refresh
from .target_bridge import _trim_cache_obj


def parse_layer_id_set_env(name: str) -> set[int] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_int_schedule_env(name: str) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return []
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def set_boundary_capture_enabled(
    mlx_cache: list[Any],
    layers: Iterable[Any],
    *,
    enabled: bool,
) -> None:
    for idx, (layer, layer_cache) in enumerate(zip(layers, mlx_cache)):
        if getattr(layer, "is_linear", False):
            layer_cache._capture_boundaries = enabled
            layer_cache._layer_idx = idx


def restore_linear_boundary(
    mlx_cache: list[Any],
    layers: Iterable[Any],
    accepted_tokens_in_block: int,
    *,
    clone_fn=lambda value: mx.array(value),
) -> None:
    if accepted_tokens_in_block <= 0:
        return
    for layer, layer_cache in zip(layers, mlx_cache):
        if not getattr(layer, "is_linear", False):
            continue
        snaps = getattr(layer_cache, "_boundary_snapshots", None)
        if not snaps or accepted_tokens_in_block > len(snaps):
            continue
        conv_state, ssm_state = snaps[accepted_tokens_in_block - 1]
        layer_cache.cache[0] = clone_fn(conv_state)
        layer_cache.cache[1] = clone_fn(ssm_state)


def trim_attention_reject_suffix(
    mlx_cache: list[Any],
    layers: Iterable[Any],
    reject_tokens: int,
    *,
    trim_fn=_trim_cache_obj,
) -> None:
    if reject_tokens <= 0:
        return
    for layer, layer_cache in zip(layers, mlx_cache):
        if getattr(layer, "is_linear", False):
            continue
        trim_fn(layer_cache, reject_tokens)


def native_text_model_fresh_cache(
    model: Any,
    prompt_ids: list[int],
    produced: list[int],
    *,
    include_last: bool = False,
    linear_only_cache: bool = True,
    max_layer: int | None = None,
) -> list[Any]:
    suffix = produced if include_last else produced[:-1]
    prefix = mx.array([prompt_ids + suffix], dtype=mx.int32)
    text_model = model.language_model.model
    cache = model.make_cache() if hasattr(model, "make_cache") else [None] * len(text_model.layers)
    hidden_states = text_model.embed_tokens(prefix)
    fa_mask = create_attention_mask(hidden_states, None if linear_only_cache else cache[text_model.fa_idx])
    ssm_mask = qwen35.create_ssm_mask(hidden_states, cache[text_model.ssm_idx])
    for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
        if max_layer is not None and idx > max_layer:
            break
        active_cache = layer_cache if layer.is_linear or not linear_only_cache else None
        hidden_states = layer(hidden_states, mask=(ssm_mask if layer.is_linear else fa_mask), cache=active_cache)
    return cache


def refresh_linear_ssm_from_native_text_fresh(
    *,
    model: Any,
    live_cache: list[Any],
    prompt_ids: list[int],
    produced: list[int],
    refresh_layer_ids: set[int] | None,
    linear_cutoff: int | None = None,
    stop_at_max_layer: bool = False,
) -> list[int]:
    max_layer = None
    if stop_at_max_layer:
        if refresh_layer_ids:
            max_layer = max(refresh_layer_ids)
        elif linear_cutoff is not None and linear_cutoff >= 0:
            max_layer = linear_cutoff
    fresh_cache = native_text_model_fresh_cache(
        model,
        prompt_ids,
        produced,
        include_last=False,
        linear_only_cache=True,
        max_layer=max_layer,
    )
    return apply_linear_ssm_refresh(
        layers=model.language_model.model.layers,
        live_cache=live_cache,
        fresh_cache=fresh_cache,
        refresh_layer_ids=refresh_layer_ids,
        linear_cutoff=linear_cutoff,
        clone_fn=lambda value: mx.array(value),
    )
