from __future__ import annotations

import os
from typing import Any

from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .bstnxbt_runtime import _lm_head_logits, _target_text_model

_EXIT_LAYER_CACHE: dict[tuple[int, tuple[int, ...], str | None], int] = {}
_LINEAR_LAYER_FLAGS_CACHE: dict[int, tuple[bool, ...]] = {}
_CAPTURE_FLAGS_CACHE: dict[tuple[int, frozenset[int]], tuple[bool, ...]] = {}


def _resolve_linear_layer_flags(inner: Any, total_layers: int) -> tuple[bool, ...]:
    cache_key = id(inner)
    cached = _LINEAR_LAYER_FLAGS_CACHE.get(cache_key)
    if cached is not None and len(cached) == total_layers:
        return cached

    flags = tuple(bool(getattr(layer, "is_linear", False)) for layer in inner.layers)
    _LINEAR_LAYER_FLAGS_CACHE[cache_key] = flags
    return flags


def _resolve_capture_flags(total_layers: int, capture_layer_ids: set[int]) -> tuple[bool, ...]:
    key = (int(total_layers), frozenset(int(layer_id) for layer_id in capture_layer_ids))
    cached = _CAPTURE_FLAGS_CACHE.get(key)
    if cached is not None:
        return cached

    flags = [False] * (total_layers + 1)
    for layer_id in key[1]:
        if 1 <= layer_id <= total_layers:
            flags[layer_id] = True

    resolved = tuple(flags)
    _CAPTURE_FLAGS_CACHE[key] = resolved
    return resolved


def resolve_mirror_exit_layer(total_layers: int, capture_layer_ids: set[int]) -> int:
    env_value = os.environ.get("DFLASH_MIRROR_SD_EXIT_LAYER")
    cache_key = (int(total_layers), tuple(sorted(capture_layer_ids)), env_value)
    cached = _EXIT_LAYER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved: int | None = None
    if env_value is not None:
        try:
            requested = int(env_value)
            resolved = max(0, min(total_layers - 1, requested))
        except ValueError:
            resolved = None

    if resolved is None:
        if capture_layer_ids:
            resolved = max(0, min(total_layers - 1, max(capture_layer_ids) - 1))
        else:
            resolved = max(0, min(total_layers - 1, (total_layers // 2) - 1))

    _EXIT_LAYER_CACHE[cache_key] = resolved
    return resolved


def mirror_target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Any,
    cache: list[Any],
    capture_layer_ids: set[int],
    exit_layer: int | None = None,
) -> tuple[Any, dict[int, Any]]:
    inner = _target_text_model(target_model)
    hidden_states = inner.embed_tokens(input_ids)
    captured: dict[int, Any] = {}

    total_layers = len(inner.layers)
    if total_layers == 0:
        normalized = inner.norm(hidden_states)
        logits = _lm_head_logits(target_model, normalized)
        return logits, captured

    if exit_layer is None:
        exit_layer = resolve_mirror_exit_layer(total_layers, capture_layer_ids)
    else:
        exit_layer = max(0, min(total_layers - 1, int(exit_layer)))

    split_start = exit_layer + 1
    has_mixed_masks = hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")
    capture_flags = _resolve_capture_flags(total_layers, capture_layer_ids)

    if has_mixed_masks:
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        linear_layer_flags = _resolve_linear_layer_flags(inner, total_layers)

        for layer_index in range(split_start):
            layer = inner.layers[layer_index]
            layer_cache = cache[layer_index]
            mask = ssm_mask if linear_layer_flags[layer_index] else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            human_index = layer_index + 1
            if capture_flags[human_index]:
                captured[human_index] = hidden_states

        for layer_index in range(split_start, total_layers):
            layer = inner.layers[layer_index]
            layer_cache = cache[layer_index]
            mask = ssm_mask if linear_layer_flags[layer_index] else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            human_index = layer_index + 1
            if capture_flags[human_index]:
                captured[human_index] = hidden_states
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        layers = inner.layers

        for layer_index, (layer, layer_cache) in enumerate(zip(layers[:split_start], cache[:split_start], strict=True)):
            hidden_states = layer(hidden_states, mask, layer_cache)
            human_index = layer_index + 1
            if capture_flags[human_index]:
                captured[human_index] = hidden_states

        for layer_index, (layer, layer_cache) in enumerate(
            zip(layers[split_start:], cache[split_start:], strict=True),
            start=split_start,
        ):
            hidden_states = layer(hidden_states, mask, layer_cache)
            human_index = layer_index + 1
            if capture_flags[human_index]:
                captured[human_index] = hidden_states

    normalized = inner.norm(hidden_states)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured
