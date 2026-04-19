from __future__ import annotations

import mlx.core as mx

from omlx.dflash.mlx_native_drafter import (
    ContextOnlyDraftKVCache,
    MLXDFlashAttention,
    MLXDFlashDraftArgs,
    MLXDFlashDraftModel,
)


def _tiny_args() -> MLXDFlashDraftArgs:
    return MLXDFlashDraftArgs(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        block_size=4,
        target_layer_ids=(0,),
        mask_token_id=0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def test_make_cache_can_use_context_only_draft_cache(monkeypatch):
    monkeypatch.setenv("DFLASH_BSTNXBT_CONTEXT_ONLY_DRAFT_CACHE", "1")
    monkeypatch.setenv("DFLASH_DRAFT_SINK", "7")
    monkeypatch.setenv("DFLASH_DRAFT_WINDOW", "33")

    model = MLXDFlashDraftModel(_tiny_args())

    caches = model.make_cache()

    assert len(caches) == 1
    assert isinstance(caches[0], ContextOnlyDraftKVCache)
    assert caches[0].sink_size == 7
    assert caches[0].window_size == 33


def test_attention_supports_context_only_draft_cache(monkeypatch):
    attn = MLXDFlashAttention(_tiny_args())
    calls: list[dict[str, object]] = []

    def fake_sdpa(queries, keys, values, *, cache, scale, mask):
        calls.append({"cache": cache, "keys_shape": tuple(keys.shape), "values_shape": tuple(values.shape)})
        return mx.zeros_like(queries)

    monkeypatch.setattr("omlx.dflash.mlx_native_drafter.scaled_dot_product_attention", fake_sdpa)

    hidden_states = mx.random.normal((1, 3, 8)).astype(mx.float32)
    target_hidden = mx.random.normal((1, 5, 8)).astype(mx.float32)
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=8)

    first = attn(hidden_states, target_hidden=target_hidden, cache=cache)
    second = attn(hidden_states, target_hidden=target_hidden, cache=cache)
    mx.eval(first, second)

    assert tuple(first.shape) == (1, 3, 8)
    assert tuple(second.shape) == (1, 3, 8)
    assert cache.offset == 10
    assert calls[0]["cache"] is None
    assert calls[1]["cache"] is None
