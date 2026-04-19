from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from omlx.dflash.bstnxbt_kernels import batched_sdpa_2pass_exact
from omlx.dflash.bstnxbt_runtime import configure_full_attention_split, _split_sdpa_mask


class _FakeAttn:
    pass


class _FakeLinearAttn:
    pass


def test_split_sdpa_mask_slices_query_and_key_ranges():
    mask = mx.arange(1 * 2 * 6 * 10).reshape(1, 2, 6, 10)

    out = _split_sdpa_mask(mask, query_start=2, query_end=5, key_end=7)
    mx.eval(out)

    assert out.shape == (1, 2, 3, 7)
    assert out.tolist() == mask[..., 2:5, :7].tolist()


def test_configure_full_attention_split_marks_only_full_attention_layers():
    attn0 = _FakeAttn()
    attn1 = _FakeAttn()
    target_model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(is_linear=False, self_attn=attn0),
                    SimpleNamespace(is_linear=True, linear_attn=_FakeLinearAttn()),
                    SimpleNamespace(is_linear=False, self_attn=attn1),
                ]
            )
        )
    )

    configure_full_attention_split(target_model, enabled=True, chunk_size=4)

    assert attn0._dflash_split_sdpa_enabled is True
    assert attn0._dflash_split_sdpa_chunk_size == 4
    assert attn1._dflash_split_sdpa_enabled is True
    assert attn1._dflash_split_sdpa_chunk_size == 4


def test_batched_sdpa_2pass_exact_returns_none_for_unsupported_q_len():
    q = mx.zeros((1, 8, 8, 128), dtype=mx.float16)
    k = mx.zeros((1, 8, 64, 128), dtype=mx.float16)
    v = mx.zeros((1, 8, 64, 128), dtype=mx.float16)

    out = batched_sdpa_2pass_exact(q, k, v, scale=1.0)

    assert out is None
