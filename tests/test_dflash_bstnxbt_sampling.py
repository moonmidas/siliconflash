from __future__ import annotations

import mlx.core as mx

from omlx.dflash.bstnxbt_runtime import argmax_tokens_with_mask, build_suppress_token_mask


def test_build_suppress_token_mask_marks_requested_ids_only():
    mask = build_suppress_token_mask(6, [1, 4])
    mx.eval(mask)

    assert mask.tolist() == [False, True, False, False, True, False]


def test_argmax_tokens_with_mask_avoids_suppressed_ids():
    logits = mx.array([[[0.1, 9.0, 0.2, 8.0]]], dtype=mx.float32)
    suppress_mask = build_suppress_token_mask(4, [1])

    token_ids = argmax_tokens_with_mask(logits, suppress_mask)
    mx.eval(token_ids)

    assert token_ids.tolist() == [[3]]
