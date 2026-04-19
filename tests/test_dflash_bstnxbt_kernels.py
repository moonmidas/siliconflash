from __future__ import annotations

import mlx.core as mx

from omlx.dflash.bstnxbt_kernels import gated_delta_kernel_with_tape, tape_replay_kernel


def test_gated_delta_kernel_with_tape_cpu_fallback_returns_expected_shapes():
    q = mx.ones((1, 2, 1, 32), dtype=mx.float32)
    k = mx.ones((1, 2, 1, 32), dtype=mx.float32)
    v = mx.ones((1, 2, 1, 4), dtype=mx.float32)
    g = mx.ones((1, 2, 1), dtype=mx.float32)
    beta = mx.ones((1, 2, 1), dtype=mx.float32)
    state = mx.zeros((1, 1, 4, 32), dtype=mx.float32)

    y, state_out, tape = gated_delta_kernel_with_tape(q, k, v, g, beta, state, None)
    mx.eval(y, state_out, tape)

    assert tuple(y.shape) == (1, 2, 1, 4)
    assert tuple(state_out.shape) == (1, 1, 4, 32)
    assert tuple(tape.shape) == (1, 2, 1, 4)


def test_tape_replay_kernel_cpu_fallback_returns_expected_shape():
    tape = mx.ones((1, 2, 1, 4), dtype=mx.float32)
    k = mx.ones((1, 2, 1, 32), dtype=mx.float32)
    g = mx.ones((1, 2, 1), dtype=mx.float32)
    state = mx.zeros((1, 1, 4, 32), dtype=mx.float32)

    state_out = tape_replay_kernel(tape, k, g, state, None)
    mx.eval(state_out)

    assert tuple(state_out.shape) == (1, 1, 4, 32)
