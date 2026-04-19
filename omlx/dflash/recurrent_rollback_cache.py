from __future__ import annotations

import os
from typing import Any, Callable

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache


class RecurrentRollbackCache(_BaseCache):
    """Rollback-capable linear cache modeled after bstnxbt/dflash-mlx.

    This cache stores a speculative snapshot before verify, records the
    recurrent innovation tape for the verified block, and can restore the
    cache to the accepted prefix without replaying the entire target prefix.

    The implementation keeps the MLX/tape-replay dependencies injectable so
    unit tests can validate rollback semantics without requiring live kernels.
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance._armed = False
        instance._tape = None
        instance._tape_k = None
        instance._tape_g = None
        instance._tape_qkv = None
        instance._snapshot = None
        return instance

    def __init__(
        self,
        size: int,
        *,
        conv_kernel_size: int = 4,
        mx_module: Any = mx,
        tape_replay_fn: Callable[..., Any] | None = None,
    ):
        self.cache = [None] * int(size)
        self.conv_kernel_size = int(conv_kernel_size)
        self._mx = mx_module
        self._tape_replay_fn = tape_replay_fn
        self._snapshot_copy = os.environ.get("DFLASH_BSTNXBT_ROLLBACK_SNAPSHOT_COPY", "0") == "1"

    def __getitem__(self, idx: int):
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self.cache[idx] = value

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value) -> None:
        self.cache = value

    def filter(self, batch_indices):
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        def cat(lhs, rhs):
            if lhs is None:
                return rhs
            if rhs is None:
                return lhs
            return self._mx.concatenate([lhs, rhs])

        self.cache = [cat(lhs, rhs) for lhs, rhs in zip(self.cache, other.cache, strict=True)]

    def extract(self, idx):
        cache = RecurrentRollbackCache(
            len(self.cache),
            conv_kernel_size=self.conv_kernel_size,
            mx_module=self._mx,
            tape_replay_fn=self._tape_replay_fn,
        )
        cache.cache = [c[idx : idx + 1] if c is not None else None for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = None if lengths is None else self._mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, n: int):
        if self.lengths is not None:
            self.lengths -= n
        if self.left_padding is not None:
            self.left_padding -= n

    def make_mask(self, n: int):
        if self.left_padding is not None:
            pos = self._mx.arange(n)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = self._mx.arange(n)
            return pos < self.lengths[:, None]
        return None

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)

    def checkpoint(self) -> None:
        if self._snapshot_copy:
            self._snapshot = [None if c is None else self._mx.array(c) for c in self.cache]
            return
        # Fast path: MLX layer/cache updates are functional (new arrays), so
        # keeping references avoids per-step snapshot copy overhead.
        self._snapshot = list(self.cache)

    def arm_rollback(self, prefix_len: int = 0) -> None:
        del prefix_len
        self._armed = True
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self.checkpoint()

    def record_tape(
        self,
        *,
        tape: Any,
        k: Any,
        g: Any,
        qkv: Any,
    ) -> None:
        tape_now = self._mx.contiguous(tape)
        k_now = self._mx.contiguous(k)
        g_now = self._mx.contiguous(g)
        qkv_now = self._mx.contiguous(qkv)

        # Verify chunking may invoke linear layers multiple times between
        # arm_rollback() and rollback(). In that case, keep a contiguous
        # sequence of tape slices so rollback can replay the full verified span.
        if self._armed and self._tape is not None:
            self._tape = self._mx.concatenate([self._tape, tape_now], axis=1)
            self._tape_k = self._mx.concatenate([self._tape_k, k_now], axis=1)
            self._tape_g = self._mx.concatenate([self._tape_g, g_now], axis=1)
            self._tape_qkv = self._mx.concatenate([self._tape_qkv, qkv_now], axis=1)
            return

        self._tape = tape_now
        self._tape_k = k_now
        self._tape_g = g_now
        self._tape_qkv = qkv_now

    def _rebuild_conv_state(self, accepted_steps: int):
        if self._tape_qkv is None:
            return self.cache[0]
        keep = self.conv_kernel_size - 1
        if keep <= 0:
            return None

        # Fast path: once we accepted at least `keep` new qkv steps, the
        # rebuilt conv state is just the trailing accepted qkv slice.
        if accepted_steps >= keep:
            start = accepted_steps - keep
            end = accepted_steps
            return self._mx.contiguous(self._tape_qkv[:, start:end, :])

        conv_state = self._snapshot[0] if self._snapshot is not None else None
        if conv_state is None:
            prefix = self._mx.zeros(
                (self._tape_qkv.shape[0], keep, self._tape_qkv.shape[-1]),
                dtype=self._tape_qkv.dtype,
            )
        else:
            prefix = conv_state
        conv_input = self._mx.concatenate([prefix, self._tape_qkv[:, :accepted_steps, :]], axis=1)
        start = accepted_steps
        end = min(start + keep, int(conv_input.shape[1]))
        return self._mx.contiguous(conv_input[:, start:end, :])

    def rollback(self, n_accepted: int) -> None:
        if self._snapshot is None:
            return

        self.cache = list(self._snapshot)
        if (
            self._tape is not None
            and self._tape_k is not None
            and self._tape_g is not None
            and self.cache[1] is not None
            and self._tape_replay_fn is not None
        ):
            accepted_steps = int(n_accepted) + 1
            self.cache[1] = self._tape_replay_fn(
                self._tape[:, :accepted_steps],
                self._tape_k[:, :accepted_steps],
                self._tape_g[:, :accepted_steps],
                self.cache[1],
                None,
            )
            self.cache[0] = self._rebuild_conv_state(accepted_steps)

        self._armed = False
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
