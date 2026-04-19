# SPDX-License-Identifier: Apache-2.0
"""Env-gated qwen3_5 GatedDeltaNet boundary snapshot capture for DFlash."""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.gated_delta as gd
import mlx_lm.models.qwen3_5 as qwen3_5

logger = logging.getLogger(__name__)

_class_patch_applied = False


def _clone_mx(value):
    return None if value is None else mx.array(value)


def _make_patched_gdn_call(original_call):
    def patched_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        capture = bool(getattr(cache, "_capture_boundaries", False)) if cache is not None else False
        if not capture:
            return original_call(self, inputs, mask=mask, cache=cache)

        bsz, seq_len, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = qwen3_5.sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            bsz, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (bsz, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)

        n_keep = self.conv_kernel_size - 1
        conv_boundary = []
        for tok_idx in range(seq_len):
            end = n_keep + tok_idx + 1
            conv_boundary.append(_clone_mx(conv_input[:, end - n_keep : end, :]))

        if cache is not None:
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, seq_len)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(bsz, seq_len, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = gd.compute_g(self.A_log, a, self.dt_bias)

        if state is None:
            _, dim_k = q.shape[-2:]
            heads_v, dim_v = v.shape[-2:]
            state = mx.zeros((bsz, heads_v, dim_v, dim_k), dtype=mx.float32)

        repeat_factor = v.shape[2] // q.shape[2]
        q_rep = mx.repeat(q, repeat_factor, -2) if repeat_factor > 1 else q
        k_rep = mx.repeat(k, repeat_factor, -2) if repeat_factor > 1 else k

        snapshot_state = _clone_mx(state)
        state_boundary = []
        for tok_idx in range(seq_len):
            _, snapshot_state = gd._gated_delta_step_ops(
                q_rep[:, tok_idx],
                k_rep[:, tok_idx],
                v[:, tok_idx],
                g[:, tok_idx],
                beta[:, tok_idx],
                snapshot_state,
                None if mask is None else mask[:, tok_idx],
            )
            state_boundary.append(_clone_mx(snapshot_state))

        out, state = gd.gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        out = self.norm(out, z)
        out = self.out_proj(out.reshape(bsz, seq_len, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        if cache is not None:
            cache[1] = state
            cache.advance(seq_len)
            cache._boundary_snapshots = list(zip(conv_boundary, state_boundary))
        return out

    return patched_call


def apply_gated_delta_boundary_capture_patch(model: Any) -> bool:
    global _class_patch_applied

    if _class_patch_applied:
        return True

    model_type = getattr(getattr(model, "args", None), "model_type", None)
    if model_type != "qwen3_5":
        return False

    original_call = qwen3_5.GatedDeltaNet.__call__
    qwen3_5.GatedDeltaNet.__call__ = _make_patched_gdn_call(original_call)
    _class_patch_applied = True
    logger.info("GatedDeltaNet boundary capture patch applied for DFlash")
    return True
