from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention

from .bstnxbt_kernels import (
    batched_sdpa_2pass_exact,
    gated_delta_kernel_with_tape,
    tape_replay_kernel,
)
from .recurrent_rollback_cache import RecurrentRollbackCache


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")



def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


class _ExactSmallProjPad:
    def __init__(self, linear: Any, *, mx_module: Any = mx, pad_m: int = 16):
        self.linear = linear
        self._mx = mx_module
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value):
        self.linear.bias = value

    def __call__(self, x):
        if getattr(x, "ndim", len(getattr(x, "shape", ()))) == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = self._mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(self._mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)



def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    mx_module: Any = mx,
    pad_m: int = 16,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, mx_module=mx_module, pad_m=pad_m))



def _tape_replay_ops(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array],
):
    del mask
    replay_state = state
    for t in range(int(tape.shape[1])):
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        else:
            decay = g[:, t, :, None, None]
        replay_state = replay_state * decay
        replay_state = replay_state + k[:, t, :, None, :] * tape[:, t, :, :, None]
    return replay_state



def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")



def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")



def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim


def _linear_forward(x: mx.array, weight: mx.array, bias: Optional[mx.array]) -> mx.array:
    out = x @ weight.T
    return out if bias is None else out + bias


def _concat_biases(*biases: Optional[mx.array]) -> Optional[mx.array]:
    present = [bias for bias in biases if bias is not None]
    if not present:
        return None
    if len(present) != len(biases):
        raise ValueError("expected either all packed biases or none")
    return mx.concatenate(present, axis=0)


def _set_linear_from_packed(
    linear: Any,
    packed_weight: mx.array,
    start: int,
    end: int,
    packed_bias: Optional[mx.array],
) -> None:
    linear.weight = packed_weight[start:end]
    if getattr(linear, "bias", None) is not None and packed_bias is not None:
        linear.bias = packed_bias[start:end]


def _is_dense_packable_linear(linear: Any) -> bool:
    weight = getattr(linear, "weight", None)
    if weight is None:
        return False
    if getattr(weight, "ndim", None) != 2:
        return False
    # QuantizedLinear exposes scales/bits/group_size and is not safe to pack
    # with raw dense concat assumptions.
    if hasattr(linear, "scales") or hasattr(linear, "bits"):
        return False
    return True


def _install_packed_mlp_hook(mlp: Any) -> None:
    cls = type(mlp)
    if getattr(cls, "_dflash_packed_call_installed", False):
        return

    original_call = cls.__call__

    def packed_call(self, x) -> mx.array:
        packed_weight = getattr(self, "_dflash_gate_up_weight", None)
        if packed_weight is None:
            return original_call(self, x)
        gate_up = _linear_forward(
            x,
            packed_weight,
            getattr(self, "_dflash_gate_up_bias", None),
        )
        gate, up = mx.split(gate_up, [self._dflash_gate_proj_out_dim], axis=-1)
        return self.down_proj(swiglu(gate, up))

    cls.__call__ = packed_call
    cls._dflash_packed_call_installed = True


def _install_packed_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_packed_call_installed", False):
        return

    original_call = cls.__call__

    def packed_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        packed_weight = getattr(self, "_dflash_qkv_weight", None)
        if packed_weight is None:
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        qkv = _linear_forward(x, packed_weight, getattr(self, "_dflash_qkv_bias", None))
        q_proj_dim = self._dflash_q_proj_out_dim
        k_proj_dim = self._dflash_k_proj_out_dim
        q_proj_output, keys, values = mx.split(
            qkv,
            [q_proj_dim, q_proj_dim + k_proj_dim],
            axis=-1,
        )

        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = packed_call
    cls._dflash_packed_call_installed = True


def _pack_qwen3next_mlp(mlp: Any) -> dict[str, Any]:
    if getattr(mlp, "_dflash_gate_up_weight", None) is not None:
        return {"packed": True, "already_packed": True}

    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if gate_proj is None or up_proj is None or down_proj is None:
        return {"packed": False, "reason": "missing_glu_projections"}
    if not (_is_dense_packable_linear(gate_proj) and _is_dense_packable_linear(up_proj)):
        return {"packed": False, "reason": "non_dense_or_quantized_projection"}

    gate_weight = gate_proj.weight
    up_weight = up_proj.weight
    gate_bias = getattr(gate_proj, "bias", None)
    up_bias = getattr(up_proj, "bias", None)
    gate_out_dim = int(gate_weight.shape[0])

    packed_weight = mx.concatenate([gate_weight, up_weight], axis=0)
    packed_bias = _concat_biases(gate_bias, up_bias)
    if packed_bias is None:
        mx.eval(packed_weight)
    else:
        mx.eval(packed_weight, packed_bias)

    _install_packed_mlp_hook(mlp)
    mlp._dflash_gate_up_weight = packed_weight
    mlp._dflash_gate_up_bias = packed_bias
    mlp._dflash_gate_proj_out_dim = gate_out_dim

    _set_linear_from_packed(gate_proj, packed_weight, 0, gate_out_dim, packed_bias)
    _set_linear_from_packed(
        up_proj,
        packed_weight,
        gate_out_dim,
        gate_out_dim + int(up_weight.shape[0]),
        packed_bias,
    )

    return {
        "packed": True,
        "gate_out_dim": gate_out_dim,
        "up_out_dim": int(up_weight.shape[0]),
    }


def _pack_qwen3next_attention(attn: Any) -> dict[str, Any]:
    if getattr(attn, "_dflash_qkv_weight", None) is not None:
        return {"packed": True, "already_packed": True}

    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if q_proj is None or k_proj is None or v_proj is None:
        return {"packed": False, "reason": "missing_qkv_projection"}
    if not (
        _is_dense_packable_linear(q_proj)
        and _is_dense_packable_linear(k_proj)
        and _is_dense_packable_linear(v_proj)
    ):
        return {"packed": False, "reason": "non_dense_or_quantized_projection"}

    q_weight = q_proj.weight
    k_weight = k_proj.weight
    v_weight = v_proj.weight
    q_bias = getattr(q_proj, "bias", None)
    k_bias = getattr(k_proj, "bias", None)
    v_bias = getattr(v_proj, "bias", None)

    q_out_dim = int(q_weight.shape[0])
    k_out_dim = int(k_weight.shape[0])
    v_out_dim = int(v_weight.shape[0])

    packed_weight = mx.concatenate([q_weight, k_weight, v_weight], axis=0)
    packed_bias = _concat_biases(q_bias, k_bias, v_bias)
    if packed_bias is None:
        mx.eval(packed_weight)
    else:
        mx.eval(packed_weight, packed_bias)

    _install_packed_attention_hook(attn)
    attn._dflash_qkv_weight = packed_weight
    attn._dflash_qkv_bias = packed_bias
    attn._dflash_q_proj_out_dim = q_out_dim
    attn._dflash_k_proj_out_dim = k_out_dim

    _set_linear_from_packed(q_proj, packed_weight, 0, q_out_dim, packed_bias)
    _set_linear_from_packed(k_proj, packed_weight, q_out_dim, q_out_dim + k_out_dim, packed_bias)
    _set_linear_from_packed(
        v_proj,
        packed_weight,
        q_out_dim + k_out_dim,
        q_out_dim + k_out_dim + v_out_dim,
        packed_bias,
    )

    return {
        "packed": True,
        "q_out_dim": q_out_dim,
        "k_out_dim": k_out_dim,
        "v_out_dim": v_out_dim,
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _target_model_type(target_model: Any) -> str:
    wrapper = _target_text_wrapper(target_model)
    for obj in (
        getattr(wrapper, "args", None),
        getattr(target_model, "args", None),
        getattr(getattr(target_model, "language_model", None), "args", None),
    ):
        model_type = getattr(obj, "model_type", None)
        if isinstance(model_type, str):
            return model_type
    return ""


def _maybe_pack_target_model_weights(target_model: Any) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    existing = getattr(text_model, "_dflash_bstnxbt_pack_info", None)
    if isinstance(existing, dict):
        return existing

    pack_mode = str(os.environ.get("DFLASH_BSTNXBT_PACK_TARGET", "auto")).strip().lower()
    model_type = _target_model_type(target_model)
    auto_enabled = model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_next"}
    enabled = False
    if pack_mode in {"1", "true", "yes", "on"}:
        enabled = True
    elif pack_mode in {"auto", ""}:
        enabled = auto_enabled

    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": pack_mode,
        "model_type": model_type,
        "auto_enabled": auto_enabled,
        "pack_mlp": False,
        "pack_attention": False,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    if not enabled:
        text_model._dflash_bstnxbt_pack_info = info
        return info

    default_pack_mlp = model_type != "qwen3_5_moe"
    pack_mlp = _env_bool("DFLASH_BSTNXBT_PACK_MLP", default_pack_mlp)
    pack_attention = _env_bool("DFLASH_BSTNXBT_PACK_ATTENTION", True)
    info["pack_mlp"] = bool(pack_mlp)
    info["pack_attention"] = bool(pack_attention)

    for layer_index, layer in enumerate(text_model.layers):
        if pack_mlp:
            mlp = getattr(layer, "mlp", None)
            if type(mlp).__name__ == "Qwen3NextMLP":
                mlp_result = _pack_qwen3next_mlp(mlp)
                mlp_result["layer_index"] = layer_index
                mlp_result["path"] = "layer.mlp"
                info["packed_mlp_layers"].append(mlp_result)
            shared_expert = getattr(mlp, "shared_expert", None)
            if type(shared_expert).__name__ == "Qwen3NextMLP":
                shared_result = _pack_qwen3next_mlp(shared_expert)
                shared_result["layer_index"] = layer_index
                shared_result["path"] = "layer.mlp.shared_expert"
                info["packed_mlp_layers"].append(shared_result)

        if pack_attention:
            attn = getattr(layer, "self_attn", None)
            if type(attn).__name__ == "Qwen3NextAttention":
                attn_result = _pack_qwen3next_attention(attn)
                attn_result["layer_index"] = layer_index
                info["packed_attention_layers"].append(attn_result)

    text_model._dflash_bstnxbt_pack_info = info
    return info


_split_sdpa_telemetry_local = threading.local()


def _set_split_sdpa_telemetry_collector(collector: dict[str, int] | None) -> None:
    _split_sdpa_telemetry_local.collector = collector


def _split_sdpa_telemetry_collector() -> dict[str, int] | None:
    return getattr(_split_sdpa_telemetry_local, "collector", None)



def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]



def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024



def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool = True,
) -> list[Any]:
    text_model = _target_text_model(target_model)
    use_recurrent_kernels = os.environ.get("DFLASH_BSTNXBT_RECURRENT_KERNELS") == "1"
    tape_replay_fn = tape_replay_kernel if use_recurrent_kernels else _tape_replay_ops
    caches: list[Any] = []
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(
                        size=2,
                        conv_kernel_size=conv_kernel_size,
                        tape_replay_fn=tape_replay_fn,
                    )
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            caches.append(cache_mod.KVCache())
    return caches



def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z_proj = self.in_proj_z(inputs)
        z = z_proj.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((B, h_v, d_v, d_k), dtype=q.dtype)
        state_in = state

        use_recurrent_kernels = os.environ.get("DFLASH_BSTNXBT_RECURRENT_KERNELS") == "1"
        if use_recurrent_kernels:
            out, state, innovation_tape = gated_delta_kernel_with_tape(
                q, k, v, g, beta, state, mask
            )
            cache.record_tape(
                tape=innovation_tape,
                k=k,
                g=g,
                qkv=qkv,
            )
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
            decayed_state = state_in[:, None, ...] * decay
            kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
            innovation_tape = (v - kv_mem) * beta[..., None]
            cache.record_tape(
                tape=innovation_tape.astype(mx.float32),
                k=k,
                g=g,
                qkv=qkv,
            )

        cache[1] = state
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True



def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not bool(getattr(self, "_dflash_split_sdpa_enabled", False)):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        collector = _split_sdpa_telemetry_collector()
        if collector is not None:
            collector["full_attention_calls"] = int(collector.get("full_attention_calls", 0)) + 1

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(0, 2, 1, 3)

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        if collector is not None and should_split:
            collector["split_exact_prefix_calls"] = int(collector.get("split_exact_prefix_calls", 0)) + 1

        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            if collector is not None:
                collector["split_path_calls"] = int(collector.get("split_path_calls", 0)) + 1
                collector["split_batched_2pass_calls"] = int(collector.get("split_batched_2pass_calls", 0)) + 1
            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                if collector is not None:
                    collector["split_batched_2pass_fallback_calls"] = int(collector.get("split_batched_2pass_fallback_calls", 0)) + 1
                    collector["split_query_chunks"] = int(collector.get("split_query_chunks", 0)) + int(queries.shape[2])
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            split_chunk_size = max(1, int(getattr(self, "_dflash_split_sdpa_chunk_size", 8)))
            if collector is not None:
                collector["split_path_calls"] = int(collector.get("split_path_calls", 0)) + 1
                collector["split_query_chunks"] = int(collector.get("split_query_chunks", 0)) + ((int(queries.shape[2]) + split_chunk_size - 1) // split_chunk_size)
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=split_chunk_size,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True



def install_target_speculative_hooks(target_model: Any, *, mx_module: Any = mx, pad_m: int = 16) -> None:
    text_model = _target_text_model(target_model)
    _maybe_pack_target_model_weights(target_model)
    if getattr(text_model, "_dflash_bstnxbt_hooks_installed", False):
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn, mx_module=mx_module, pad_m=pad_m)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_bstnxbt_hooks_installed = True



def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    install_target_speculative_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = _HYBRID_SDPA_EXACT_KV_THRESHOLD



def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens



def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)



def build_suppress_token_mask(vocab_size: int, suppress_token_ids: list[int] | None) -> mx.array | None:
    if not suppress_token_ids:
        return None
    mask = mx.zeros((int(vocab_size),), dtype=mx.bool_)
    valid_ids = [int(token_id) for token_id in suppress_token_ids if 0 <= int(token_id) < int(vocab_size)]
    if not valid_ids:
        return None
    indices = mx.array(valid_ids, dtype=mx.int32)
    vocab = mx.arange(int(vocab_size), dtype=mx.int32)
    return mx.any(mx.equal(vocab[None, :], indices[:, None]), axis=0)



def argmax_tokens_with_mask(logits: mx.array, suppress_mask: mx.array | None) -> mx.array:
    if suppress_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    masked_logits = mx.where(suppress_mask.reshape((1,) * (logits.ndim - 1) + (-1,)), -mx.inf, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)



def _argmax_tokens(logits: mx.array) -> mx.array:
    return argmax_tokens_with_mask(logits, None)



def _match_acceptance_length(drafted_tokens: mx.array, posterior_tokens: mx.array) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))



def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)



def _get_think_token_id(tokenizer: Any, attr: str) -> int | None:
    try:
        return getattr(tokenizer, attr, None)
    except (TypeError, ValueError):
        return None


def _resolve_think_end_token_ids(tokenizer: Any) -> list[int] | None:
    think_end_id = _get_think_token_id(tokenizer, "think_end_id")
    if think_end_id is not None:
        return [think_end_id]
    think_end_str = getattr(tokenizer, "think_end", "</think>")
    try:
        ids = tokenizer.encode(think_end_str, add_special_tokens=False)
        if ids:
            return list(ids)
    except Exception:
        pass
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
        if tid != getattr(tokenizer, "unk_token_id", None):
            return [tid]
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def _prompt_needs_think_prefix(tokenizer: Any, prompt_token_ids: list[int]) -> bool:
    think_start_id = _get_think_token_id(tokenizer, "think_start_id")
    if think_start_id is None:
        try:
            think_start_id = tokenizer.convert_tokens_to_ids("<think>")
            if think_start_id == getattr(tokenizer, "unk_token_id", None):
                return False
        except (AttributeError, KeyError, TypeError):
            return False
    if not think_start_id or not prompt_token_ids:
        return False
    last_tokens = list(prompt_token_ids[-3:])
    if think_start_id not in last_tokens:
        return False
    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1:]
    if after_start:
        think_end_ids = _resolve_think_end_token_ids(tokenizer)
        if think_end_ids and think_end_ids[0] in after_start:
            return False
    return True


@dataclass
class _ThinkingBudgetState:
    budget: int
    think_end_ids: list[int]
    thinking_tokens: int = 0
    in_thinking: bool = True
    force_queue: list[int] = field(default_factory=list)
    recent_tokens: list[int] = field(default_factory=list)


def _init_thinking_budget_state(
    tokenizer: Any,
    prompt_token_ids: list[int],
    thinking_budget: int | None,
) -> _ThinkingBudgetState | None:
    if thinking_budget is None or thinking_budget <= 0:
        return None
    if not _prompt_needs_think_prefix(tokenizer, prompt_token_ids):
        return None
    think_end_ids = _resolve_think_end_token_ids(tokenizer)
    if not think_end_ids:
        return None
    return _ThinkingBudgetState(budget=int(thinking_budget), think_end_ids=list(think_end_ids))


def _update_thinking_budget_state(
    state: _ThinkingBudgetState | None,
    committed_segment: list[int],
    forced_prefix_len: int,
) -> None:
    if state is None or not committed_segment:
        return
    max_end = len(state.think_end_ids)
    for idx, token_id in enumerate(committed_segment):
        state.recent_tokens.append(int(token_id))
        if len(state.recent_tokens) > max_end:
            state.recent_tokens.pop(0)
        if state.recent_tokens == state.think_end_ids:
            state.in_thinking = False
            state.force_queue.clear()
            continue
        forced = idx < forced_prefix_len
        if state.in_thinking and not forced:
            state.thinking_tokens += 1
            if state.thinking_tokens >= state.budget and not state.force_queue:
                state.force_queue = list(state.think_end_ids)


def _resolve_block_tokens(
    draft_model: Any,
    *,
    enable_thinking: bool = False,
    ignore_eos: bool = False,
) -> int:
    draft_block_size = int(getattr(draft_model, "block_size", 16) or 16)
    override_raw = os.environ.get("DFLASH_BLOCK_TOKENS", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(draft_block_size, override))
    if enable_thinking:
        thinking_default = 9 if ignore_eos else 11
        return max(1, min(draft_block_size, thinking_default))
    return max(1, draft_block_size)


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


def _resolve_verify_len_warmup(
    *,
    block_tokens: int,
    verify_len_cap: int,
) -> tuple[int, int]:
    steps_raw = os.environ.get("DFLASH_VERIFY_LEN_WARMUP_STEPS", "").strip()
    if steps_raw:
        try:
            warmup_steps = int(steps_raw)
        except ValueError:
            warmup_steps = 0
    else:
        warmup_steps = 0
    warmup_steps = max(0, warmup_steps)

    cap_raw = os.environ.get("DFLASH_VERIFY_LEN_WARMUP_CAP", "").strip()
    if cap_raw:
        try:
            warmup_cap = int(cap_raw)
        except ValueError:
            warmup_cap = verify_len_cap
    else:
        warmup_cap = verify_len_cap
    warmup_cap = max(1, min(int(block_tokens), warmup_cap))
    warmup_cap = max(1, min(int(verify_len_cap), warmup_cap))
    return warmup_steps, warmup_cap


def _resolve_verify_chunk_tokens() -> int | None:
    override_raw = os.environ.get("DFLASH_BSTNXBT_VERIFY_CHUNK_TOKENS", "").strip()
    if not override_raw:
        return None
    try:
        override = int(override_raw)
    except ValueError:
        return None
    return override if override > 0 else None


def _resolve_speculative_linear_cache_enabled(verify_chunk_tokens: int | None) -> bool:
    raw = os.environ.get("DFLASH_BSTNXBT_SPECULATIVE_LINEAR_CACHE", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return verify_chunk_tokens is None


@dataclass
class _AdaptiveBlockConfig:
    enabled: bool
    min_block_tokens: int
    max_block_tokens: int
    warmup_steps: int
    window_steps: int
    grow_acceptance_threshold: float
    shrink_acceptance_threshold: float
    grow_step: int
    shrink_step: int
    cooldown_steps: int


@dataclass
class _AdaptiveBlockState:
    current_block_tokens: int
    acceptance_window: list[float] = field(default_factory=list)
    last_adjust_step: int = 0
    grow_events: int = 0
    shrink_events: int = 0


def _resolve_adaptive_block_config(
    *,
    initial_block_tokens: int,
    draft_model: Any,
) -> _AdaptiveBlockConfig:
    enabled = _env_bool("DFLASH_BSTNXBT_ADAPTIVE_BLOCK_TOKENS", False)
    draft_block_size = int(getattr(draft_model, "block_size", initial_block_tokens) or initial_block_tokens)
    max_default = max(1, min(draft_block_size, int(initial_block_tokens)))
    max_block_tokens = _env_int(
        "DFLASH_BSTNXBT_ADAPTIVE_MAX_BLOCK_TOKENS",
        max_default,
        minimum=1,
    )
    max_block_tokens = max(1, min(draft_block_size, max_block_tokens))

    min_default = max(1, min(max_block_tokens, max_default - 1))
    min_block_tokens = _env_int(
        "DFLASH_BSTNXBT_ADAPTIVE_MIN_BLOCK_TOKENS",
        min_default,
        minimum=1,
    )
    min_block_tokens = max(1, min(max_block_tokens, min_block_tokens))

    grow_acceptance_threshold = _env_float(
        "DFLASH_BSTNXBT_ADAPTIVE_GROW_ACCEPTANCE",
        0.85,
        minimum=0.0,
    )
    shrink_acceptance_threshold = _env_float(
        "DFLASH_BSTNXBT_ADAPTIVE_SHRINK_ACCEPTANCE",
        0.35,
        minimum=0.0,
    )
    if shrink_acceptance_threshold > grow_acceptance_threshold:
        shrink_acceptance_threshold = grow_acceptance_threshold

    config = _AdaptiveBlockConfig(
        enabled=enabled,
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
        warmup_steps=_env_int("DFLASH_BSTNXBT_ADAPTIVE_WARMUP_STEPS", 6, minimum=0),
        window_steps=_env_int("DFLASH_BSTNXBT_ADAPTIVE_WINDOW_STEPS", 6, minimum=1),
        grow_acceptance_threshold=grow_acceptance_threshold,
        shrink_acceptance_threshold=shrink_acceptance_threshold,
        grow_step=_env_int("DFLASH_BSTNXBT_ADAPTIVE_GROW_STEP", 1, minimum=1),
        shrink_step=_env_int("DFLASH_BSTNXBT_ADAPTIVE_SHRINK_STEP", 1, minimum=1),
        cooldown_steps=_env_int("DFLASH_BSTNXBT_ADAPTIVE_COOLDOWN_STEPS", 3, minimum=0),
    )
    if not config.enabled:
        config.min_block_tokens = max_default
        config.max_block_tokens = max_default
    return config


def _update_adaptive_block_tokens(
    *,
    state: _AdaptiveBlockState,
    config: _AdaptiveBlockConfig,
    step_index: int,
    accepted_tokens: int,
    drafted_tokens: int,
    max_block_tokens: int,
) -> int:
    if not config.enabled or drafted_tokens <= 0:
        return 0

    max_tokens = max(1, min(int(max_block_tokens), int(config.max_block_tokens)))
    min_tokens = max(1, min(max_tokens, int(config.min_block_tokens)))
    if state.current_block_tokens > max_tokens:
        state.current_block_tokens = max_tokens
    elif state.current_block_tokens < min_tokens:
        state.current_block_tokens = min_tokens

    acceptance_ratio = float(max(0, accepted_tokens)) / float(max(1, drafted_tokens))
    state.acceptance_window.append(acceptance_ratio)
    if len(state.acceptance_window) > config.window_steps:
        state.acceptance_window.pop(0)

    if step_index <= config.warmup_steps:
        return 0
    if len(state.acceptance_window) < config.window_steps:
        return 0
    if state.last_adjust_step > 0 and (step_index - state.last_adjust_step) <= config.cooldown_steps:
        return 0

    window_acceptance = sum(state.acceptance_window) / float(len(state.acceptance_window))
    if window_acceptance >= config.grow_acceptance_threshold and state.current_block_tokens < max_tokens:
        state.current_block_tokens = min(max_tokens, state.current_block_tokens + config.grow_step)
        state.last_adjust_step = int(step_index)
        state.grow_events += 1
        state.acceptance_window.clear()
        return 1

    if window_acceptance <= config.shrink_acceptance_threshold and state.current_block_tokens > min_tokens:
        state.current_block_tokens = max(min_tokens, state.current_block_tokens - config.shrink_step)
        state.last_adjust_step = int(step_index)
        state.shrink_events += 1
        state.acceptance_window.clear()
        return -1

    return 0


@dataclass
class _CollapseWatchdogConfig:
    enabled: bool
    spike_ratio: float
    min_eval_s: float
    warmup_steps: int
    consecutive_spikes: int
    severe_eval_step_s: float
    severe_consecutive_spikes: int
    safe_block_tokens: int
    clear_cache_on_activate: bool
    async_drain_every_steps: int


@dataclass
class _CollapseWatchdogState:
    baseline_eval_s: float = 0.0
    baseline_samples: int = 0
    consecutive_spikes: int = 0
    consecutive_severe_spikes: int = 0
    spike_events: int = 0
    severe_spike_events: int = 0
    safe_mode_active: bool = False
    safe_mode_activations: int = 0
    safe_mode_step: int = -1


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(minimum, int(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(minimum, float(default))
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


@dataclass
class _NumericStat:
    count: int = 0
    total: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    def add(self, value: float) -> None:
        value_f = float(value)
        self.count += 1
        self.total += value_f
        if self.count == 1:
            self.min_value = value_f
            self.max_value = value_f
        else:
            self.min_value = min(self.min_value, value_f)
            self.max_value = max(self.max_value, value_f)

    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / float(self.count)


@dataclass
class _ThermalSidecarState:
    enabled: bool
    sample_interval_s: float
    sample_timeout_s: float
    samples: int = 0
    failures: int = 0
    last_sample_ts_s: float = 0.0
    thermal_warning_samples: int = 0
    performance_warning_samples: int = 0
    cpu_power_status_samples: int = 0
    thermal_warning_level_max: int = 0
    performance_warning_level_max: int = 0
    cpu_power_status_max: int = 0
    cpu_speed_limit_pct: _NumericStat = field(default_factory=_NumericStat)
    gpu_speed_limit_pct: _NumericStat = field(default_factory=_NumericStat)
    cpu_scheduler_limit_pct: _NumericStat = field(default_factory=_NumericStat)
    cpu_available_cpus: _NumericStat = field(default_factory=_NumericStat)


def _parse_pmset_therm_output(raw_output: str) -> dict[str, int | None]:
    values: dict[str, int] = {}
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key_raw, value_raw = line.split("=", 1)
        key = key_raw.strip().lower().replace(" ", "_")
        token = value_raw.strip().split(" ", 1)[0]
        try:
            values[key] = int(token)
        except ValueError:
            continue

    lowered = raw_output.lower()
    if "no thermal warning level has been recorded" in lowered:
        values.setdefault("thermal_warning_level", 0)
    if "no performance warning level has been recorded" in lowered:
        values.setdefault("performance_warning_level", 0)
    if "no cpu power status has been recorded" in lowered:
        values.setdefault("cpu_power_status", 0)

    cpu_power_status = values.get("cpu_power_status")
    if cpu_power_status is None:
        cpu_power_status = values.get("cpu_power_notify")

    return {
        "thermal_warning_level": values.get("thermal_warning_level"),
        "performance_warning_level": values.get("performance_warning_level"),
        "cpu_power_status": cpu_power_status,
        "cpu_speed_limit": values.get("cpu_speed_limit"),
        "gpu_speed_limit": values.get("gpu_speed_limit"),
        "cpu_scheduler_limit": values.get("cpu_scheduler_limit"),
        "cpu_available_cpus": values.get("cpu_available_cpus"),
    }


class _ThermalSidecar:
    def __init__(self, *, enabled: bool, sample_interval_s: float, sample_timeout_s: float):
        self._state = _ThermalSidecarState(
            enabled=bool(enabled),
            sample_interval_s=max(0.2, float(sample_interval_s)),
            sample_timeout_s=max(0.1, float(sample_timeout_s)),
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._state.enabled

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dflash-thermal-sidecar",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=max(1.0, self._state.sample_timeout_s * 2.0))
        self._thread = None

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            state = self._state
            samples = int(state.samples)
            failures = int(state.failures)
            thermal_warning_samples = int(state.thermal_warning_samples)
            performance_warning_samples = int(state.performance_warning_samples)
            cpu_power_status_samples = int(state.cpu_power_status_samples)
            thermal_warning_level_max = int(state.thermal_warning_level_max)
            performance_warning_level_max = int(state.performance_warning_level_max)
            cpu_power_status_max = int(state.cpu_power_status_max)
            last_sample_ts_s = float(state.last_sample_ts_s)
            cpu_speed_limit = state.cpu_speed_limit_pct
            gpu_speed_limit = state.gpu_speed_limit_pct
            cpu_scheduler_limit = state.cpu_scheduler_limit_pct
            cpu_available = state.cpu_available_cpus

        last_sample_age_s = 0.0
        if last_sample_ts_s > 0.0:
            last_sample_age_s = max(0.0, time.perf_counter() - last_sample_ts_s)

        return {
            "thermal_sidecar_enabled": int(1 if state.enabled else 0),
            "thermal_sidecar_sample_interval_s": float(state.sample_interval_s),
            "thermal_sidecar_sample_timeout_s": float(state.sample_timeout_s),
            "thermal_sidecar_samples": int(samples),
            "thermal_sidecar_failures": int(failures),
            "thermal_sidecar_last_sample_age_s": float(last_sample_age_s),
            "thermal_sidecar_thermal_warning_samples": int(thermal_warning_samples),
            "thermal_sidecar_performance_warning_samples": int(performance_warning_samples),
            "thermal_sidecar_cpu_power_status_samples": int(cpu_power_status_samples),
            "thermal_sidecar_thermal_warning_level_max": int(thermal_warning_level_max),
            "thermal_sidecar_performance_warning_level_max": int(performance_warning_level_max),
            "thermal_sidecar_cpu_power_status_max": int(cpu_power_status_max),
            "thermal_sidecar_cpu_speed_limit_samples": int(cpu_speed_limit.count),
            "thermal_sidecar_cpu_speed_limit_mean_pct": float(cpu_speed_limit.mean()),
            "thermal_sidecar_cpu_speed_limit_min_pct": float(cpu_speed_limit.min_value),
            "thermal_sidecar_cpu_speed_limit_max_pct": float(cpu_speed_limit.max_value),
            "thermal_sidecar_gpu_speed_limit_samples": int(gpu_speed_limit.count),
            "thermal_sidecar_gpu_speed_limit_mean_pct": float(gpu_speed_limit.mean()),
            "thermal_sidecar_gpu_speed_limit_min_pct": float(gpu_speed_limit.min_value),
            "thermal_sidecar_gpu_speed_limit_max_pct": float(gpu_speed_limit.max_value),
            "thermal_sidecar_cpu_scheduler_limit_samples": int(cpu_scheduler_limit.count),
            "thermal_sidecar_cpu_scheduler_limit_mean_pct": float(cpu_scheduler_limit.mean()),
            "thermal_sidecar_cpu_scheduler_limit_min_pct": float(cpu_scheduler_limit.min_value),
            "thermal_sidecar_cpu_scheduler_limit_max_pct": float(cpu_scheduler_limit.max_value),
            "thermal_sidecar_cpu_available_samples": int(cpu_available.count),
            "thermal_sidecar_cpu_available_mean": float(cpu_available.mean()),
            "thermal_sidecar_cpu_available_min": float(cpu_available.min_value),
            "thermal_sidecar_cpu_available_max": float(cpu_available.max_value),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            now_s = time.perf_counter()
            with self._lock:
                if sample is None:
                    self._state.failures += 1
                else:
                    self._state.samples += 1
                    self._state.last_sample_ts_s = now_s
                    self._merge_sample(sample)
            self._stop_event.wait(self._state.sample_interval_s)

    def _sample_once(self) -> dict[str, int | None] | None:
        try:
            proc = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=self._state.sample_timeout_s,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

        output = (proc.stdout or "").strip()
        err_output = (proc.stderr or "").strip()
        merged = output
        if err_output:
            merged = f"{merged}\n{err_output}" if merged else err_output
        if not merged:
            return None
        if proc.returncode != 0 and not output:
            return None
        return _parse_pmset_therm_output(merged)

    def _merge_sample(self, sample: dict[str, int | None]) -> None:
        thermal_warning_level = sample.get("thermal_warning_level")
        if isinstance(thermal_warning_level, int):
            if thermal_warning_level > 0:
                self._state.thermal_warning_samples += 1
            self._state.thermal_warning_level_max = max(
                self._state.thermal_warning_level_max,
                int(thermal_warning_level),
            )

        performance_warning_level = sample.get("performance_warning_level")
        if isinstance(performance_warning_level, int):
            if performance_warning_level > 0:
                self._state.performance_warning_samples += 1
            self._state.performance_warning_level_max = max(
                self._state.performance_warning_level_max,
                int(performance_warning_level),
            )

        cpu_power_status = sample.get("cpu_power_status")
        if isinstance(cpu_power_status, int):
            if cpu_power_status > 0:
                self._state.cpu_power_status_samples += 1
            self._state.cpu_power_status_max = max(
                self._state.cpu_power_status_max,
                int(cpu_power_status),
            )

        cpu_speed_limit = sample.get("cpu_speed_limit")
        if isinstance(cpu_speed_limit, int):
            self._state.cpu_speed_limit_pct.add(cpu_speed_limit)

        gpu_speed_limit = sample.get("gpu_speed_limit")
        if isinstance(gpu_speed_limit, int):
            self._state.gpu_speed_limit_pct.add(gpu_speed_limit)

        cpu_scheduler_limit = sample.get("cpu_scheduler_limit")
        if isinstance(cpu_scheduler_limit, int):
            self._state.cpu_scheduler_limit_pct.add(cpu_scheduler_limit)

        cpu_available_cpus = sample.get("cpu_available_cpus")
        if isinstance(cpu_available_cpus, int):
            self._state.cpu_available_cpus.add(cpu_available_cpus)


def _resolve_collapse_watchdog_config(effective_block_tokens: int) -> _CollapseWatchdogConfig:
    enabled = os.environ.get("DFLASH_BSTNXBT_COLLAPSE_WATCHDOG", "1") == "1"
    safe_block_tokens_default = max(1, min(int(effective_block_tokens), 13))
    safe_block_tokens = _env_int(
        "DFLASH_BSTNXBT_COLLAPSE_SAFE_BLOCK_TOKENS",
        safe_block_tokens_default,
        minimum=0,
    )
    if safe_block_tokens > 0:
        safe_block_tokens = max(1, min(int(effective_block_tokens), safe_block_tokens))
    return _CollapseWatchdogConfig(
        enabled=enabled,
        spike_ratio=_env_float("DFLASH_BSTNXBT_COLLAPSE_SPIKE_RATIO", 1.7, minimum=1.05),
        min_eval_s=_env_float("DFLASH_BSTNXBT_COLLAPSE_MIN_EVAL_S", 0.10, minimum=0.0),
        warmup_steps=_env_int("DFLASH_BSTNXBT_COLLAPSE_WARMUP_STEPS", 6, minimum=0),
        consecutive_spikes=_env_int("DFLASH_BSTNXBT_COLLAPSE_CONSECUTIVE_SPIKES", 3, minimum=1),
        severe_eval_step_s=_env_float("DFLASH_BSTNXBT_COLLAPSE_SEVERE_EVAL_STEP_S", 0.20, minimum=0.0),
        severe_consecutive_spikes=_env_int("DFLASH_BSTNXBT_COLLAPSE_SEVERE_CONSECUTIVE", 2, minimum=1),
        safe_block_tokens=safe_block_tokens,
        clear_cache_on_activate=os.environ.get("DFLASH_BSTNXBT_COLLAPSE_CLEAR_CACHE", "1") == "1",
        async_drain_every_steps=_env_int("DFLASH_BSTNXBT_ASYNC_DRAIN_EVERY_STEPS", 8, minimum=0),
    )


def _observe_eval_step(
    *,
    state: _CollapseWatchdogState,
    config: _CollapseWatchdogConfig,
    step_index: int,
    step_eval_s: float,
) -> bool:
    if step_eval_s <= 0.0:
        return False

    baseline = state.baseline_eval_s if state.baseline_samples > 0 else step_eval_s
    if baseline <= 0.0:
        baseline = step_eval_s

    is_spike = False
    if step_index > config.warmup_steps:
        threshold = max(config.min_eval_s, baseline * config.spike_ratio)
        is_spike = step_eval_s >= threshold

    severe_spike = config.severe_eval_step_s > 0.0 and step_eval_s >= config.severe_eval_step_s
    if is_spike or severe_spike:
        state.spike_events += 1
        state.consecutive_spikes += 1
    else:
        state.consecutive_spikes = 0

    if severe_spike:
        state.severe_spike_events += 1
        state.consecutive_severe_spikes += 1
    else:
        state.consecutive_severe_spikes = 0

    sample_for_baseline = step_eval_s
    if step_index > config.warmup_steps and baseline > 0.0:
        sample_for_baseline = min(sample_for_baseline, baseline * config.spike_ratio)

    if state.baseline_samples == 0:
        state.baseline_eval_s = sample_for_baseline
    else:
        alpha = 0.15
        state.baseline_eval_s = ((1.0 - alpha) * state.baseline_eval_s) + (alpha * sample_for_baseline)
    state.baseline_samples += 1

    if not state.safe_mode_active and (
        state.consecutive_spikes >= config.consecutive_spikes
        or state.consecutive_severe_spikes >= config.severe_consecutive_spikes
    ):
        state.safe_mode_active = True
        state.safe_mode_activations += 1
        state.safe_mode_step = int(step_index)
        return True
    return False



def _forward_linear_with_tape(
    layer: Any,
    hidden_states: mx.array,
    mask: Optional[mx.array],
    cache: RecurrentRollbackCache,
) -> mx.array:
    linear = layer.linear_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    B, S, _ = inputs.shape

    qkv = linear.in_proj_qkv(inputs)
    z = linear.in_proj_z(inputs).reshape(B, S, linear.num_v_heads, linear.head_v_dim)
    b_raw = linear.in_proj_b(inputs)
    a_raw = linear.in_proj_a(inputs)

    if cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, linear.conv_kernel_size - 1, linear.conv_dim), dtype=inputs.dtype)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
            strict=True,
        )
    ]

    state = cache[1]
    inv_scale = keys.shape[-1] ** -0.5
    queries = (inv_scale**2) * mx.fast.rms_norm(queries, None, 1e-6)
    keys = inv_scale * mx.fast.rms_norm(keys, None, 1e-6)
    g = gated_delta_mod.compute_g(linear.A_log, a_raw, linear.dt_bias)
    beta = mx.sigmoid(b_raw)

    if state is None:
        _, _, h_k, d_k = queries.shape
        h_v, d_v = values.shape[-2:]
        state = mx.zeros((B, h_v, d_v, d_k), dtype=queries.dtype)
    state_in = state

    out, state = gated_delta_mod.gated_delta_ops(queries, keys, values, g, beta, state, mask)
    if getattr(cache, "_armed", False):
        decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
        decayed_state = state_in[:, None, ...] * decay
        kv_mem = (decayed_state * keys[..., None, :]).sum(axis=-1)
        innovation_tape = (values - kv_mem) * beta[..., None]
        cache.record_tape(
            tape=innovation_tape.astype(mx.float32),
            k=keys,
            g=g,
            qkv=qkv,
        )

    cache[1] = state
    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(B, S, -1))
    hidden_states = residual + out

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states



def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: mx.array,
    cache: list[Any],
    capture_layer_ids: set[int],
) -> tuple[mx.array, dict[int, mx.array]]:
    inner = _target_text_model(target_model)
    hidden_states = inner.embed_tokens(input_ids)
    captured: dict[int, mx.array] = {}

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if (layer_index + 1) in capture_layer_ids:
                captured[layer_index + 1] = hidden_states
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            hidden_states = layer(hidden_states, mask, layer_cache)
            if (layer_index + 1) in capture_layer_ids:
                captured[layer_index + 1] = hidden_states

    normalized = inner.norm(hidden_states)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def _concat_hidden_state_chunk_dicts(
    hidden_state_chunks: list[dict[int, mx.array]],
    capture_layer_ids: set[int],
) -> dict[int, mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return {
        layer_id: mx.concatenate([chunk[layer_id] for chunk in hidden_state_chunks], axis=1)
        for layer_id in sorted(capture_layer_ids)
    }


def _verify_target_block(
    *,
    target_model: Any,
    target_forward_fn: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: int | None,
    capture_layer_ids: set[int],
) -> tuple[mx.array, dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        return target_forward_fn(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    logits_chunks: list[mx.array] = []
    hidden_state_chunks: list[dict[int, mx.array]] = []
    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_hidden_states = target_forward_fn(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(chunk_logits)
        hidden_state_chunks.append(chunk_hidden_states)

    return (
        mx.concatenate(logits_chunks, axis=1),
        _concat_hidden_state_chunk_dicts(hidden_state_chunks, capture_layer_ids),
    )



def _arm_target_rollback(cache_entries: list[Any]) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback()



@dataclass
class _CacheRestoreStats:
    rollback_calls: int = 0
    trim_calls: int = 0
    trim_tokens: int = 0
    full_accept_clears: int = 0



def _restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int,
) -> _CacheRestoreStats:
    stats = _CacheRestoreStats()
    fully_accepted = drafted_tokens > 0 and acceptance_length == drafted_tokens
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                cache_entry._armed = False
                cache_entry._tape = None
                cache_entry._tape_k = None
                cache_entry._tape_g = None
                cache_entry._tape_qkv = None
                cache_entry._snapshot = None
                stats.full_accept_clears += 1
                continue
            cache_entry.rollback(acceptance_length)
            stats.rollback_calls += 1
            continue
        current = int(getattr(cache_entry, "offset", 0) or 0)
        trim = max(0, current - target_len)
        if trim > 0 and hasattr(cache_entry, "trim"):
            cache_entry.trim(trim)
            stats.trim_calls += 1
            stats.trim_tokens += int(trim)
    return stats



def iterate_bstnxbt_mlx_generate_commits(
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
    should_abort: Any | None = None,
    target_forward_mode: str = "default",
    target_forward_fn_override: Any | None = None,
    telemetry_out: dict[str, Any] | None = None,
    copy_output_token_ids: bool = True,
    emit_commits: bool = True,
):
    exact_small_proj_pad_m = _env_int("DFLASH_BSTNXBT_EXACT_SMALL_PROJ_PAD_M", 16, minimum=1)
    install_target_speculative_hooks(target_model, pad_m=exact_small_proj_pad_m)
    configure_full_attention_split(
        target_model,
        enabled=os.environ.get("DFLASH_BSTNXBT_SPLIT_SDPA", "0") == "1",
        chunk_size=int(os.environ.get("DFLASH_BSTNXBT_SPLIT_CHUNK_SIZE", "8")),
    )

    collect_telemetry = telemetry_out is not None
    t_total_start = time.perf_counter() if collect_telemetry else 0.0
    t_prefill_start = time.perf_counter() if collect_telemetry else 0.0

    draft_model = getattr(drafter_model, "model", drafter_model)
    target_layer_ids = list(getattr(draft_model, "target_layer_ids", []) or [])
    if not target_layer_ids:
        raise ValueError("bstnxbt_mlx backend requires draft target_layer_ids")

    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_set = set(int(t) for t in stop_token_ids)
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32)
        if stop_token_ids
        else None
    )
    thinking_state = _init_thinking_budget_state(tokenizer, prompt_tokens, thinking_budget)

    if target_forward_fn_override is not None:
        target_forward_fn = target_forward_fn_override
    elif target_forward_mode == "default":
        target_forward_fn = target_forward_with_hidden_states
    elif target_forward_mode == "mirror_sd_split":
        from .mirror_sd_target import mirror_target_forward_with_hidden_states

        target_forward_fn = mirror_target_forward_with_hidden_states
    else:
        raise ValueError(f"Unsupported target_forward_mode: {target_forward_mode}")

    verify_chunk_tokens = _resolve_verify_chunk_tokens()
    if verify_chunk_tokens is not None and target_forward_mode != "default":
        verify_chunk_tokens = None
    use_speculative_linear_cache = _resolve_speculative_linear_cache_enabled(verify_chunk_tokens)

    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=use_speculative_linear_cache,
    )
    draft_cache = draft_model.make_cache()
    capture_layer_ids = {int(layer_id) + 1 for layer_id in target_layer_ids}

    suppress_mask = build_suppress_token_mask(int(getattr(tokenizer, "vocab_size", 0) or 0), suppress_token_ids)

    prefill_logits, prefill_hidden_states = target_forward_fn(
        target_model,
        input_ids=prompt_array,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    mx.eval(prefill_logits, *prefill_hidden_states.values())
    staged_first = argmax_tokens_with_mask(prefill_logits[:, -1, :], suppress_mask).reshape(-1)
    target_hidden = extract_context_feature_from_dict(prefill_hidden_states, target_layer_ids)

    effective_block_tokens = _resolve_block_tokens(
        draft_model,
        enable_thinking=enable_thinking,
        ignore_eos=bool(suppress_token_ids),
    )
    adaptive_block_config = _resolve_adaptive_block_config(
        initial_block_tokens=effective_block_tokens,
        draft_model=draft_model,
    )
    adaptive_block_state = _AdaptiveBlockState(current_block_tokens=int(effective_block_tokens))
    block_token_buffer = mx.full((int(effective_block_tokens),), int(getattr(draft_model, "mask_token_id", 0)), dtype=mx.uint32)
    verify_len_cap = _resolve_verify_len_cap(effective_block_tokens)
    verify_len_warmup_steps, verify_len_warmup_cap = _resolve_verify_len_warmup(
        block_tokens=effective_block_tokens,
        verify_len_cap=verify_len_cap,
    )
    async_draft_eval = os.environ.get("DFLASH_BSTNXBT_ASYNC_DRAFT_EVAL", "0") == "1"
    fused_draft_verify_eval = os.environ.get("DFLASH_BSTNXBT_FUSED_DRAFT_VERIFY_EVAL", "0") == "1"
    slice_committed_hidden_before_eval = _env_bool(
        "DFLASH_BSTNXBT_SLICE_COMMITTED_HIDDEN_BEFORE_EVAL",
        True,
    )
    watchdog_config = _resolve_collapse_watchdog_config(effective_block_tokens)
    watchdog_state = _CollapseWatchdogState()
    mask_token_id = int(getattr(draft_model, "mask_token_id", 0))
    generated_token_ids: list[int] = []
    generated_token_count = 0
    use_device_output_buffer = bool(not emit_commits and thinking_state is None)
    generated_token_buffer = (
        mx.full((max_new_tokens,), mask_token_id, dtype=mx.uint32)
        if use_device_output_buffer and max_new_tokens > 0
        else None
    )
    start = len(prompt_tokens)

    draft_steps = 0
    drafted_tokens = 0
    accepted_tokens = 0
    committed_tokens = 0
    full_accept_steps = 0
    verify_passes = 0
    verify_warmup_limited_steps = 0
    target_forward_passes = 1  # prefill
    block_tokens_total = 0
    block_tokens_min = 0
    block_tokens_max = 0
    adaptive_block_clamp_events = 0
    collapse_async_drain_calls = 0
    collapse_async_drain_s = 0.0
    collapse_clear_cache_calls = 0
    collapse_safe_sync_calls = 0
    collapse_safe_sync_s = 0.0
    collapse_eval_step_total_s = 0.0
    collapse_eval_step_max_s = 0.0
    collapse_eval_step_min_s = 0.0
    collapse_eval_step_count = 0
    async_submit_calls = 0
    async_submit_to_consume_total_s = 0.0
    async_submit_to_consume_max_s = 0.0
    async_submit_to_consume_min_s = 0.0
    async_submit_to_consume_samples = 0
    async_submit_unconsumed_steps = 0
    draft_submit_step_s = _NumericStat()
    draft_sync_eval_wait_s = _NumericStat()
    verify_submit_step_s = _NumericStat()
    verify_host_gap_s = _NumericStat()
    verify_eval_wait_s = _NumericStat()
    verify_eval_wait_fused_s = _NumericStat()
    verify_eval_wait_unfused_s = _NumericStat()
    verify_eval_wait_unfused_target_posterior_s = _NumericStat()
    verify_eval_wait_unfused_draft_logits_s = _NumericStat()
    verify_eval_fused_steps = 0
    verify_eval_unfused_steps = 0
    cache_restore_s = 0.0
    cache_rollback_calls = 0
    cache_trim_calls = 0
    cache_trim_tokens = 0
    cache_full_accept_clears = 0
    mx_active_mem_total_bytes = 0.0
    mx_active_mem_max_bytes = 0
    mx_cache_mem_total_bytes = 0.0
    mx_cache_mem_max_bytes = 0
    mx_peak_mem_max_bytes = 0
    mx_mem_samples = 0
    mx_recommended_working_set_bytes = 0
    mx_peak_over_recommended_ratio = 0.0
    mx_peak_over_recommended_events = 0
    t_prefill_s = (time.perf_counter() - t_prefill_start) if collect_telemetry else 0.0
    t_draft_s = 0.0
    t_verify_s = 0.0
    t_eval_s = 0.0

    split_sdpa_telemetry: dict[str, int] | None = None
    if collect_telemetry:
        split_sdpa_telemetry = {
            "full_attention_calls": 0,
            "split_path_calls": 0,
            "split_exact_prefix_calls": 0,
            "split_batched_2pass_calls": 0,
            "split_batched_2pass_fallback_calls": 0,
            "split_query_chunks": 0,
        }

        device_info_fn = getattr(mx, "device_info", None)
        if callable(device_info_fn):
            try:
                device_info = device_info_fn()
                mx_recommended_working_set_bytes = int(
                    device_info.get("max_recommended_working_set_size", 0) or 0
                )
            except Exception:
                mx_recommended_working_set_bytes = 0

        reset_peak_memory = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak_memory):
            try:
                reset_peak_memory()
            except Exception:
                pass

    get_active_memory = getattr(mx, "get_active_memory", None)
    get_cache_memory = getattr(mx, "get_cache_memory", None)
    get_peak_memory = getattr(mx, "get_peak_memory", None)

    thermal_sidecar = _ThermalSidecar(
        enabled=collect_telemetry and os.environ.get("DFLASH_BSTNXBT_THERMAL_SIDECAR", "0") == "1",
        sample_interval_s=_env_float("DFLASH_BSTNXBT_THERMAL_SAMPLE_S", 1.0, minimum=0.2),
        sample_timeout_s=_env_float("DFLASH_BSTNXBT_THERMAL_SAMPLE_TIMEOUT_S", 0.5, minimum=0.1),
    )
    thermal_sidecar.start()

    _set_split_sdpa_telemetry_collector(split_sdpa_telemetry)
    try:
        while generated_token_count < max_new_tokens:
            if callable(should_abort) and should_abort():
                break

            if (
                watchdog_config.enabled
                and async_draft_eval
                and watchdog_config.async_drain_every_steps > 0
                and draft_steps > 0
                and (draft_steps % watchdog_config.async_drain_every_steps) == 0
            ):
                t_drain_start = time.perf_counter() if collect_telemetry else 0.0
                mx.eval()
                collapse_async_drain_calls += 1
                if collect_telemetry:
                    drain_elapsed_s = time.perf_counter() - t_drain_start
                    t_eval_s += drain_elapsed_s
                    collapse_async_drain_s += drain_elapsed_s

            remaining = max_new_tokens - generated_token_count
            active_block_tokens = int(effective_block_tokens)
            if adaptive_block_config.enabled:
                active_block_tokens = max(1, min(active_block_tokens, int(adaptive_block_state.current_block_tokens)))
            block_len = max(1, min(active_block_tokens, remaining))
            block_token_buffer[:block_len] = mask_token_id
            block_token_ids = block_token_buffer[:block_len]
            forced_prefix = list((thinking_state.force_queue if thinking_state is not None else [])[:block_len])
            forced_prefix_len = len(forced_prefix)
            if forced_prefix_len > 0:
                block_token_ids[:forced_prefix_len] = mx.array(forced_prefix, dtype=mx.uint32)
            else:
                block_token_ids[:1] = staged_first

            draft_logits = None
            draft_async_submit_ts: float | None = None
            t_draft_start = time.perf_counter() if collect_telemetry else 0.0
            if block_len > 1:
                t_draft_submit_start = time.perf_counter() if collect_telemetry else 0.0
                noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
                if collect_telemetry:
                    draft_submit_step_s.add(time.perf_counter() - t_draft_submit_start)
                if async_draft_eval:
                    mx.async_eval(draft_logits)
                    if collect_telemetry:
                        async_submit_calls += 1
                        draft_async_submit_ts = time.perf_counter()
                if not fused_draft_verify_eval:
                    t_draft_sync_eval_start = time.perf_counter() if collect_telemetry else 0.0
                    mx.eval(draft_logits)
                    if collect_telemetry:
                        draft_sync_eval_wait_elapsed_s = time.perf_counter() - t_draft_sync_eval_start
                        draft_sync_eval_wait_s.add(draft_sync_eval_wait_elapsed_s)
                        verify_eval_wait_unfused_draft_logits_s.add(draft_sync_eval_wait_elapsed_s)
                    if collect_telemetry and draft_async_submit_ts is not None:
                        submit_to_consume_s = time.perf_counter() - draft_async_submit_ts
                        async_submit_to_consume_total_s += submit_to_consume_s
                        async_submit_to_consume_samples += 1
                        if async_submit_to_consume_samples == 1:
                            async_submit_to_consume_min_s = submit_to_consume_s
                            async_submit_to_consume_max_s = submit_to_consume_s
                        else:
                            async_submit_to_consume_min_s = min(async_submit_to_consume_min_s, submit_to_consume_s)
                            async_submit_to_consume_max_s = max(async_submit_to_consume_max_s, submit_to_consume_s)
                        draft_async_submit_ts = None
                drafted = argmax_tokens_with_mask(draft_logits, suppress_mask).squeeze(0)
                fill_start = max(1, forced_prefix_len)
                if fill_start < block_len:
                    drafted_offset = fill_start - 1
                    block_token_ids[fill_start:block_len] = drafted[drafted_offset:block_len - 1]
            if collect_telemetry:
                t_draft_s += time.perf_counter() - t_draft_start

            step_verify_len_cap = verify_len_cap
            if verify_len_warmup_steps > 0 and draft_steps < verify_len_warmup_steps:
                step_verify_len_cap = min(step_verify_len_cap, verify_len_warmup_cap)
                verify_warmup_limited_steps += 1
            verify_token_ids = block_token_ids[: min(block_len, step_verify_len_cap)]
            verify_ids = verify_token_ids[None]
            _arm_target_rollback(target_cache)
            t_verify_start = time.perf_counter() if collect_telemetry else 0.0
            verify_logits, verify_hidden_states = _verify_target_block(
                target_model=target_model,
                target_forward_fn=target_forward_fn,
                verify_ids=verify_ids,
                target_cache=target_cache,
                verify_chunk_tokens=verify_chunk_tokens,
                capture_layer_ids=capture_layer_ids,
            )
            verify_submit_elapsed_s = (time.perf_counter() - t_verify_start) if collect_telemetry else 0.0
            if collect_telemetry:
                t_verify_s += verify_submit_elapsed_s
                verify_submit_step_s.add(verify_submit_elapsed_s)
            posterior = argmax_tokens_with_mask(verify_logits[0], suppress_mask)
            if forced_prefix_len > 0:
                posterior[:forced_prefix_len] = mx.array(forced_prefix, dtype=mx.uint32)
            if slice_committed_hidden_before_eval:
                acceptance_length = int(_match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item())
                commit_count = 1 + acceptance_length
                committed_hidden = extract_context_feature_from_dict(
                    verify_hidden_states,
                    target_layer_ids,
                )[:, :commit_count, :]
            else:
                committed_hidden = extract_context_feature_from_dict(
                    verify_hidden_states,
                    target_layer_ids,
                )

            used_fused_verify_eval = bool(fused_draft_verify_eval and draft_logits is not None)
            measure_eval_timing = collect_telemetry or watchdog_config.enabled
            t_eval_start = time.perf_counter() if measure_eval_timing else 0.0
            if collect_telemetry:
                verify_submit_end_ts = t_verify_start + verify_submit_elapsed_s
                verify_host_gap_s.add(max(0.0, t_eval_start - verify_submit_end_ts))
            if used_fused_verify_eval:
                if collect_telemetry and draft_async_submit_ts is not None:
                    submit_to_consume_s = time.perf_counter() - draft_async_submit_ts
                    async_submit_to_consume_total_s += submit_to_consume_s
                    async_submit_to_consume_samples += 1
                    if async_submit_to_consume_samples == 1:
                        async_submit_to_consume_min_s = submit_to_consume_s
                        async_submit_to_consume_max_s = submit_to_consume_s
                    else:
                        async_submit_to_consume_min_s = min(async_submit_to_consume_min_s, submit_to_consume_s)
                        async_submit_to_consume_max_s = max(async_submit_to_consume_max_s, submit_to_consume_s)
                    draft_async_submit_ts = None
                mx.eval(draft_logits, posterior, committed_hidden)
            else:
                mx.eval(posterior, committed_hidden)
            step_eval_s = (time.perf_counter() - t_eval_start) if measure_eval_timing else 0.0
            if collect_telemetry:
                t_eval_s += step_eval_s
                verify_eval_wait_s.add(step_eval_s)
                if used_fused_verify_eval:
                    verify_eval_fused_steps += 1
                    verify_eval_wait_fused_s.add(step_eval_s)
                else:
                    verify_eval_unfused_steps += 1
                    verify_eval_wait_unfused_s.add(step_eval_s)
                    verify_eval_wait_unfused_target_posterior_s.add(step_eval_s)

            if measure_eval_timing and step_eval_s > 0.0:
                collapse_eval_step_total_s += step_eval_s
                collapse_eval_step_count += 1
                if collapse_eval_step_count == 1:
                    collapse_eval_step_min_s = step_eval_s
                    collapse_eval_step_max_s = step_eval_s
                else:
                    collapse_eval_step_min_s = min(collapse_eval_step_min_s, step_eval_s)
                    collapse_eval_step_max_s = max(collapse_eval_step_max_s, step_eval_s)

            activated_safe_mode = False
            if watchdog_config.enabled and step_eval_s > 0.0:
                activated_safe_mode = _observe_eval_step(
                    state=watchdog_state,
                    config=watchdog_config,
                    step_index=draft_steps + 1,
                    step_eval_s=step_eval_s,
                )

            if activated_safe_mode:
                async_draft_eval = False
                fused_draft_verify_eval = False
                if watchdog_config.safe_block_tokens > 0:
                    effective_block_tokens = max(1, min(effective_block_tokens, watchdog_config.safe_block_tokens))
                    if adaptive_block_config.enabled and adaptive_block_state.current_block_tokens > effective_block_tokens:
                        adaptive_block_state.current_block_tokens = int(effective_block_tokens)
                        adaptive_block_clamp_events += 1
                if watchdog_config.clear_cache_on_activate:
                    metal_mod = getattr(mx, "metal", None)
                    clear_cache = getattr(metal_mod, "clear_cache", None) if metal_mod is not None else None
                    if callable(clear_cache):
                        clear_cache()
                        collapse_clear_cache_calls += 1
                t_safe_sync_start = time.perf_counter() if collect_telemetry else 0.0
                mx.eval()
                if collect_telemetry:
                    safe_sync_elapsed_s = time.perf_counter() - t_safe_sync_start
                    t_eval_s += safe_sync_elapsed_s
                    collapse_safe_sync_calls += 1
                    collapse_safe_sync_s += safe_sync_elapsed_s

            if not slice_committed_hidden_before_eval:
                acceptance_length = int(_match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item())
                commit_count = 1 + acceptance_length
                target_hidden = committed_hidden[:, :commit_count, :]
            else:
                target_hidden = committed_hidden

            committed_token_ids = verify_token_ids[:commit_count]
            committed_segment: list[int] | None = None
            if generated_token_buffer is not None:
                generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_token_ids
                generated_token_count += commit_count
            else:
                committed_segment = [int(tok) for tok in committed_token_ids.tolist()]
                generated_token_ids.extend(committed_segment)
                generated_token_count += commit_count

            forced_committed = min(commit_count, forced_prefix_len)
            start += commit_count
            t_restore_start = time.perf_counter() if collect_telemetry else 0.0
            cache_restore_stats = _restore_target_cache_after_acceptance(
                target_cache,
                target_len=start,
                acceptance_length=acceptance_length,
                drafted_tokens=block_len - 1,
            )
            if collect_telemetry:
                cache_restore_s += time.perf_counter() - t_restore_start
                cache_rollback_calls += int(cache_restore_stats.rollback_calls)
                cache_trim_calls += int(cache_restore_stats.trim_calls)
                cache_trim_tokens += int(cache_restore_stats.trim_tokens)
                cache_full_accept_clears += int(cache_restore_stats.full_accept_clears)

            drafted_tokens_this_step = max(0, block_len - 1)
            draft_steps += 1
            drafted_tokens += drafted_tokens_this_step
            accepted_tokens += max(0, acceptance_length)
            committed_tokens += max(0, commit_count)
            verify_passes += 1
            target_forward_passes += 1
            block_tokens_total += block_len
            if draft_steps == 1:
                block_tokens_min = block_len
                block_tokens_max = block_len
            else:
                block_tokens_min = min(block_tokens_min, block_len)
                block_tokens_max = max(block_tokens_max, block_len)
            if drafted_tokens_this_step > 0 and acceptance_length == drafted_tokens_this_step:
                full_accept_steps += 1

            if adaptive_block_config.enabled:
                adaptive_direction = _update_adaptive_block_tokens(
                    state=adaptive_block_state,
                    config=adaptive_block_config,
                    step_index=draft_steps,
                    accepted_tokens=acceptance_length,
                    drafted_tokens=drafted_tokens_this_step,
                    max_block_tokens=effective_block_tokens,
                )
                if adaptive_direction != 0 and adaptive_block_state.current_block_tokens > effective_block_tokens:
                    adaptive_block_state.current_block_tokens = int(effective_block_tokens)
                    adaptive_block_clamp_events += 1

            if collect_telemetry:
                active_mem_bytes = 0
                cache_mem_bytes = 0
                peak_mem_bytes = 0
                if callable(get_active_memory):
                    try:
                        active_mem_bytes = int(get_active_memory())
                    except Exception:
                        active_mem_bytes = 0
                if callable(get_cache_memory):
                    try:
                        cache_mem_bytes = int(get_cache_memory())
                    except Exception:
                        cache_mem_bytes = 0
                if callable(get_peak_memory):
                    try:
                        peak_mem_bytes = int(get_peak_memory())
                    except Exception:
                        peak_mem_bytes = 0

                mx_mem_samples += 1
                mx_active_mem_total_bytes += float(active_mem_bytes)
                mx_cache_mem_total_bytes += float(cache_mem_bytes)
                mx_active_mem_max_bytes = max(mx_active_mem_max_bytes, active_mem_bytes)
                mx_cache_mem_max_bytes = max(mx_cache_mem_max_bytes, cache_mem_bytes)
                mx_peak_mem_max_bytes = max(mx_peak_mem_max_bytes, peak_mem_bytes)

                if mx_recommended_working_set_bytes > 0 and peak_mem_bytes > 0:
                    peak_ratio = float(peak_mem_bytes) / float(mx_recommended_working_set_bytes)
                    mx_peak_over_recommended_ratio = max(mx_peak_over_recommended_ratio, peak_ratio)
                    if peak_ratio > 1.0:
                        mx_peak_over_recommended_events += 1

                if draft_async_submit_ts is not None:
                    async_submit_unconsumed_steps += 1
                    draft_async_submit_ts = None

            if generated_token_buffer is not None:
                if stop_token_array is not None:
                    hit_stop = bool(
                        mx.any(
                            mx.equal(
                                committed_token_ids[:, None],
                                stop_token_array[None, :],
                            )
                        ).item()
                    )
                else:
                    hit_stop = False
            else:
                hit_stop = bool(committed_segment and any(tok in stop_token_set for tok in committed_segment))
            reached_limit = generated_token_count >= max_new_tokens
            finish_reason = None
            if hit_stop:
                finish_reason = "stop"
            elif reached_limit:
                finish_reason = "length"

            if thinking_state is not None:
                if committed_segment is None:
                    committed_segment = [int(tok) for tok in committed_token_ids.tolist()]
                if forced_committed > 0:
                    del thinking_state.force_queue[:forced_committed]
                _update_thinking_budget_state(thinking_state, committed_segment, forced_committed)

            if emit_commits:
                if committed_segment is None:
                    committed_segment = [int(tok) for tok in committed_token_ids.tolist()]
                if committed_segment:
                    emitted_output_token_ids = (
                        list(generated_token_ids)
                        if copy_output_token_ids
                        else generated_token_ids
                    )
                    yield (
                        committed_segment,
                        emitted_output_token_ids,
                        bool(hit_stop or reached_limit),
                        finish_reason,
                    )

            if hit_stop:
                break
            staged_first = posterior[acceptance_length : acceptance_length + 1]
    finally:
        thermal_sidecar.stop()
        _set_split_sdpa_telemetry_collector(None)

    if generated_token_buffer is not None:
        generated_token_ids = [int(tok) for tok in generated_token_buffer[:generated_token_count].tolist()]

    if collect_telemetry:
        acceptance_rate = (accepted_tokens / drafted_tokens) if drafted_tokens > 0 else 0.0
        full_accept_rate = (full_accept_steps / draft_steps) if draft_steps > 0 else 0.0
        mean_commit_tokens = (committed_tokens / draft_steps) if draft_steps > 0 else 0.0
        block_tokens_mean = (block_tokens_total / draft_steps) if draft_steps > 0 else 0.0
        collapse_eval_step_mean_s = (
            collapse_eval_step_total_s / collapse_eval_step_count
            if collapse_eval_step_count > 0
            else 0.0
        )
        mx_active_mem_mean_bytes = (
            mx_active_mem_total_bytes / float(mx_mem_samples)
            if mx_mem_samples > 0
            else 0.0
        )
        mx_cache_mem_mean_bytes = (
            mx_cache_mem_total_bytes / float(mx_mem_samples)
            if mx_mem_samples > 0
            else 0.0
        )
        split_full_attention_calls = int((split_sdpa_telemetry or {}).get("full_attention_calls", 0))
        split_path_calls = int((split_sdpa_telemetry or {}).get("split_path_calls", 0))
        split_exact_prefix_calls = int((split_sdpa_telemetry or {}).get("split_exact_prefix_calls", 0))
        split_batched_2pass_calls = int((split_sdpa_telemetry or {}).get("split_batched_2pass_calls", 0))
        split_batched_2pass_fallback_calls = int((split_sdpa_telemetry or {}).get("split_batched_2pass_fallback_calls", 0))
        split_query_chunks = int((split_sdpa_telemetry or {}).get("split_query_chunks", 0))
        split_path_hit_rate = (
            float(split_path_calls) / float(split_full_attention_calls)
            if split_full_attention_calls > 0
            else 0.0
        )
        async_submit_to_consume_mean_s = (
            async_submit_to_consume_total_s / float(async_submit_to_consume_samples)
            if async_submit_to_consume_samples > 0
            else 0.0
        )
        thermal_sidecar_stats = thermal_sidecar.snapshot()
        adaptive_window_acceptance_mean = (
            sum(adaptive_block_state.acceptance_window) / float(len(adaptive_block_state.acceptance_window))
            if adaptive_block_state.acceptance_window
            else 0.0
        )

        t_total_s = time.perf_counter() - t_total_start
        telemetry_out.update({
            "draft_steps": int(draft_steps),
            "drafted_tokens": int(drafted_tokens),
            "accepted_tokens": int(accepted_tokens),
            "acceptance_rate": float(acceptance_rate),
            "commit_events": int(draft_steps),
            "mean_commit_tokens": float(mean_commit_tokens),
            "full_accept_steps": int(full_accept_steps),
            "full_accept_rate": float(full_accept_rate),
            "verify_passes": int(verify_passes),
            "verify_len_cap": int(verify_len_cap),
            "verify_len_warmup_steps": int(verify_len_warmup_steps),
            "verify_len_warmup_cap": int(verify_len_warmup_cap),
            "verify_warmup_limited_steps": int(verify_warmup_limited_steps),
            "target_forward_passes": int(target_forward_passes),
            "speculative_linear_cache": int(1 if use_speculative_linear_cache else 0),
            "block_tokens_mean": float(block_tokens_mean),
            "block_tokens_min": int(block_tokens_min),
            "block_tokens_max": int(block_tokens_max),
            "adaptive_block_enabled": int(1 if adaptive_block_config.enabled else 0),
            "adaptive_block_current_tokens": int(adaptive_block_state.current_block_tokens),
            "adaptive_block_min_tokens": int(adaptive_block_config.min_block_tokens),
            "adaptive_block_max_tokens": int(adaptive_block_config.max_block_tokens),
            "adaptive_block_window_steps": int(adaptive_block_config.window_steps),
            "adaptive_block_window_fill": int(len(adaptive_block_state.acceptance_window)),
            "adaptive_block_window_acceptance_mean": float(adaptive_window_acceptance_mean),
            "adaptive_block_grow_events": int(adaptive_block_state.grow_events),
            "adaptive_block_shrink_events": int(adaptive_block_state.shrink_events),
            "adaptive_block_last_adjust_step": int(adaptive_block_state.last_adjust_step),
            "adaptive_block_clamp_events": int(adaptive_block_clamp_events),
            "prefill_s": float(t_prefill_s),
            "draft_s": float(t_draft_s),
            "verify_s": float(t_verify_s),
            "eval_s": float(t_eval_s),
            "total_s": float(t_total_s),
            "cache_restore_s": float(cache_restore_s),
            "cache_rollback_calls": int(cache_rollback_calls),
            "cache_trim_calls": int(cache_trim_calls),
            "cache_trim_tokens": int(cache_trim_tokens),
            "cache_full_accept_clears": int(cache_full_accept_clears),
            "split_full_attention_calls": int(split_full_attention_calls),
            "split_path_calls": int(split_path_calls),
            "split_path_hit_rate": float(split_path_hit_rate),
            "split_exact_prefix_calls": int(split_exact_prefix_calls),
            "split_batched_2pass_calls": int(split_batched_2pass_calls),
            "split_batched_2pass_fallback_calls": int(split_batched_2pass_fallback_calls),
            "split_query_chunks": int(split_query_chunks),
            "async_submit_calls": int(async_submit_calls),
            "async_submit_to_consume_samples": int(async_submit_to_consume_samples),
            "async_submit_to_consume_mean_s": float(async_submit_to_consume_mean_s),
            "async_submit_to_consume_max_s": float(async_submit_to_consume_max_s),
            "async_submit_to_consume_min_s": float(async_submit_to_consume_min_s),
            "async_submit_unconsumed_steps": int(async_submit_unconsumed_steps),
            "draft_submit_steps": int(draft_submit_step_s.count),
            "draft_submit_mean_s": float(draft_submit_step_s.mean()),
            "draft_submit_min_s": float(draft_submit_step_s.min_value),
            "draft_submit_max_s": float(draft_submit_step_s.max_value),
            "draft_sync_eval_wait_steps": int(draft_sync_eval_wait_s.count),
            "draft_sync_eval_wait_mean_s": float(draft_sync_eval_wait_s.mean()),
            "draft_sync_eval_wait_min_s": float(draft_sync_eval_wait_s.min_value),
            "draft_sync_eval_wait_max_s": float(draft_sync_eval_wait_s.max_value),
            "verify_submit_steps": int(verify_submit_step_s.count),
            "verify_submit_mean_s": float(verify_submit_step_s.mean()),
            "verify_submit_min_s": float(verify_submit_step_s.min_value),
            "verify_submit_max_s": float(verify_submit_step_s.max_value),
            "verify_host_gap_steps": int(verify_host_gap_s.count),
            "verify_host_gap_mean_s": float(verify_host_gap_s.mean()),
            "verify_host_gap_min_s": float(verify_host_gap_s.min_value),
            "verify_host_gap_max_s": float(verify_host_gap_s.max_value),
            "verify_eval_wait_steps": int(verify_eval_wait_s.count),
            "verify_eval_wait_mean_s": float(verify_eval_wait_s.mean()),
            "verify_eval_wait_min_s": float(verify_eval_wait_s.min_value),
            "verify_eval_wait_max_s": float(verify_eval_wait_s.max_value),
            "verify_eval_wait_fused_steps": int(verify_eval_wait_fused_s.count),
            "verify_eval_wait_fused_mean_s": float(verify_eval_wait_fused_s.mean()),
            "verify_eval_wait_fused_min_s": float(verify_eval_wait_fused_s.min_value),
            "verify_eval_wait_fused_max_s": float(verify_eval_wait_fused_s.max_value),
            "verify_eval_wait_unfused_steps": int(verify_eval_wait_unfused_s.count),
            "verify_eval_wait_unfused_mean_s": float(verify_eval_wait_unfused_s.mean()),
            "verify_eval_wait_unfused_min_s": float(verify_eval_wait_unfused_s.min_value),
            "verify_eval_wait_unfused_max_s": float(verify_eval_wait_unfused_s.max_value),
            "verify_eval_wait_unfused_target_posterior_steps": int(verify_eval_wait_unfused_target_posterior_s.count),
            "verify_eval_wait_unfused_target_posterior_mean_s": float(verify_eval_wait_unfused_target_posterior_s.mean()),
            "verify_eval_wait_unfused_target_posterior_min_s": float(verify_eval_wait_unfused_target_posterior_s.min_value),
            "verify_eval_wait_unfused_target_posterior_max_s": float(verify_eval_wait_unfused_target_posterior_s.max_value),
            "verify_eval_wait_unfused_draft_logits_steps": int(verify_eval_wait_unfused_draft_logits_s.count),
            "verify_eval_wait_unfused_draft_logits_mean_s": float(verify_eval_wait_unfused_draft_logits_s.mean()),
            "verify_eval_wait_unfused_draft_logits_min_s": float(verify_eval_wait_unfused_draft_logits_s.min_value),
            "verify_eval_wait_unfused_draft_logits_max_s": float(verify_eval_wait_unfused_draft_logits_s.max_value),
            "verify_eval_fused_steps": int(verify_eval_fused_steps),
            "verify_eval_unfused_steps": int(verify_eval_unfused_steps),
            "mx_active_mem_mean_bytes": float(mx_active_mem_mean_bytes),
            "mx_active_mem_max_bytes": int(mx_active_mem_max_bytes),
            "mx_cache_mem_mean_bytes": float(mx_cache_mem_mean_bytes),
            "mx_cache_mem_max_bytes": int(mx_cache_mem_max_bytes),
            "mx_peak_mem_max_bytes": int(mx_peak_mem_max_bytes),
            "mx_recommended_working_set_bytes": int(mx_recommended_working_set_bytes),
            "mx_peak_over_recommended_ratio": float(mx_peak_over_recommended_ratio),
            "mx_peak_over_recommended_events": int(mx_peak_over_recommended_events),
            "collapse_watchdog_enabled": int(1 if watchdog_config.enabled else 0),
            "collapse_spike_events": int(watchdog_state.spike_events),
            "collapse_severe_spike_events": int(watchdog_state.severe_spike_events),
            "collapse_safe_mode_active": int(1 if watchdog_state.safe_mode_active else 0),
            "collapse_safe_mode_activations": int(watchdog_state.safe_mode_activations),
            "collapse_safe_mode_step": int(watchdog_state.safe_mode_step),
            "collapse_safe_mode_block_tokens": int(watchdog_config.safe_block_tokens),
            "collapse_async_drain_calls": int(collapse_async_drain_calls),
            "collapse_async_drain_s": float(collapse_async_drain_s),
            "collapse_clear_cache_calls": int(collapse_clear_cache_calls),
            "collapse_safe_sync_calls": int(collapse_safe_sync_calls),
            "collapse_safe_sync_s": float(collapse_safe_sync_s),
            "collapse_eval_step_mean_s": float(collapse_eval_step_mean_s),
            "collapse_eval_step_max_s": float(collapse_eval_step_max_s),
            "collapse_eval_step_min_s": float(collapse_eval_step_min_s),
        })
        telemetry_out.update(thermal_sidecar_stats)

    return list(generated_token_ids)


def execute_bstnxbt_mlx_generate(
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
    on_commit: Any | None = None,
    should_abort: Any | None = None,
    target_forward_mode: str = "default",
    target_forward_fn_override: Any | None = None,
    telemetry_out: dict[str, Any] | None = None,
) -> list[int]:
    final_tokens: list[int] = []
    if callable(on_commit):
        for committed_segment, generated_token_ids, finished, finish_reason in iterate_bstnxbt_mlx_generate_commits(
            target_model=target_model,
            tokenizer=tokenizer,
            drafter_model=drafter_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            should_abort=should_abort,
            target_forward_mode=target_forward_mode,
            target_forward_fn_override=target_forward_fn_override,
            telemetry_out=telemetry_out,
            copy_output_token_ids=True,
            emit_commits=True,
        ):
            final_tokens = generated_token_ids
            on_commit(committed_segment, generated_token_ids, finished, finish_reason)
    else:
        iterator = iterate_bstnxbt_mlx_generate_commits(
            target_model=target_model,
            tokenizer=tokenizer,
            drafter_model=drafter_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            should_abort=should_abort,
            target_forward_mode=target_forward_mode,
            target_forward_fn_override=target_forward_fn_override,
            telemetry_out=telemetry_out,
            copy_output_token_ids=False,
            emit_commits=False,
        )
        while True:
            try:
                _ = next(iterator)
            except StopIteration as stop:
                stop_value = stop.value
                if stop_value is None:
                    final_tokens = []
                elif isinstance(stop_value, list):
                    final_tokens = stop_value
                else:
                    final_tokens = list(stop_value)
                break

    trimmed = final_tokens[:max_new_tokens]
    stop_token_set = set(int(t) for t in (stop_token_ids or []))
    if stop_token_set:
        for idx, token_id in enumerate(trimmed):
            if token_id in stop_token_set:
                return trimmed[: idx + 1]
    return trimmed
