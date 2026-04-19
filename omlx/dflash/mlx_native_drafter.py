from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope

from .mlx_port import MLXDFlashDraftConfig, load_hf_dflash_config


class ContextOnlyDraftKVCache:
    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys = None
        self.values = None
        self.offset = 0

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        if self.keys is None:
            self.keys = context_keys
            self.values = context_values
        else:
            self.keys = mx.concatenate([self.keys, context_keys], axis=2)
            self.values = mx.concatenate([self.values, context_values], axis=2)
        self.offset += int(num_positions)
        self._apply_window()

    def _apply_window(self) -> None:
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_k = self.keys[:, :, : self.sink_size, :]
        sink_v = self.values[:, :, : self.sink_size, :]
        window_k = self.keys[:, :, -self.window_size :, :]
        window_v = self.values[:, :, -self.window_size :, :]
        self.keys = mx.concatenate([sink_k, window_k], axis=2)
        self.values = mx.concatenate([sink_v, window_v], axis=2)

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        return self.keys, self.values



def _resolve_context_only_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


@dataclass
class MLXDFlashDraftArgs:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool = False
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 262144

    @property
    def conditioning_input_dim(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


class MLXDFlashAttention(nn.Module):
    def __init__(self, args: MLXDFlashDraftArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states).reshape(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(0, 2, 1, 3)

        k_ctx = self.k_proj(target_hidden).reshape(
            bsz, ctx_len, self.num_key_value_heads, self.head_dim
        )
        k_noise = self.k_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        v_ctx = self.v_proj(target_hidden).reshape(
            bsz, ctx_len, self.num_key_value_heads, self.head_dim
        )
        v_noise = self.v_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        k_ctx = self.k_norm(k_ctx).transpose(0, 2, 1, 3)
        k_noise = self.k_norm(k_noise).transpose(0, 2, 1, 3)
        v_ctx = v_ctx.transpose(0, 2, 1, 3)
        v_noise = v_noise.transpose(0, 2, 1, 3)

        if cache is not None and hasattr(cache, "append_context") and hasattr(cache, "fetch"):
            cache_offset = int(cache.offset)
            query_offset = cache_offset + ctx_len
            q = self.rope(q, offset=query_offset)
            k_ctx = self.rope(k_ctx, offset=cache_offset)
            k_noise = self.rope(k_noise, offset=query_offset)

            cache.append_context(k_ctx, v_ctx, ctx_len)
            cached_keys, cached_values = cache.fetch()
            k = mx.concatenate([cached_keys, k_noise], axis=2)
            v = mx.concatenate([cached_values, v_noise], axis=2)
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                cache=None,
                scale=self.scale,
                mask=None,
            )
        else:
            base_offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=base_offset + ctx_len)
            k_ctx = self.rope(k_ctx, offset=base_offset)
            k_noise = self.rope(k_noise, offset=base_offset + ctx_len)

            k = mx.concatenate([k_ctx, k_noise], axis=2)
            v = mx.concatenate([v_ctx, v_noise], axis=2)

            if cache is not None:
                k, v = cache.update_and_fetch(k, v)

            out = scaled_dot_product_attention(
                q,
                k,
                v,
                cache=cache,
                scale=self.scale,
                mask=None,
            )
        out = out.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(out)


class MLXDFlashDecoderLayer(nn.Module):
    def __init__(self, args: MLXDFlashDraftArgs):
        super().__init__()
        self.self_attn = MLXDFlashAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array, cache: Optional[KVCache] = None) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden=target_hidden, cache=cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MLXDFlashDraftModel(nn.Module):
    def __init__(self, args: MLXDFlashDraftArgs):
        super().__init__()
        self.args = args
        self.layers = [MLXDFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.conditioning_input_dim, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = args.block_size
        self.mask_token_id = args.mask_token_id
        self.target_layer_ids = list(args.target_layer_ids)

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        if cache is None:
            cache = [KVCache() for _ in self.layers]
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, target_hidden=target_hidden, cache=layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> list[Any]:
        if os.environ.get("DFLASH_BSTNXBT_CONTEXT_ONLY_DRAFT_CACHE") == "1":
            sink_size, window_size = _resolve_context_only_draft_window()
            return [
                ContextOnlyDraftKVCache(sink_size=sink_size, window_size=window_size)
                for _ in self.layers
            ]
        return [KVCache() for _ in self.layers]


def config_to_args(config: MLXDFlashDraftConfig) -> MLXDFlashDraftArgs:
    return MLXDFlashDraftArgs(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        target_layer_ids=config.target_layer_ids,
        mask_token_id=config.mask_token_id,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
    )


def _torch_tensor_to_mx(param: Any) -> mx.array:
    import numpy as np

    arr = param.detach().cpu()
    try:
        return mx.array(np.asarray(arr))
    except Exception:
        return mx.array(np.asarray(arr.float()))


def load_mlx_dflash_draft_model(model_path: str) -> MLXDFlashDraftModel:
    from transformers import AutoModel

    cfg = load_hf_dflash_config(model_path)
    model = MLXDFlashDraftModel(config_to_args(cfg))
    hf_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype="auto").eval()

    weights = []
    for name, param in hf_model.state_dict().items():
        weights.append((name, _torch_tensor_to_mx(param)))
    model.load_weights(weights, strict=True)
    return model
