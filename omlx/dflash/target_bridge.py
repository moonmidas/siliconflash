from __future__ import annotations

import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
import mlx_lm.models.qwen3_5 as qwen35
from mlx_lm.models.base import create_attention_mask


def _cache_token_count(cache_obj: Any) -> int:
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)) and sub_caches:
        return max(_cache_token_count(c) for c in sub_caches)
    offset = getattr(cache_obj, "offset", None)
    if isinstance(offset, (int, float)):
        return int(offset)
    size_fn = getattr(cache_obj, "size", None)
    if callable(size_fn):
        try:
            return int(size_fn())
        except Exception:
            return 0
    return 0


def _prompt_cache_length(prompt_cache: list[Any]) -> int:
    if not prompt_cache:
        return 0
    return max(_cache_token_count(c) for c in prompt_cache)


def _trim_cache_obj(cache_obj: Any, trim_tokens: int) -> None:
    if trim_tokens <= 0:
        return
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        for sub in sub_caches:
            _trim_cache_obj(sub, trim_tokens)
        return
    trim_fn = getattr(cache_obj, "trim", None)
    if callable(trim_fn):
        trim_fn(trim_tokens)


class TrimmableArraysCache(ArraysCache):
    """Minimal rollback-capable ArraysCache for Qwen3.5 linear layers."""

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        inst.left_padding = None
        inst.lengths = None
        inst._history = []
        inst._pending_prev = None
        return inst

    def __init__(self, size, left_padding=None, max_history=32):
        super().__init__(size, left_padding=left_padding)
        self.max_history = max_history
        self._history = []
        self._pending_prev = None

    def __setitem__(self, idx, value):
        if self._pending_prev is None and any(c is not None for c in self.cache):
            self._pending_prev = (
                [None if c is None else mx.array(c) for c in self.cache],
                None if self.left_padding is None else mx.array(self.left_padding),
                None if self.lengths is None else mx.array(self.lengths),
            )
        super().__setitem__(idx, value)

    def advance(self, n):
        if n <= self.max_history and self._pending_prev is not None:
            self._history.append(self._pending_prev)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]
        else:
            self._history.clear()
        self._pending_prev = None
        super().advance(n)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, len(self._history))
        if n <= 0:
            return 0
        cache, left_padding, lengths = self._history[-n]
        self.cache = [None if c is None else mx.array(c) for c in cache]
        self.left_padding = None if left_padding is None else mx.array(left_padding)
        self.lengths = None if lengths is None else mx.array(lengths)
        self._history = self._history[:-n]
        self._pending_prev = None
        return n


def make_trimmable_qwen3_5_cache(target_model: Any) -> list[Any]:
    return [
        TrimmableArraysCache(size=2, max_history=64) if layer.is_linear else KVCache()
        for layer in target_model.layers
    ]


@dataclass
class MLXBridgeOutput:
    logits: Any | None = None
    sampled_token_ids: Any | None = None
    hidden_states: list[Any] | None = None


class _EmbedTokensProxy:
    def __init__(self, bridge: "MLXQwenDFlashTargetBridge"):
        self.bridge = bridge

    def __call__(self, input_ids):
        torch = self.bridge._require_torch()
        inputs = self.bridge._torch_to_mx_ids(input_ids)
        embeddings = self.bridge._text_model.embed_tokens(inputs)
        return self.bridge._mx_to_torch(embeddings, torch=torch)


class _LMHeadProxy:
    def __init__(self, bridge: "MLXQwenDFlashTargetBridge"):
        self.bridge = bridge

    def __call__(self, hidden_states):
        torch = self.bridge._require_torch()
        hidden_states_mx = self.bridge._torch_to_mx(hidden_states)
        language_model = self.bridge._language_model
        if self.bridge._args.tie_word_embeddings:
            logits = self.bridge._text_model.embed_tokens.as_linear(hidden_states_mx)
        else:
            logits = language_model.lm_head(hidden_states_mx)
        return self.bridge._mx_to_torch(logits, torch=torch)


class MLXQwenDFlashTargetBridge:
    """Expose an mlx-lm Qwen3.5/OmniCoder target through the z-lab torch API."""

    def __init__(self, target_model: Any, target_layer_ids: list[int] | tuple[int, ...] | None = None):
        self.target_model = target_model
        self._language_model = target_model.language_model
        self._text_model = self._language_model.model
        self._args = self._language_model.args
        self.model = SimpleNamespace(embed_tokens=_EmbedTokensProxy(self))
        self.lm_head = _LMHeadProxy(self)
        self.device = self._device()
        self.target_layer_ids = tuple(target_layer_ids or ())

    def _device(self):
        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"
        return torch.device("cpu")

    @staticmethod
    def availability_reason() -> str | None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return "PyTorch is not installed; z-lab DFlash requires torch/transformers"
        return None

    def _require_torch(self):
        import torch
        return torch

    def _torch_to_mx_ids(self, tensor):
        return self._torch_to_mx(tensor).astype(mx.int32)

    def _torch_to_mx(self, tensor):
        import numpy as np
        tensor = tensor.detach().cpu()
        try:
            return mx.array(np.asarray(tensor))
        except Exception:
            return mx.array(np.asarray(tensor.float()))

    def _mx_to_torch(self, array, *, torch):
        mx.eval(array)
        try:
            if getattr(array, "dtype", None) == mx.bfloat16:
                array = array.astype(mx.float16)
                mx.eval(array)
            return torch.utils.dlpack.from_dlpack(array)
        except Exception:
            import numpy as np
            try:
                np_array = np.asarray(array)
                if str(np_array.dtype) == "bfloat16":
                    np_array = np.asarray(array.astype(mx.float32))
            except Exception:
                np_array = np.array(array.astype(mx.float32).tolist(), dtype=np.float32)
            return torch.from_numpy(np_array)

    def make_cache(self):
        return make_trimmable_qwen3_5_cache(self.target_model)

    def extract_context_feature(self, hidden_states: Any, layer_ids: list[int] | tuple[int, ...] | None):
        import torch

        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
        if hidden_states and isinstance(hidden_states[0], torch.Tensor):
            return torch.cat(hidden_states, dim=-1)
        raise TypeError("Unsupported hidden-state payload for DFlash context extraction")

    def sample_from_hidden_mx(self, hidden_states: mx.array):
        logits = self._text_model.embed_tokens.as_linear(hidden_states) if self._args.tie_word_embeddings else self._language_model.lm_head(hidden_states)
        sampled_token_ids = mx.argmax(logits, axis=-1).astype(mx.int32)
        return sampled_token_ids

    def _ensure_cache_bridge(self, past_key_values):
        if past_key_values is None:
            return self.make_cache()

        cache = getattr(past_key_values, "_mlx_cache", None)
        if cache is None:
            cache = self.make_cache()
            past_key_values._mlx_cache = cache

        if not getattr(past_key_values, "_mlx_bridge_patched", False):
            def crop(this, length):
                current = _prompt_cache_length(this._mlx_cache)
                trim = max(0, current - int(length))
                if trim > 0:
                    for layer_cache in this._mlx_cache:
                        _trim_cache_obj(layer_cache, trim)
                return this

            def get_seq_length(this):
                return _prompt_cache_length(this._mlx_cache)

            past_key_values.crop = types.MethodType(crop, past_key_values)
            past_key_values.get_seq_length = types.MethodType(get_seq_length, past_key_values)
            past_key_values._mlx_bridge_patched = True

        return cache

    def forward_mx(
        self,
        input_ids: mx.array,
        cache: list[Any] | None,
        *,
        logits_to_keep: int | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        hidden_states = self._text_model.embed_tokens(input_ids)
        selected_states = [] if output_hidden_states else None

        if cache is None:
            cache = [None] * len(self._text_model.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self._text_model.fa_idx])
        ssm_mask = qwen35.create_ssm_mask(hidden_states, cache[self._text_model.ssm_idx])

        for idx, (layer, layer_cache) in enumerate(zip(self._text_model.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if selected_states is not None and idx in self.target_layer_ids:
                selected_states.append(hidden_states)

        final_hidden = self._text_model.norm(hidden_states)
        if logits_to_keep is not None:
            final_hidden = final_hidden[:, -int(logits_to_keep) :, :]
        sampled_token_ids = self.sample_from_hidden_mx(final_hidden)
        return sampled_token_ids, selected_states

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=True, logits_to_keep=None, output_hidden_states=False, **_: Any):
        torch = self._require_torch()
        inputs = self._torch_to_mx_ids(input_ids)
        cache = self._ensure_cache_bridge(past_key_values) if use_cache else None
        sampled_token_ids, selected_states = self.forward_mx(
            inputs,
            cache,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
        )
        return MLXBridgeOutput(
            logits=None,
            sampled_token_ids=self._mx_to_torch(sampled_token_ids, torch=torch),
            hidden_states=[self._mx_to_torch(h, torch=torch) for h in selected_states] if selected_states is not None else None,
        )
