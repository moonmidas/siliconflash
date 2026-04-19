from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import DFlashConfig
from .interfaces import ContextFeatureBundle, DraftBlock

logger = logging.getLogger(__name__)


@dataclass
class ExternalDFlashModel:
    """Container for an externally loaded z-lab DFlash drafter."""

    model: Any
    backend: str
    model_path: str

    @property
    def supports_spec_generate(self) -> bool:
        return callable(getattr(self.model, "spec_generate", None))


def load_zlab_dflash_model(model_path: str) -> ExternalDFlashModel:
    """Load a z-lab DFlash drafter via the upstream Transformers interface."""
    try:
        from transformers import AutoModel
    except Exception as e:
        raise RuntimeError(
            "Loading z-lab DFlash drafter requires transformers + torch in this environment"
        ) from e

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
    ).eval()
    return ExternalDFlashModel(
        model=model,
        backend="zlab_spec_generate",
        model_path=model_path,
    )


@dataclass
class NativeMLXDFlashModel:
    model: Any
    backend: str
    model_path: str


def load_external_bstnxbt_dflash_model(model_path: str) -> NativeMLXDFlashModel:
    external_root = Path(__file__).resolve().parents[2] / "external" / "dflash-mlx-bstnxbt"
    external_root_str = str(external_root)
    inserted = False
    if external_root_str not in sys.path:
        sys.path.insert(0, external_root_str)
        inserted = True
    try:
        from dflash_mlx.model import ContextOnlyDraftKVCache
        from dflash_mlx.runtime import _resolve_draft_window, load_draft_bundle

        model, _ = load_draft_bundle(model_path, lazy=False)
        sink_size, window_size = _resolve_draft_window()
        model.make_cache = lambda: [
            ContextOnlyDraftKVCache(sink_size=sink_size, window_size=window_size)
            for _ in range(len(model.layers))
        ]
        return NativeMLXDFlashModel(
            model=model,
            backend="bstnxbt_external_mlx",
            model_path=model_path,
        )
    finally:
        if inserted:
            try:
                sys.path.remove(external_root_str)
            except ValueError:
                pass



def load_native_mlx_dflash_model(model_path: str) -> NativeMLXDFlashModel:
    if (
        os.environ.get("DFLASH_BSTNXBT_RUNTIME") == "1"
        and os.environ.get("DFLASH_BSTNXBT_EXTERNAL_DRAFT") == "1"
    ):
        return load_external_bstnxbt_dflash_model(model_path)

    from .mlx_native_drafter import load_mlx_dflash_draft_model

    model = load_mlx_dflash_draft_model(model_path)
    return NativeMLXDFlashModel(
        model=model,
        backend="mlx_native_hybrid",
        model_path=model_path,
    )


def _extract_context_feature(hidden_states: list[Any], layer_ids: list[int] | tuple[int, ...] | None):
    import torch

    if not layer_ids:
        raise ValueError("DFlash drafter requires target_layer_ids for context extraction")
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def _sample(logits, temperature: float = 0.0):
    import torch

    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def execute_zlab_spec_generate(
    draft_model: Any,
    *,
    target: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
):
    """Run the public z-lab DFlash decoding loop against an adapted target."""
    import torch
    from transformers.cache_utils import DynamicCache

    draft_model.eval()
    draft_dtype = next(draft_model.parameters()).dtype
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    block_size = int(draft_model.block_size)
    output_ids = torch.full(
        (1, max_length + block_size),
        draft_model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    if getattr(output, "sampled_token_ids", None) is not None and temperature < 1e-5:
        output_ids[:, num_input_tokens : num_input_tokens + 1] = output.sampled_token_ids[:, -1:]
    else:
        output_ids[:, num_input_tokens : num_input_tokens + 1] = _sample(output.logits, temperature)
    if hasattr(target, "extract_context_feature"):
        target_hidden = target.extract_context_feature(output.hidden_states, draft_model.target_layer_ids).to(draft_dtype)
    else:
        target_hidden = _extract_context_feature(output.hidden_states, draft_model.target_layer_ids).to(draft_dtype)

    start = input_ids.shape[1]
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids).to(draft_dtype)
        draft_hidden = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = _sample(draft_logits, temperature)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        if getattr(output, "sampled_token_ids", None) is not None and temperature < 1e-5:
            posterior = output.sampled_token_ids
        else:
            posterior = _sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if hasattr(target, "extract_context_feature"):
            target_hidden = target.extract_context_feature(output.hidden_states, draft_model.target_layer_ids)[:, : acceptance_length + 1, :].to(draft_dtype)
        else:
            target_hidden = _extract_context_feature(output.hidden_states, draft_model.target_layer_ids)[:, : acceptance_length + 1, :].to(draft_dtype)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != draft_model.mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    return output_ids


def _trim_mlx_kv_caches(caches: list[Any], keep_tokens: int) -> None:
    for cache in caches:
        trim = max(0, int(getattr(cache, 'offset', 0)) - keep_tokens)
        if trim > 0 and hasattr(cache, 'trim'):
            cache.trim(trim)


def execute_mlx_hybrid_spec_generate_native_ids(
    draft_model: Any,
    *,
    target: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
):
    import os
    import torch

    from .boundary_refresh import (
        parse_int_schedule_env,
        parse_layer_id_set_env,
        refresh_linear_ssm_from_native_text_fresh,
        restore_linear_boundary,
        set_boundary_capture_enabled,
        trim_attention_reject_suffix,
    )

    if temperature >= 1e-5:
        raise ValueError("MLX hybrid DFlash path currently supports greedy decoding only")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = int(draft_model.block_size)
    input_ids_mx = target._torch_to_mx_ids(input_ids)
    output_ids = mx.full((1, max_length + block_size), draft_model.mask_token_id, dtype=mx.int32)
    target_cache = target.make_cache()
    draft_cache = draft_model.make_cache()

    sampled_token_ids, selected_states = target.forward_mx(
        input_ids_mx,
        target_cache,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    draft_dtype = getattr(getattr(draft_model, "fc", None), "weight", None)
    draft_dtype = getattr(draft_dtype, "dtype", mx.float32)

    output_ids[:, :num_input_tokens] = input_ids_mx
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sampled_token_ids[:, -1:]
    target_hidden = mx.concatenate(selected_states, axis=-1).astype(draft_dtype)

    mask_noise_embedding = None
    if block_size > 1:
        mask_ids = mx.full((1, block_size - 1), draft_model.mask_token_id, dtype=mx.int32)
        mask_noise_embedding = target._text_model.embed_tokens(mask_ids).astype(draft_dtype)

    boundary_enabled = os.environ.get("DFLASH_BOUNDARY_SNAPSHOT") == "1"
    refresh_layer_ids = parse_layer_id_set_env("DFLASH_REFRESH_LINEAR_LAYER_IDS")
    reanchor_schedule = parse_int_schedule_env("DFLASH_REANCHOR_ATS")
    if not reanchor_schedule:
        reanchor_single = os.environ.get("DFLASH_NATIVE_FRESH_LINEAR_REANCHOR_AT")
        if reanchor_single:
            reanchor_schedule = [int(reanchor_single)]
    greedy_cutoff = int(os.environ.get("DFLASH_BOUNDARY_GREEDY_CUTOFF_AT", "-1"))
    stop_refresh_at_max_layer = os.environ.get("DFLASH_REFRESH_NATIVE_TEXT_STOP_AT_MAX_LAYER") == "1"
    prompt_ids = input_ids[0].tolist()
    produced = [int(sampled_token_ids[0, -1].item())]
    text_layers = target.target_model.language_model.model.layers

    start = num_input_tokens
    while start < max_length:
        next_reanchor = reanchor_schedule[0] if reanchor_schedule else None
        if next_reanchor is not None and len(produced) >= next_reanchor:
            refresh_linear_ssm_from_native_text_fresh(
                model=target.target_model,
                live_cache=target_cache,
                prompt_ids=prompt_ids,
                produced=produced,
                refresh_layer_ids=refresh_layer_ids,
                stop_at_max_layer=stop_refresh_at_max_layer,
            )
            reanchor_schedule.pop(0)

        if greedy_cutoff >= 0 and len(produced) >= greedy_cutoff:
            tid = mx.array([[produced[-1]]], dtype=mx.int32)
            next_token_ids, _ = target.forward_mx(tid, target_cache, logits_to_keep=1, output_hidden_states=False)
            next_token = int(next_token_ids[0, -1].item())
            output_ids[:, start] = tid[:, 0]
            output_ids[:, start + 1] = next_token_ids[:, -1]
            produced.append(next_token)
            start += 1
            if stop_token_ids is not None and next_token in stop_token_ids:
                break
            continue

        block_output_ids = mx.array(output_ids[:, start : start + block_size])
        last_token_embedding = target._text_model.embed_tokens(block_output_ids[:, :1]).astype(draft_dtype)
        if mask_noise_embedding is not None:
            noise_embedding = mx.concatenate([last_token_embedding, mask_noise_embedding], axis=1)
        else:
            noise_embedding = last_token_embedding
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_hidden_tail = draft_hidden[:, -block_size + 1 :, :]
        draft_ids = target.sample_from_hidden_mx(draft_hidden_tail)
        block_output_ids[:, 1:] = draft_ids
        _trim_mlx_kv_caches(draft_cache, start)

        if boundary_enabled:
            set_boundary_capture_enabled(target_cache, text_layers, enabled=True)
        posterior, selected_states = target.forward_mx(
            block_output_ids,
            target_cache,
            logits_to_keep=None,
            output_hidden_states=True,
        )
        if boundary_enabled:
            set_boundary_capture_enabled(target_cache, text_layers, enabled=False)

        draft_list = draft_ids[0].tolist()
        posterior_list = posterior[0].tolist()
        acceptance_length = 0
        for pred, ref in zip(draft_list, posterior_list[:-1]):
            if pred == ref:
                acceptance_length += 1
            else:
                break
        accepted_tokens_in_block = acceptance_length + 1
        reject_tokens = block_size - accepted_tokens_in_block
        output_ids[:, start : start + accepted_tokens_in_block] = block_output_ids[:, :accepted_tokens_in_block]
        output_ids[:, start + accepted_tokens_in_block] = posterior[:, acceptance_length]
        produced.extend([int(x) for x in draft_list[:acceptance_length]])
        bonus = int(posterior_list[acceptance_length])
        produced.append(bonus)
        start += accepted_tokens_in_block
        if boundary_enabled:
            restore_linear_boundary(target_cache, text_layers, accepted_tokens_in_block)
            trim_attention_reject_suffix(target_cache, text_layers, reject_tokens)
        else:
            _trim_mlx_kv_caches(target_cache, start)
        target_hidden = mx.concatenate(selected_states, axis=-1)[:, :accepted_tokens_in_block, :].astype(draft_dtype)
        if stop_token_ids is not None and any(stop_token_id in produced for stop_token_id in stop_token_ids):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != draft_model.mask_token_id]
    if stop_token_ids is not None:
        stop_set = set(stop_token_ids)
        trimmed = output_ids[0, num_input_tokens:].tolist()
        stop_idx = next((idx for idx, tok in enumerate(trimmed) if tok in stop_set), None)
        if stop_idx is not None:
            output_ids = output_ids[:, : num_input_tokens + stop_idx + 1]
    return target._mx_to_torch(output_ids, torch=torch)


def execute_mlx_hybrid_spec_generate(
    draft_model: Any,
    *,
    target: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
):
    import os
    import torch
    from transformers.cache_utils import DynamicCache

    if os.environ.get("DFLASH_HYBRID_NATIVE_IDS") == "1":
        return execute_mlx_hybrid_spec_generate_native_ids(
            draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    if temperature >= 1e-5:
        raise ValueError("MLX hybrid DFlash path currently supports greedy decoding only")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = int(draft_model.block_size)
    output_ids = torch.full(
        (1, max_length + block_size),
        draft_model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = draft_model.make_cache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    draft_dtype = getattr(getattr(draft_model, "fc", None), "weight", None)
    draft_dtype = getattr(draft_dtype, "dtype", mx.float32)

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = output.sampled_token_ids[:, -1:]
    target_hidden = target._torch_to_mx(target.extract_context_feature(output.hidden_states, draft_model.target_layer_ids)).astype(draft_dtype)

    mask_token_ids = torch.full(
        (1, max(0, block_size - 1)),
        draft_model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    mask_noise_embedding = None
    if block_size > 1:
        mask_noise_embedding = target._torch_to_mx(target.model.embed_tokens(mask_token_ids)).astype(draft_dtype)

    import os

    from .boundary_refresh import (
        parse_int_schedule_env,
        parse_layer_id_set_env,
        refresh_linear_ssm_from_native_text_fresh,
        restore_linear_boundary,
        set_boundary_capture_enabled,
        trim_attention_reject_suffix,
    )

    boundary_enabled = os.environ.get("DFLASH_BOUNDARY_SNAPSHOT") == "1"
    refresh_layer_ids = parse_layer_id_set_env("DFLASH_REFRESH_LINEAR_LAYER_IDS")
    reanchor_schedule = parse_int_schedule_env("DFLASH_REANCHOR_ATS")
    if not reanchor_schedule:
        reanchor_single = os.environ.get("DFLASH_NATIVE_FRESH_LINEAR_REANCHOR_AT")
        if reanchor_single:
            reanchor_schedule = [int(reanchor_single)]
    greedy_cutoff = int(os.environ.get("DFLASH_BOUNDARY_GREEDY_CUTOFF_AT", "-1"))
    stop_refresh_at_max_layer = os.environ.get("DFLASH_REFRESH_NATIVE_TEXT_STOP_AT_MAX_LAYER") == "1"
    prompt_ids = input_ids[0].tolist()
    produced = [int(output.sampled_token_ids[0, -1].item())]
    text_layers = target.target_model.language_model.model.layers

    start = input_ids.shape[1]
    while start < max_length:
        next_reanchor = reanchor_schedule[0] if reanchor_schedule else None
        if next_reanchor is not None and len(produced) >= next_reanchor:
            refresh_linear_ssm_from_native_text_fresh(
                model=target.target_model,
                live_cache=past_key_values_target._mlx_cache,
                prompt_ids=prompt_ids,
                produced=produced,
                refresh_layer_ids=refresh_layer_ids,
                stop_at_max_layer=stop_refresh_at_max_layer,
            )
            reanchor_schedule.pop(0)

        if greedy_cutoff >= 0 and len(produced) >= greedy_cutoff:
            tid = torch.tensor([[produced[-1]]], dtype=torch.long, device=target.device)
            pos = num_input_tokens + len(produced) - 1
            out_greedy = target(
                tid,
                position_ids=position_ids[:, pos : pos + 1],
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=False,
            )
            next_token = int(out_greedy.sampled_token_ids[0, -1].item())
            output_ids[:, start] = tid[:, 0]
            output_ids[:, start + 1] = out_greedy.sampled_token_ids[:, -1]
            produced.append(next_token)
            start += 1
            if stop_token_ids is not None and next_token in stop_token_ids:
                break
            continue

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        last_token_embedding = target._torch_to_mx(target.model.embed_tokens(block_output_ids[:, :1])).astype(draft_dtype)
        if mask_noise_embedding is not None:
            noise_embedding = mx.concatenate([last_token_embedding, mask_noise_embedding], axis=1)
        else:
            noise_embedding = last_token_embedding
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=past_key_values_draft,
        )
        draft_hidden_tail = draft_hidden[:, -block_size + 1 :, :]
        draft_ids = target.sample_from_hidden_mx(draft_hidden_tail)
        block_output_ids[:, 1:] = target._mx_to_torch(draft_ids, torch=torch)
        _trim_mlx_kv_caches(past_key_values_draft, start)

        if boundary_enabled:
            set_boundary_capture_enabled(past_key_values_target._mlx_cache, text_layers, enabled=True)
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        if boundary_enabled:
            set_boundary_capture_enabled(past_key_values_target._mlx_cache, text_layers, enabled=False)
        posterior = output.sampled_token_ids
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        accepted_tokens_in_block = acceptance_length + 1
        reject_tokens = block_size - accepted_tokens_in_block
        output_ids[:, start : start + accepted_tokens_in_block] = block_output_ids[:, :accepted_tokens_in_block]
        output_ids[:, start + accepted_tokens_in_block] = posterior[:, acceptance_length]
        produced.extend([int(x) for x in block_output_ids[0, 1:accepted_tokens_in_block].tolist()])
        produced.append(int(posterior[0, acceptance_length].item()))
        start += accepted_tokens_in_block
        if boundary_enabled:
            restore_linear_boundary(
                past_key_values_target._mlx_cache,
                text_layers,
                accepted_tokens_in_block,
            )
            trim_attention_reject_suffix(
                past_key_values_target._mlx_cache,
                text_layers,
                reject_tokens,
            )
        else:
            past_key_values_target.crop(start)
        target_hidden = target._torch_to_mx(
            target.extract_context_feature(output.hidden_states, draft_model.target_layer_ids)[:, :accepted_tokens_in_block, :]
        ).astype(draft_dtype)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != draft_model.mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]
    return output_ids


class DFlashDrafter:
    """Thin wrapper around the block-diffusion draft model."""

    def __init__(self, model: Any, config: DFlashConfig):
        self.model = model
        self.config = config

    @property
    def backend(self) -> str:
        return getattr(self.model, "backend", self.config.draft_backend)

    @property
    def ready(self) -> bool:
        inner = getattr(self.model, "model", self.model)
        if callable(getattr(inner, "spec_generate", None)):
            return True
        return hasattr(inner, "target_layer_ids") and hasattr(inner, "block_size")

    def prime(self, context_bundle: ContextFeatureBundle) -> None:
        raise NotImplementedError(
            "DFlash drafter priming not implemented yet. "
            "Need MLX-native context-feature cache injection."
        )

    def draft_block(self, prefix_state: Any) -> DraftBlock:
        raise NotImplementedError(
            "DFlash block-diffusion draft pass not implemented yet."
        )
