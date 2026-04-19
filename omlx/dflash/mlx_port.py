from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MLXDFlashDraftConfig:
    """Minimal MLX-native view of a z-lab DFlash checkpoint config.

    This is the first building block for eliminating the torch bridge. It keeps
    only the fields the native MLX draft model needs so we can port the 5-layer
    Qwen3.5 DFlash checkpoint incrementally and verify shapes before wiring the
    full generation loop.
    """

    model_path: str
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

    @property
    def conditioning_input_dim(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


@dataclass(frozen=True)
class WeightManifestEntry:
    hf_name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DFlashCheckpointManifest:
    config: MLXDFlashDraftConfig
    weights: tuple[WeightManifestEntry, ...]


def load_hf_dflash_config(model_path: str | Path) -> MLXDFlashDraftConfig:
    model_path = str(model_path)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
    rope_parameters = getattr(cfg, "rope_parameters", {}) or {}

    return MLXDFlashDraftConfig(
        model_path=model_path,
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        num_hidden_layers=int(cfg.num_hidden_layers),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        vocab_size=int(cfg.vocab_size),
        block_size=int(getattr(cfg, "block_size", 16)),
        target_layer_ids=tuple(dflash_cfg.get("target_layer_ids", [])),
        mask_token_id=int(dflash_cfg.get("mask_token_id")),
        rope_theta=float(rope_parameters.get("rope_theta", 10000000)),
        rms_norm_eps=float(cfg.rms_norm_eps),
        attention_bias=bool(getattr(cfg, "attention_bias", False)),
    )


def load_hf_dflash_manifest(model_path: str | Path) -> DFlashCheckpointManifest:
    config = load_hf_dflash_config(model_path)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype="auto").eval()
    weights = tuple(
        WeightManifestEntry(hf_name=name, shape=tuple(param.shape))
        for name, param in model.state_dict().items()
    )
    return DFlashCheckpointManifest(config=config, weights=weights)


def export_manifest_json(model_path: str | Path, output_path: str | Path) -> None:
    manifest = load_hf_dflash_manifest(model_path)
    payload: dict[str, Any] = {
        "config": {
            "model_path": manifest.config.model_path,
            "hidden_size": manifest.config.hidden_size,
            "intermediate_size": manifest.config.intermediate_size,
            "num_hidden_layers": manifest.config.num_hidden_layers,
            "num_attention_heads": manifest.config.num_attention_heads,
            "num_key_value_heads": manifest.config.num_key_value_heads,
            "head_dim": manifest.config.head_dim,
            "vocab_size": manifest.config.vocab_size,
            "block_size": manifest.config.block_size,
            "target_layer_ids": list(manifest.config.target_layer_ids),
            "mask_token_id": manifest.config.mask_token_id,
            "rope_theta": manifest.config.rope_theta,
            "rms_norm_eps": manifest.config.rms_norm_eps,
            "attention_bias": manifest.config.attention_bias,
            "conditioning_input_dim": manifest.config.conditioning_input_dim,
        },
        "weights": [
            {"hf_name": entry.hf_name, "shape": list(entry.shape)}
            for entry in manifest.weights
        ],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
