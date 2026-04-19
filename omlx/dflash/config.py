from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


@dataclass
class DFlashConfig:
    """Configuration for native DFlash speculative decoding.

    Notes:
    - Day-1 scope is bf16 + single-request.
    - The default draft model path points at the official Qwen3.5-9B drafter,
      but loading remains optional until the runtime is wired into generation.
    - `conditioning_layer_indices` mirrors the paper's middle/deep hidden-state
      extraction design while allowing model-specific tuning later.
    """

    enabled: bool = False
    draft_model_path: Optional[str] = None
    target_model_name: Optional[str] = None
    draft_backend: Literal["zlab_spec_generate", "bstnxbt_mlx", "mirror_sd_mlx", "ddtree_mlx"] = "zlab_spec_generate"
    block_size: int = 16
    bonus_token: bool = True
    greedy_exact_match: bool = True
    bf16_only: bool = True
    single_batch_only: bool = True
    verify_kernel: str = "metal_batched_gemv"
    use_kernel_replay: bool = True
    use_sync_elision: bool = True
    collect_hidden_states: bool = True
    conditioning_layer_indices: Sequence[int] = field(default_factory=lambda: (4, 8, 12, 16, 20))
    conditioning_projection_dim: int = 1024
    max_verify_chunk_tokens: int = 16
    max_accept_tokens: int = 17
    report_acceptance_rate: bool = True

    # Compatibility gates for later phases.
    allow_quantized_target: bool = False
    allow_continuous_batching: bool = False
    allow_paged_ssd_restore: bool = True
    fallback_to_target_only: bool = True

    def supports_day1_request(self, *, stream: bool, concurrent_requests: int) -> bool:
        if (
            self.single_batch_only
            and concurrent_requests > 1
            and self.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx")
        ):
            return False
        if stream and self.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx"):
            return False
        return True

    def supports_sampling(self, *, temperature: float, top_p: float, top_k: int, min_p: float) -> bool:
        if temperature > 1e-5:
            return False
        if top_k not in (0, 1):
            return False
        if min_p > 1e-8:
            return False
        if top_p < 1.0 - 1e-8:
            return False
        return True

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("DFlash block_size must be > 0")
        if self.max_verify_chunk_tokens < self.block_size:
            raise ValueError(
                "DFlash max_verify_chunk_tokens must be >= block_size"
            )
        if len(tuple(self.conditioning_layer_indices)) == 0:
            raise ValueError("DFlash requires at least one conditioning layer")
