from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ContextFeatureBundle:
    """Persistent target-side conditioning features for the drafter.

    `layer_features` is intentionally generic for now because the exact MLX
    tensor shapes depend on the target-model integration point we choose inside
    oMLX / mlx-lm.
    """

    layer_features: list[Any]
    fused_feature: Any
    source_layer_indices: tuple[int, ...]
    seq_len: int
    hidden_size: Optional[int] = None


@dataclass
class DraftBlock:
    """One speculative draft block emitted by the DFlash drafter."""

    token_ids: list[int]
    logits: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceptanceResult:
    """Result of exact-match verification against the target model."""

    accepted_token_ids: list[int]
    rejected_from_index: int
    bonus_token_id: Optional[int]
    target_token_ids: list[int]
    accepted_count: int
    drafted_count: int
    exact_match: bool

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_count <= 0:
            return 0.0
        return self.accepted_count / self.drafted_count


@dataclass
class DFlashMetrics:
    """Runtime counters for benchmarking and autoresearch."""

    draft_steps: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    bonus_tokens: int = 0
    verify_passes: int = 0
    target_forward_passes: int = 0
    drafter_forward_passes: int = 0
    acceptance_events: int = 0

    def record(self, result: AcceptanceResult) -> None:
        self.draft_steps += 1
        self.drafted_tokens += result.drafted_count
        self.accepted_tokens += result.accepted_count
        self.verify_passes += 1
        self.target_forward_passes += 1
        self.drafter_forward_passes += 1
        self.acceptance_events += 1
        if result.bonus_token_id is not None:
            self.bonus_tokens += 1

    @property
    def mean_acceptance_rate(self) -> float:
        if self.drafted_tokens <= 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens
