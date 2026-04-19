from __future__ import annotations

from typing import Any

from .interfaces import ContextFeatureBundle


class TargetContextFusion:
    """Extract and fuse target hidden states for DFlash conditioning.

    This is a placeholder boundary around the paper's core idea:
    collect hidden states from several uniformly sampled middle/deep layers,
    then fuse them into a persistent context feature for the drafter.

    The implementation is intentionally deferred until we patch the exact oMLX /
    mlx-lm model forward path used by OmniCoder/Qwen3.5.
    """

    def __init__(self, layer_indices: tuple[int, ...], projection_dim: int):
        self.layer_indices = layer_indices
        self.projection_dim = projection_dim

    def extract(self, model: Any, prompt_state: Any) -> ContextFeatureBundle:
        raise NotImplementedError(
            "Target hidden-state extraction is not wired yet. "
            "Patch the target model forward path to expose selected layer states."
        )

    def inject_into_drafter_cache(
        self,
        drafter: Any,
        context_bundle: ContextFeatureBundle,
    ) -> None:
        raise NotImplementedError(
            "Drafter KV/context injection is not wired yet. "
            "Implement after selecting the MLX drafter runtime shape contract."
        )
