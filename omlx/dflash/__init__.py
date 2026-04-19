"""SiliconFlash DFlash integration scaffolding for oMLX.

This package intentionally starts as a thin, non-invasive skeleton so the
workspace can be benchmarked and evolved incrementally.

Day-1 prototype goals:
- OmniCoder-9B / Qwen3.5-9B target compatibility
- single-request bf16 execution path
- exact-match speculative verification metrics
- preserve oMLX scheduler and paged SSD cache contracts

The actual generation path is not wired in yet; these modules define the
interfaces and extension points needed for native integration.
"""

from .config import DFlashConfig
from .drafter import ExternalDFlashModel, load_zlab_dflash_model
from .interfaces import (
    AcceptanceResult,
    ContextFeatureBundle,
    DraftBlock,
    DFlashMetrics,
)
from .mlx_native_drafter import MLXDFlashDraftModel, load_mlx_dflash_draft_model
from .runtime import DFlashRuntime
from .target_bridge import MLXQwenDFlashTargetBridge

__all__ = [
    "AcceptanceResult",
    "ContextFeatureBundle",
    "DFlashConfig",
    "DFlashMetrics",
    "DFlashRuntime",
    "DraftBlock",
    "ExternalDFlashModel",
    "MLXDFlashDraftModel",
    "MLXQwenDFlashTargetBridge",
    "load_mlx_dflash_draft_model",
    "load_zlab_dflash_model",
]
