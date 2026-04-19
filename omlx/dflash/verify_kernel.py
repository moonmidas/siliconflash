from __future__ import annotations

from pathlib import Path
from typing import Any


def get_kernel_source_path() -> Path:
    return Path(__file__).with_name("verify_kernel.metal")


class BatchedGEMVVerifyKernel:
    """Placeholder for the custom Metal M=16 verify kernel wrapper.

    We are intentionally separating the Python call surface from the eventual
    MLX `mx.fast.metal_kernel` / `mx.custom_function` implementation so the
    research loop can stabilize interfaces before low-level optimization lands.
    """

    def __init__(self) -> None:
        self.source_path = get_kernel_source_path()

    def __call__(self, q: Any, k: Any, v: Any, mask: Any | None = None) -> Any:
        raise NotImplementedError(
            "Custom Metal verify kernel not implemented yet. "
            "Use this module as the integration target for deliverable #2."
        )
