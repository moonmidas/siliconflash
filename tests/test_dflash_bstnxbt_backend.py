from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.config import DFlashConfig
from omlx.dflash.runtime import DFlashRuntime


def test_config_allows_bstnxbt_mlx_backend():
    config = DFlashConfig(draft_backend="bstnxbt_mlx")
    assert config.draft_backend == "bstnxbt_mlx"


def test_runtime_treats_bstnxbt_backend_as_ready_without_bridge_or_verify_kernel():
    config = DFlashConfig(draft_backend="bstnxbt_mlx")
    runtime = DFlashRuntime(
        config=config,
        target_model=SimpleNamespace(),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    assert runtime.ready is True
    assert runtime.availability_reason is None
