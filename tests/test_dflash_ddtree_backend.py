from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.config import DFlashConfig
from omlx.dflash.runtime import DFlashRuntime


def test_config_allows_ddtree_mlx_backend():
    config = DFlashConfig(draft_backend="ddtree_mlx")
    assert config.draft_backend == "ddtree_mlx"


def test_runtime_reports_unavailable_when_ddtree_dependency_missing(monkeypatch):
    monkeypatch.setattr(
        "omlx.dflash.runtime.ddtree_runtime_availability_reason",
        lambda: "missing ddtree runtime",
    )

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="ddtree_mlx"),
        target_model=SimpleNamespace(),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    assert runtime.ready is False
    assert runtime.availability_reason == "missing ddtree runtime"


def test_runtime_treats_ddtree_backend_as_ready_when_dependency_available(monkeypatch):
    monkeypatch.setattr(
        "omlx.dflash.runtime.ddtree_runtime_availability_reason",
        lambda: None,
    )

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="ddtree_mlx"),
        target_model=SimpleNamespace(),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    assert runtime.ready is True
    assert runtime.availability_reason is None
