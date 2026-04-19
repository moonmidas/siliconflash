from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.drafter import load_native_mlx_dflash_model


def test_load_native_mlx_dflash_model_uses_external_loader_when_env_enabled(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("DFLASH_BSTNXBT_RUNTIME", "1")
    monkeypatch.setenv("DFLASH_BSTNXBT_EXTERNAL_DRAFT", "1")
    monkeypatch.setattr(
        "omlx.dflash.drafter.load_external_bstnxbt_dflash_model",
        lambda model_path: sentinel,
    )

    loaded = load_native_mlx_dflash_model("z-lab/Qwen3.5-9B-DFlash")

    assert loaded is sentinel


def test_load_native_mlx_dflash_model_uses_in_tree_native_loader_by_default(monkeypatch):
    sentinel_model = object()
    monkeypatch.delenv("DFLASH_BSTNXBT_RUNTIME", raising=False)
    monkeypatch.delenv("DFLASH_BSTNXBT_EXTERNAL_DRAFT", raising=False)
    monkeypatch.setattr(
        "omlx.dflash.mlx_native_drafter.load_mlx_dflash_draft_model",
        lambda model_path: sentinel_model,
    )

    loaded = load_native_mlx_dflash_model("z-lab/Qwen3.5-9B-DFlash")

    assert loaded.model is sentinel_model
    assert loaded.backend == "mlx_native_hybrid"
    assert loaded.model_path == "z-lab/Qwen3.5-9B-DFlash"
