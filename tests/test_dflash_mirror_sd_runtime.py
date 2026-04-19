from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.mirror_sd_runtime import (
    execute_mirror_sd_mlx_generate,
    iterate_mirror_sd_mlx_generate_commits,
)


def test_execute_mirror_sd_mlx_generate_delegates_to_bstnxbt_with_split_mode(monkeypatch):
    called = {}

    def fake_execute(**kwargs):
        called.update(kwargs)
        return [10, 11]

    monkeypatch.setattr("omlx.dflash.mirror_sd_runtime.execute_bstnxbt_mlx_generate", fake_execute)

    out = execute_mirror_sd_mlx_generate(
        target_model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        drafter_model=SimpleNamespace(),
        prompt="hello",
        max_new_tokens=2,
        stop_token_ids=[77],
        suppress_token_ids=[99],
        enable_thinking=True,
        thinking_budget=128,
    )

    assert out == [10, 11]
    assert called["target_forward_mode"] == "mirror_sd_split"


def test_iterate_mirror_sd_mlx_generate_commits_delegates_to_bstnxbt_with_split_mode(monkeypatch):
    called = {}

    def fake_iterate(**kwargs):
        called.update(kwargs)
        yield [10], [10], True, "length"

    monkeypatch.setattr("omlx.dflash.mirror_sd_runtime.iterate_bstnxbt_mlx_generate_commits", fake_iterate)

    outputs = list(iterate_mirror_sd_mlx_generate_commits(
        target_model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        drafter_model=SimpleNamespace(),
        prompt="hello",
        max_new_tokens=1,
        stop_token_ids=[77],
        suppress_token_ids=[99],
        enable_thinking=False,
        thinking_budget=None,
    ))

    assert outputs == [([10], [10], True, "length")]
    assert called["target_forward_mode"] == "mirror_sd_split"
