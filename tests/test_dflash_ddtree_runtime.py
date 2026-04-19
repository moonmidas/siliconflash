from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.ddtree_runtime import (
    ddtree_runtime_availability_reason,
    execute_ddtree_mlx_generate,
)


class _FakeTokenizer:
    def encode(self, prompt):
        assert prompt == "hello"
        return [1, 2, 3]


def test_ddtree_availability_reason_includes_dependency_hint(monkeypatch):
    monkeypatch.setattr(
        "omlx.dflash.ddtree_runtime._resolve_generate_ddtree_once",
        lambda: (None, RuntimeError("import failed")),
    )

    reason = ddtree_runtime_availability_reason()

    assert reason is not None
    assert "ddtree_mlx backend unavailable" in reason
    assert "DFLASH_DDTREE_PATH" in reason


def test_execute_ddtree_generate_delegates_and_populates_telemetry(monkeypatch):
    called = {}
    monkeypatch.setenv("DFLASH_DDTREE_NATIVE_RUNTIME", "0")

    def fake_generate_once(**kwargs):
        called.update(kwargs)
        return {
            "generated_token_ids": [10, 11, 12],
            "elapsed_us": 2_500_000,
            "prefill_us": 100_000,
            "tokens_per_second": 123.4,
            "tree_budget": 7,
            "cycles_completed": 3,
            "ddtree_cycles_completed": 2,
            "dflash_cycles_completed": 1,
            "dflash_accepted_from_draft": 4,
            "avg_acceptance": 2.5,
            "fast_path_ratio": 0.6,
            "fast_path_count": 6,
            "slow_path_count": 4,
            "tree_aware_commit_count": 2,
            "tree_aware_linear": True,
            "exact_commit": False,
            "dflash_controller_enabled": True,
            "dflash_controller_probe_count": 9,
            "dflash_controller_switch_count": 1,
            "phase_timings_us": {
                "draft": 200_000,
                "tree_build": 300_000,
                "tree_verify": 400_000,
                "commit": 100_000,
                "dflash_draft": 50_000,
                "dflash_verify": 60_000,
                "dflash_replay": 70_000,
                "dflash_commit": 80_000,
            },
        }

    monkeypatch.setattr(
        "omlx.dflash.ddtree_runtime._resolve_generate_ddtree_once",
        lambda: (fake_generate_once, None),
    )
    monkeypatch.setattr(
        "omlx.dflash.ddtree_runtime._maybe_patch_mirror_target_forward",
        lambda: None,
    )
    monkeypatch.setenv("DFLASH_DDTREE_BUDGET", "7")

    telemetry = {}
    output_ids = execute_ddtree_mlx_generate(
        target_model=SimpleNamespace(),
        tokenizer=_FakeTokenizer(),
        drafter_model=SimpleNamespace(),
        prompt="hello",
        max_new_tokens=16,
        stop_token_ids=[99],
        suppress_token_ids=[0],
        telemetry_out=telemetry,
    )

    assert output_ids == [10, 11, 12]
    assert called["max_new_tokens"] == 16
    assert called["tree_budget"] == 7
    assert called["prompt_tokens"] == [1, 2, 3]
    assert telemetry["ddtree_enabled"] == 1
    assert telemetry["ddtree_tree_budget"] == 7
    assert telemetry["ddtree_cycles_completed"] == 3
    assert telemetry["ddtree_tokens_per_second"] == 123.4
    assert telemetry["ddtree_tree_verify_s"] == 0.4
    assert telemetry["ddtree_dflash_replay_s"] == 0.07
    assert telemetry["eval_s"] == 2.5


def test_execute_ddtree_generate_uses_native_runtime_when_enabled(monkeypatch):
    called = {}

    def fake_native_generate(**kwargs):
        called.update(kwargs)
        return {
            "native_runtime": 1,
            "generated_token_ids": [41, 42],
            "elapsed_us": 1_000_000,
            "prefill_us": 10_000,
            "tokens_per_second": 999.0,
            "tree_budget": 5,
            "cycles_completed": 2,
            "ddtree_cycles_completed": 2,
            "dflash_cycles_completed": 0,
            "dflash_accepted_from_draft": 0,
            "avg_acceptance": 3.0,
            "fast_path_ratio": 1.0,
            "fast_path_count": 2,
            "slow_path_count": 0,
            "tree_aware_commit_count": 2,
            "tree_aware_linear": True,
            "exact_commit": False,
            "dflash_controller_enabled": False,
            "dflash_controller_probe_count": 0,
            "dflash_controller_switch_count": 0,
            "phase_timings_us": {
                "draft": 10_000,
                "tree_build": 20_000,
                "tree_verify": 30_000,
                "commit": 40_000,
                "dflash_draft": 0,
                "dflash_verify": 0,
                "dflash_replay": 0,
                "dflash_commit": 0,
            },
        }

    monkeypatch.setenv("DFLASH_DDTREE_NATIVE_RUNTIME", "1")
    monkeypatch.setattr(
        "omlx.dflash.ddtree_runtime._execute_ddtree_native_generate",
        fake_native_generate,
    )

    telemetry = {}
    output_ids = execute_ddtree_mlx_generate(
        target_model=SimpleNamespace(),
        tokenizer=_FakeTokenizer(),
        drafter_model=SimpleNamespace(),
        prompt="hello",
        max_new_tokens=8,
        stop_token_ids=[99],
        suppress_token_ids=[0],
        telemetry_out=telemetry,
    )

    assert output_ids == [41, 42]
    assert called["max_new_tokens"] == 8
    assert called["prompt"] == "hello"
    assert telemetry["ddtree_native_runtime"] == 1
    assert telemetry["ddtree_tree_verify_s"] == 0.03
