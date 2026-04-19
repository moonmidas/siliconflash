import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "siliconflash_benchmark",
    Path(__file__).resolve().parents[1] / "scripts" / "siliconflash_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "usage": {"completion_tokens": 8, "prompt_tokens": 16},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }


class _FakeReasoningOnlyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "usage": {"completion_tokens": 8, "prompt_tokens": 16},
            "choices": [{
                "message": {"content": None, "reasoning_content": "thinking..."},
                "finish_reason": "length",
            }],
        }


class _FakeTelemetryResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "usage": {
                "completion_tokens": 8,
                "prompt_tokens": 16,
                "dflash_requested": True,
                "dflash_used": True,
                "dflash_backend": "mirror_sd_mlx",
                "dflash_acceptance_rate": 0.75,
                "dflash_draft_steps": 12,
                "dflash_total_s": 1.25,
            },
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }


def test_run_once_includes_dflash_in_request_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    perf_values = iter([100.0, 101.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    result = benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="omni9b-phase2prime",
        prompt="hello",
        max_tokens=8,
        dflash=True,
    )

    assert captured["url"] == "http://127.0.0.1:8011/v1/chat/completions"
    assert captured["json"]["dflash"] is True
    assert captured["json"]["temperature"] == 0
    assert captured["json"]["top_p"] == 1
    assert captured["json"]["top_k"] == 0
    assert captured["json"]["min_p"] == 0.0
    assert captured["json"]["stream"] is False
    assert result["completion_tokens"] == 8
    assert result["prompt_tokens"] == 16
    assert result["tok_s"] == pytest.approx(8.0)


def test_run_once_sets_dflash_false_by_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse()

    perf_values = iter([10.0, 12.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="omni9b-phase2prime",
        prompt="hello",
        max_tokens=8,
    )

    assert captured["json"]["dflash"] is False
    assert captured["json"]["top_p"] == 1
    assert captured["json"]["top_k"] == 0
    assert captured["json"]["min_p"] == 0.0


def test_run_once_can_enable_thinking(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse()

    perf_values = iter([30.0, 31.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="Qwen3.5-9B-bf16",
        prompt="reason about this",
        max_tokens=64,
        enable_thinking=True,
    )

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_run_once_can_set_thinking_budget(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse()

    perf_values = iter([40.0, 41.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="Qwen3.5-9B-bf16",
        prompt="reason about this",
        max_tokens=64,
        enable_thinking=True,
        thinking_budget=128,
    )

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["json"]["thinking_budget"] == 128


def test_run_once_can_disable_eos_stop(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse()

    perf_values = iter([50.0, 51.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="Qwen3.5-9B-bf16",
        prompt="math prompt",
        max_tokens=1024,
        ignore_eos=True,
    )

    assert captured["json"]["ignore_eos"] is True


def test_run_once_copies_dflash_usage_fields(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeTelemetryResponse()

    perf_values = iter([70.0, 71.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    result = benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="Qwen3.5-9B-bf16",
        prompt="math prompt",
        max_tokens=1024,
        enable_thinking=True,
    )

    assert result["dflash_requested"] is True
    assert result["dflash_used"] is True
    assert result["dflash_backend"] == "mirror_sd_mlx"
    assert result["dflash_acceptance_rate"] == pytest.approx(0.75)
    assert result["dflash_draft_steps"] == 12
    assert result["dflash_total_s"] == pytest.approx(1.25)


def test_run_once_handles_reasoning_only_response(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeReasoningOnlyResponse()

    perf_values = iter([60.0, 61.0])

    monkeypatch.setattr(benchmark.requests, "post", fake_post)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    result = benchmark.run_once(
        base_url="http://127.0.0.1:8011",
        model="Qwen3.5-9B-bf16",
        prompt="math prompt",
        max_tokens=1024,
        enable_thinking=True,
    )

    assert result["finish_reason"] == "length"
    assert result["text_preview"] == "thinking..."
