from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.config import DFlashConfig
import pytest

from omlx.dflash.runtime import DFlashRuntime


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, prompt):
        return [1, 2, 3]

    def decode(self, token_ids):
        return ",".join(str(t) for t in token_ids)


class _StreamingDetokenizer:
    def __init__(self, token_map):
        self._token_map = token_map
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token_id):
        self.last_segment = self._token_map[token_id]

    def finalize(self):
        self.last_segment = ""


class _PrefixUnstableTokenizer(_FakeTokenizer):
    think_start = "<think>"
    think_start_id = 7
    think_end_id = 8

    def __init__(self):
        self._streaming_token_map = {10: "B", 11: "C"}

    @property
    def detokenizer(self):
        return _StreamingDetokenizer(self._streaming_token_map)

    def encode(self, prompt):
        if prompt == "thinking":
            return [1, self.think_start_id]
        return [1, 2, 3]

    def decode(self, token_ids):
        mapping = {
            (10,): "A",
            (10, 11): "BC",
            (11,): "Y",
        }
        return mapping.get(tuple(token_ids), super().decode(token_ids))


def test_runtime_generate_dispatches_to_bstnxbt_backend(monkeypatch):
    called = {}

    def fake_execute(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, telemetry_out=None):
        called["target_model"] = target_model
        called["tokenizer"] = tokenizer
        called["drafter_model"] = drafter_model
        called["prompt"] = prompt
        called["max_new_tokens"] = max_new_tokens
        called["stop_token_ids"] = stop_token_ids
        called["suppress_token_ids"] = suppress_token_ids
        return [10, 11, 12]

    monkeypatch.setattr("omlx.dflash.runtime.execute_bstnxbt_mlx_generate", fake_execute)

    config = DFlashConfig(draft_backend="bstnxbt_mlx")
    target_model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16))
    runtime = DFlashRuntime(config=config, target_model=target_model, drafter_model=drafter_model, verify_kernel=None)

    output = runtime.generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=3, stop_token_ids=[77], temperature=0.0, ignore_eos=False),
    )

    assert called["target_model"] is target_model
    assert called["drafter_model"] is drafter_model
    assert called["prompt"] == "hello"
    assert called["max_new_tokens"] == 3
    assert called["stop_token_ids"] == [77, 99]
    assert called["suppress_token_ids"] is None
    assert output.output_token_ids == [10, 11, 12]
    assert output.output_text == "10,11,12"


def test_runtime_generate_dispatches_to_mirror_sd_backend(monkeypatch):
    called = {}

    def fake_execute(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, telemetry_out=None):
        called["target_model"] = target_model
        called["tokenizer"] = tokenizer
        called["drafter_model"] = drafter_model
        called["prompt"] = prompt
        called["max_new_tokens"] = max_new_tokens
        called["stop_token_ids"] = stop_token_ids
        called["suppress_token_ids"] = suppress_token_ids
        return [20, 21]

    monkeypatch.setattr("omlx.dflash.runtime.execute_mirror_sd_mlx_generate", fake_execute)

    config = DFlashConfig(draft_backend="mirror_sd_mlx")
    target_model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16))
    runtime = DFlashRuntime(config=config, target_model=target_model, drafter_model=drafter_model, verify_kernel=None)

    output = runtime.generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=2, stop_token_ids=[77], temperature=0.0, ignore_eos=False),
    )

    assert called["target_model"] is target_model
    assert called["drafter_model"] is drafter_model
    assert called["prompt"] == "hello"
    assert called["max_new_tokens"] == 2
    assert called["stop_token_ids"] == [77, 99]
    assert called["suppress_token_ids"] is None
    assert output.output_token_ids == [20, 21]
    assert output.output_text == "20,21"


def test_runtime_generate_dispatches_to_ddtree_backend(monkeypatch):
    called = {}

    def fake_execute(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, telemetry_out=None):
        called["target_model"] = target_model
        called["tokenizer"] = tokenizer
        called["drafter_model"] = drafter_model
        called["prompt"] = prompt
        called["max_new_tokens"] = max_new_tokens
        called["stop_token_ids"] = stop_token_ids
        called["suppress_token_ids"] = suppress_token_ids
        telemetry_out["ddtree_enabled"] = 1
        return [30, 31]

    monkeypatch.setattr("omlx.dflash.runtime.execute_ddtree_mlx_generate", fake_execute)
    monkeypatch.setattr("omlx.dflash.runtime.ddtree_runtime_availability_reason", lambda: None)
    monkeypatch.setenv("DFLASH_BSTNXBT_EMIT_TELEMETRY", "1")

    config = DFlashConfig(draft_backend="ddtree_mlx")
    target_model = SimpleNamespace(name="target")
    drafter_model = SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16))
    runtime = DFlashRuntime(config=config, target_model=target_model, drafter_model=drafter_model, verify_kernel=None)

    output = runtime.generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=2, stop_token_ids=[77], temperature=0.0, ignore_eos=False),
    )

    assert called["target_model"] is target_model
    assert called["drafter_model"] is drafter_model
    assert called["prompt"] == "hello"
    assert called["max_new_tokens"] == 2
    assert called["stop_token_ids"] == [77, 99]
    assert called["suppress_token_ids"] is None
    assert output.output_token_ids == [30, 31]
    assert output.output_text == "30,31"
    telemetry = runtime.consume_last_telemetry()
    assert telemetry is not None
    assert telemetry["ddtree_enabled"] == 1


def test_runtime_generate_passes_thinking_budget_to_bstnxbt_backend(monkeypatch):
    called = {}

    def fake_execute(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, telemetry_out=None):
        called["thinking_budget"] = thinking_budget
        return [10]

    monkeypatch.setattr("omlx.dflash.runtime.execute_bstnxbt_mlx_generate", fake_execute)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    runtime.generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=1, stop_token_ids=[77], temperature=0.0, ignore_eos=False, thinking_budget=128),
    )

    assert called["thinking_budget"] == 128


def test_runtime_generate_can_ignore_eos_for_bstnxbt_backend(monkeypatch):
    called = {}

    def fake_execute(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, telemetry_out=None):
        called["stop_token_ids"] = stop_token_ids
        called["suppress_token_ids"] = suppress_token_ids
        return [10]

    monkeypatch.setattr("omlx.dflash.runtime.execute_bstnxbt_mlx_generate", fake_execute)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    runtime.generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=1, stop_token_ids=[77], temperature=0.0, ignore_eos=True),
    )

    assert called["stop_token_ids"] == [77]
    assert called["suppress_token_ids"] == [99]


@pytest.mark.asyncio
async def test_runtime_stream_generate_dispatches_thinking_budget_to_bstnxbt_backend(monkeypatch):
    called = {}

    def fake_iterate(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, should_abort=None, telemetry_out=None):
        called["thinking_budget"] = thinking_budget
        yield [10], [10], True, "stop"

    monkeypatch.setattr("omlx.dflash.runtime.iterate_bstnxbt_mlx_generate_commits", fake_iterate)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    outputs = []
    async for output in runtime.stream_generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=3, stop_token_ids=[77], temperature=0.0, ignore_eos=False, thinking_budget=128),
    ):
        outputs.append(output)

    assert called["thinking_budget"] == 128
    assert outputs[-1].finished is True


@pytest.mark.asyncio
async def test_runtime_stream_generate_dispatches_to_mirror_sd_backend(monkeypatch):
    def fake_iterate(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, should_abort=None, telemetry_out=None):
        yield [20], [20], False, None
        yield [21], [20, 21], True, "stop"

    monkeypatch.setattr("omlx.dflash.runtime.iterate_mirror_sd_mlx_generate_commits", fake_iterate)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="mirror_sd_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    outputs = []
    async for output in runtime.stream_generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=2, stop_token_ids=[77], temperature=0.0, ignore_eos=False),
    ):
        outputs.append(output)

    assert [o.new_token_ids for o in outputs] == [[20], [21]]
    assert outputs[-1].output_text == "20,21"
    assert outputs[-1].finished is True


@pytest.mark.asyncio
async def test_runtime_stream_generate_dispatches_to_bstnxbt_backend(monkeypatch):
    def fake_iterate(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, should_abort=None, telemetry_out=None):
        yield [10, 11], [10, 11], False, None
        yield [12], [10, 11, 12], True, "stop"

    monkeypatch.setattr("omlx.dflash.runtime.iterate_bstnxbt_mlx_generate_commits", fake_iterate)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    outputs = []
    async for output in runtime.stream_generate(
        prompt="hello",
        tokenizer=_FakeTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=3, stop_token_ids=[77], temperature=0.0, ignore_eos=False),
    ):
        outputs.append(output)

    assert [o.new_token_ids for o in outputs] == [[10, 11], [12]]
    assert outputs[0].output_token_ids == [10, 11]
    assert outputs[0].finished is False
    assert outputs[1].output_token_ids == [10, 11, 12]
    assert outputs[1].output_text == "10,11,12"
    assert outputs[1].finished is True
    assert outputs[1].finish_reason == "stop"


def test_runtime_request_supports_interleaved_concurrent_bstnxbt_requests():
    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    supported, reason = runtime.request_is_supported(
        stream=False,
        concurrent_requests=2,
        sampling_params=SimpleNamespace(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            stop=[],
        ),
    )

    assert supported is True
    assert reason is None


def test_runtime_request_supports_greedy_equivalent_top_k_one():
    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    supported, reason = runtime.request_is_supported(
        stream=False,
        concurrent_requests=1,
        sampling_params=SimpleNamespace(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            stop=[],
        ),
    )

    assert supported is True
    assert reason is None


def test_runtime_request_rejects_streaming_for_ddtree_backend(monkeypatch):
    monkeypatch.setattr("omlx.dflash.runtime.ddtree_runtime_availability_reason", lambda: None)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="ddtree_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    supported, reason = runtime.request_is_supported(
        stream=True,
        concurrent_requests=1,
        sampling_params=SimpleNamespace(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            stop=[],
        ),
    )

    assert supported is False
    assert reason is not None
    assert "streaming" in reason.lower()


@pytest.mark.asyncio
async def test_runtime_stream_generate_uses_streaming_detokenizer_for_prefix_unstable_decodes(monkeypatch):
    def fake_iterate(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, should_abort=None, telemetry_out=None):
        yield [10], [10], False, None
        yield [11], [10, 11], True, "stop"

    monkeypatch.setattr("omlx.dflash.runtime.iterate_bstnxbt_mlx_generate_commits", fake_iterate)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    outputs = []
    async for output in runtime.stream_generate(
        prompt="hello",
        tokenizer=_PrefixUnstableTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=2, stop_token_ids=[], temperature=0.0, ignore_eos=False),
    ):
        outputs.append(output)

    assert [o.new_text for o in outputs] == ["B", "C"]
    assert outputs[-1].output_text == "BC"


@pytest.mark.asyncio
async def test_runtime_stream_generate_injects_think_prefix_for_thinking_prompts(monkeypatch):
    def fake_iterate(*, target_model, tokenizer, drafter_model, prompt, max_new_tokens, stop_token_ids, suppress_token_ids=None, enable_thinking=False, thinking_budget=None, should_abort=None, telemetry_out=None):
        yield [10], [10], True, "stop"

    monkeypatch.setattr("omlx.dflash.runtime.iterate_bstnxbt_mlx_generate_commits", fake_iterate)

    runtime = DFlashRuntime(
        config=DFlashConfig(draft_backend="bstnxbt_mlx"),
        target_model=SimpleNamespace(name="target"),
        drafter_model=SimpleNamespace(model=SimpleNamespace(target_layer_ids=[1], block_size=16)),
        verify_kernel=None,
    )

    outputs = []
    async for output in runtime.stream_generate(
        prompt="thinking",
        tokenizer=_PrefixUnstableTokenizer(),
        sampling_params=SimpleNamespace(max_tokens=1, stop_token_ids=[], temperature=0.0, ignore_eos=False),
    ):
        outputs.append(output)

    assert outputs[0].new_text == "<think>\nB"
    assert outputs[0].output_text == "<think>\nB"
