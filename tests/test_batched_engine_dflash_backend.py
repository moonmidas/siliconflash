from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.engine.base import GenerationOutput


@pytest.mark.asyncio
async def test_maybe_build_dflash_runtime_selects_bstnxbt_backend_for_native_drafter(monkeypatch):
    settings = SimpleNamespace(
        dflash_enabled=True,
        dflash_draft_model=None,
        dflash_use_mlx_native_drafter=True,
        dflash_block_size=16,
        dflash_verify_kernel="metal_batched_gemv",
        dflash_single_batch_only=True,
        dflash_report_acceptance_rate=True,
    )
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._model = SimpleNamespace()

    monkeypatch.setenv("DFLASH_BSTNXBT_RUNTIME", "1")

    runtime = await engine._maybe_build_dflash_runtime(loop=None)

    assert runtime is not None
    assert runtime.config.draft_backend == "bstnxbt_mlx"


@pytest.mark.asyncio
async def test_maybe_build_dflash_runtime_selects_mirror_backend_when_env_enabled(monkeypatch):
    settings = SimpleNamespace(
        dflash_enabled=True,
        dflash_draft_model=None,
        dflash_use_mlx_native_drafter=True,
        dflash_block_size=16,
        dflash_verify_kernel="metal_batched_gemv",
        dflash_single_batch_only=True,
        dflash_report_acceptance_rate=True,
    )
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._model = SimpleNamespace()

    monkeypatch.setenv("DFLASH_MIRROR_SD_RUNTIME", "1")
    monkeypatch.setenv("DFLASH_BSTNXBT_RUNTIME", "1")

    runtime = await engine._maybe_build_dflash_runtime(loop=None)

    assert runtime is not None
    assert runtime.config.draft_backend == "mirror_sd_mlx"


@pytest.mark.asyncio
async def test_maybe_build_dflash_runtime_selects_ddtree_backend_when_env_enabled(monkeypatch):
    settings = SimpleNamespace(
        dflash_enabled=True,
        dflash_draft_model=None,
        dflash_use_mlx_native_drafter=True,
        dflash_block_size=16,
        dflash_verify_kernel="metal_batched_gemv",
        dflash_single_batch_only=True,
        dflash_report_acceptance_rate=True,
    )
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._model = SimpleNamespace()

    monkeypatch.setenv("DFLASH_DDTREE_RUNTIME", "1")
    monkeypatch.setenv("DFLASH_MIRROR_SD_RUNTIME", "1")
    monkeypatch.setenv("DFLASH_BSTNXBT_RUNTIME", "1")

    runtime = await engine._maybe_build_dflash_runtime(loop=None)

    assert runtime is not None
    assert runtime.config.draft_backend == "ddtree_mlx"


@pytest.mark.asyncio
async def test_stream_generate_cleans_special_tokens_from_dflash_new_text():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True
    engine._tokenizer = SimpleNamespace()

    class _UnusedEngine:
        async def add_request(self, *args, **kwargs):
            raise AssertionError("stock engine should not be used")

        async def stream_outputs(self, request_id):
            yield None

        async def abort_request(self, request_id):
            return None

    async def _stream_generate(*, prompt, tokenizer, sampling_params, executor=None):
        yield SimpleNamespace(
            output_text="hello<|im_end|>",
            new_text="hello<|im_end|>",
            prompt_tokens=3,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine = _UnusedEngine()
    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=lambda **kwargs: (True, None),
        stream_generate=_stream_generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx"),
    )

    outputs = []
    async for output in engine.stream_generate(
        prompt="hello",
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
    ):
        outputs.append(output)

    assert [o.new_text for o in outputs] == ["hello"]
    assert [o.text for o in outputs] == ["hello"]


@pytest.mark.asyncio
async def test_stream_generate_uses_dflash_runtime_when_supported():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True
    engine._tokenizer = SimpleNamespace()

    class _UnusedEngine:
        async def add_request(self, *args, **kwargs):
            raise AssertionError("stock engine should not be used")

        async def stream_outputs(self, request_id):
            yield None

        async def abort_request(self, request_id):
            return None

    async def _stream_generate(*, prompt, tokenizer, sampling_params, executor=None):
        yield SimpleNamespace(
            output_text="hello",
            new_text="hello",
            prompt_tokens=3,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            cached_tokens=0,
        )
        yield SimpleNamespace(
            output_text="hello world",
            new_text=" world",
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine = _UnusedEngine()
    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=lambda **kwargs: (True, None),
        stream_generate=_stream_generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx"),
    )

    outputs = []
    async for output in engine.stream_generate(
        prompt="hello",
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
    ):
        outputs.append(output)

    assert [o.new_text for o in outputs] == ["hello", " world"]
    assert outputs[-1].finished is True
    assert outputs[-1].backend_metadata == {
        "dflash": {
            "requested": True,
            "used": True,
            "reason": None,
            "backend": "bstnxbt_mlx",
        }
    }


@pytest.mark.asyncio
async def test_chat_uses_dflash_runtime_with_thinking_and_top_k_one(monkeypatch):
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.encode.return_value = [1, 2, 3]
    engine._tokenizer = tokenizer

    executor = object()
    engine._engine = SimpleNamespace(_mlx_executor=executor)
    runtime_calls = {}

    class _FakeLoop:
        async def run_in_executor(self, executor_arg, fn, *args):
            runtime_calls["executor"] = executor_arg
            return fn(*args)

    monkeypatch.setattr("omlx.engine.batched.asyncio.get_running_loop", lambda: _FakeLoop())

    def request_is_supported(**kwargs):
        runtime_calls["sampling_params"] = kwargs["sampling_params"]
        return True, None

    def generate(**kwargs):
        runtime_calls["prompt"] = kwargs["prompt"]
        return SimpleNamespace(
            output_text="answer",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=request_is_supported,
        generate=generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx", fallback_to_target_only=True),
    )

    output = await engine.chat(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        min_p=0.0,
        thinking_budget=128,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert output.text == "answer"
    assert runtime_calls["prompt"] == "formatted prompt"
    assert runtime_calls["sampling_params"].top_k == 1
    assert runtime_calls["sampling_params"].thinking_budget == 128
    assert runtime_calls["sampling_params"].enable_thinking is True
    assert runtime_calls["executor"] is executor
    assert tokenizer.apply_chat_template.call_args[1]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_stream_chat_uses_dflash_runtime_with_thinking_and_top_k_one():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.encode.return_value = [1, 2, 3]
    engine._tokenizer = tokenizer
    executor = object()
    engine._engine = SimpleNamespace(_mlx_executor=executor)

    runtime_calls = {}

    def request_is_supported(**kwargs):
        runtime_calls["sampling_params"] = kwargs["sampling_params"]
        return True, None

    async def stream_generate(**kwargs):
        runtime_calls["prompt"] = kwargs["prompt"]
        runtime_calls["executor"] = kwargs.get("executor")
        yield SimpleNamespace(
            output_text="think",
            new_text="think",
            prompt_tokens=3,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            cached_tokens=0,
        )
        yield SimpleNamespace(
            output_text="think answer",
            new_text=" answer",
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=request_is_supported,
        stream_generate=stream_generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx"),
    )

    outputs = []
    async for output in engine.stream_chat(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        min_p=0.0,
        thinking_budget=128,
        chat_template_kwargs={"enable_thinking": True},
    ):
        outputs.append(output)

    assert [o.new_text for o in outputs] == ["think", " answer"]
    assert runtime_calls["prompt"] == "formatted prompt"
    assert runtime_calls["sampling_params"].top_k == 1
    assert runtime_calls["sampling_params"].thinking_budget == 128
    assert runtime_calls["sampling_params"].enable_thinking is True
    assert runtime_calls["executor"] is executor
    assert tokenizer.apply_chat_template.call_args[1]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_generate_allows_interleaved_concurrent_dflash_requests():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True
    engine._tokenizer = SimpleNamespace()

    class _UnusedEngine:
        async def generate(self, *args, **kwargs):
            raise AssertionError("stock engine should not be used")

    active = 0
    max_active = 0
    support_counts = []

    def request_is_supported(**kwargs):
        support_counts.append(kwargs["concurrent_requests"])
        return True, None

    async def async_generate(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return SimpleNamespace(
            output_text="answer",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine = _UnusedEngine()
    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=request_is_supported,
        async_generate=async_generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx", fallback_to_target_only=True),
    )

    out1, out2 = await asyncio.gather(
        engine.generate(prompt="a", max_tokens=2, temperature=0.0, top_p=1.0, top_k=1, min_p=0.0),
        engine.generate(prompt="b", max_tokens=2, temperature=0.0, top_p=1.0, top_k=1, min_p=0.0),
    )

    assert out1.text == "answer"
    assert out2.text == "answer"
    assert max_active == 2
    assert sorted(set(support_counts)) == [1, 2]


@pytest.mark.asyncio
async def test_generate_serializes_concurrent_non_bstnxbt_dflash_requests():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True
    engine._tokenizer = SimpleNamespace()

    class _UnusedEngine:
        async def generate(self, *args, **kwargs):
            raise AssertionError("stock engine should not be used")

    active = 0
    max_active = 0
    async_generate_calls = 0

    def request_is_supported(**kwargs):
        return True, None

    async def async_generate(**kwargs):
        nonlocal async_generate_calls
        async_generate_calls += 1
        raise AssertionError("async_generate should not be used for non-bstnxbt backends")

    def generate(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.03)
        active -= 1
        return SimpleNamespace(
            output_text="answer",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine = _UnusedEngine()
    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=request_is_supported,
        async_generate=async_generate,
        generate=generate,
        config=SimpleNamespace(draft_backend="zlab_spec_generate", fallback_to_target_only=True),
    )

    out1, out2 = await asyncio.gather(
        engine.generate(prompt="a", max_tokens=2, temperature=0.0, top_p=1.0, top_k=1, min_p=0.0),
        engine.generate(prompt="b", max_tokens=2, temperature=0.0, top_p=1.0, top_k=1, min_p=0.0),
    )

    assert out1.text == "answer"
    assert out2.text == "answer"
    assert max_active == 1
    assert async_generate_calls == 0


@pytest.mark.asyncio
async def test_stream_generate_allows_interleaved_concurrent_dflash_requests():
    settings = SimpleNamespace(dflash_enabled=True)
    engine = BatchedEngine(model_name="Qwen/Qwen3.5-9B", model_settings=settings)
    engine._loaded = True
    engine._tokenizer = SimpleNamespace()

    class _UnusedEngine:
        async def add_request(self, *args, **kwargs):
            raise AssertionError("stock engine should not be used")

        async def stream_outputs(self, request_id):
            yield None

        async def abort_request(self, request_id):
            return None

    active = 0
    max_active = 0

    def request_is_supported(**kwargs):
        return True, None

    async def stream_generate(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            yield SimpleNamespace(
                output_text="hello",
                new_text="hello",
                prompt_tokens=3,
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
                cached_tokens=0,
            )
            await asyncio.sleep(0.03)
            yield SimpleNamespace(
                output_text="hello world",
                new_text=" world",
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
                cached_tokens=0,
            )
        finally:
            active -= 1

    engine._engine = _UnusedEngine()
    engine._dflash_runtime = SimpleNamespace(
        request_is_supported=request_is_supported,
        stream_generate=stream_generate,
        config=SimpleNamespace(draft_backend="bstnxbt_mlx"),
    )

    async def collect(prompt):
        vals = []
        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
        ):
            vals.append(output.new_text)
        return vals

    out1, out2 = await asyncio.gather(collect("a"), collect("b"))

    assert out1 == ["hello", " world"]
    assert out2 == ["hello", " world"]
    assert max_active == 2
