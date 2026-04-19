# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.
"""

import asyncio
import copy
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import SPECIAL_TOKENS_PATTERN, clean_special_tokens, detect_and_strip_partial
from ..dflash import DFlashConfig, DFlashRuntime
from ..utils.tokenizer import get_tokenizer_config
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


# Optional Harmony adapter import
try:
    from ..adapter.harmony import preprocess_harmony_messages

    HAS_HARMONY_ADAPTER = True
except ImportError:
    HAS_HARMONY_ADAPTER = False
    preprocess_harmony_messages = None  # type: ignore


class BatchedEngine(BaseEngine):
    """
    Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 8,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (8=lower collector overhead,
                1=every token)
            enable_thinking: Enable thinking mode for reasoning models (passed to chat_template_kwargs)
            model_settings: Optional per-model settings for post-load transforms
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False
        self._grammar_compiler = None
        self._grammar_compiler_init_attempted = False
        self._dflash_runtime: DFlashRuntime | None = None
        self._dflash_request_lock: asyncio.Lock | None = None
        self._dflash_state_lock: asyncio.Lock | None = None
        self._active_dflash_requests = 0

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        """Get the model type from config (e.g., 'gpt_oss', 'llama', 'qwen2')."""
        if self._model is None:
            return None
        # Try different ways to access model_type
        try:
            if hasattr(self._model, 'config'):
                config = self._model.config
                if hasattr(config, 'model_type'):
                    model_type = config.model_type
                    return model_type if isinstance(model_type, str) else None
                elif isinstance(config, dict):
                    model_type = config.get('model_type')
                    return model_type if isinstance(model_type, str) else None
            if hasattr(self._model, 'args'):
                args = self._model.args
                if hasattr(args, 'model_type'):
                    model_type = args.model_type
                    return model_type if isinstance(model_type, str) else None
        except Exception as e:
            logger.debug(f"Error getting model_type: {e}")
        return None

    @property
    def message_extractor(self):
        """Return the model-specific message extractor function, or ``None``.

        ``None`` means the server should use its default extractor
        (``extract_text_content`` or ``extract_multimodal_content``).
        """
        try:
            from ..adapter.output_parser import detect_message_extractor
            model_config = None
            if self._model is not None and hasattr(self._model, "config"):
                cfg = self._model.config
                if hasattr(cfg, "model_type"):
                    model_config = {"model_type": cfg.model_type}
                elif isinstance(cfg, dict):
                    model_config = cfg
            return detect_message_extractor(self._model_name, model_config)
        except Exception:
            return None

    @property
    def grammar_compiler(self):
        """Lazily create and return a GrammarCompiler for this model.

        Returns ``None`` when xgrammar is not installed or initialization fails.
        """
        if self._grammar_compiler is not None:
            return self._grammar_compiler
        if self._grammar_compiler_init_attempted:
            return None
        self._grammar_compiler_init_attempted = True
        try:
            from ..api.grammar import create_grammar_compiler

            self._grammar_compiler = create_grammar_compiler(self._tokenizer, self._model)
            logger.info("GrammarCompiler initialized for %s", self._model_name)
        except Exception:
            from ..utils.install import get_install_method

            method = get_install_method()
            if method == "dmg":
                logger.info(
                    "Structured output is not available in the DMG version "
                    "(xgrammar requires torch which significantly increases app size). "
                    "Use the pip or Homebrew version for structured output support."
                )
            elif method == "homebrew":
                logger.info(
                    "Structured output requires xgrammar. "
                    "Reinstall with: brew reinstall omlx --with-grammar"
                )
            else:
                logger.info(
                    "Structured output requires xgrammar. "
                    "Install with: pip install 'omlx[grammar]'"
                )
        return self._grammar_compiler

    def _preprocess_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Preprocess messages for model-specific formats.

        Currently handles Harmony (gpt-oss) models.

        Args:
            messages: List of chat messages

        Returns:
            Preprocessed messages
        """
        if self.model_type == "gpt_oss" and HAS_HARMONY_ADAPTER:
            return preprocess_harmony_messages(messages)
        return messages

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        import asyncio

        from mlx_lm import load

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Build tokenizer config with model-specific fixes
        tokenizer_config = get_tokenizer_config(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model on the global MLX executor to avoid blocking the event loop
        # while ensuring no concurrent Metal operations. See issue #85.
        from ..engine_core import get_mlx_executor

        def _load_model_sync():
            return load(
                self._model_name,
                tokenizer_config=tokenizer_config,
            )

        loop = asyncio.get_running_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            get_mlx_executor(), _load_model_sync
        )

        # Apply post-load transforms (e.g., IndexCache for DSA models)
        from ..utils.model_loading import apply_post_load_transforms

        self._model = apply_post_load_transforms(
            self._model, self._model_settings
        )

        # TurboQuant KV cache: patch attention and set kv_bits on scheduler
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_kv_enabled", False)
            if tq_enabled:
                from ..patches.turboquant_attention import apply_turboquant_attention_patch
                apply_turboquant_attention_patch()
                tq_bits = float(getattr(self._model_settings, "turboquant_kv_bits", 4))
                logger.info(f"TurboQuant KV cache enabled: {tq_bits} bits")

        # Create engine config (copy to avoid mutating the shared instance)
        scheduler_config = copy.copy(self._scheduler_config) if self._scheduler_config else SchedulerConfig()
        scheduler_config.model_name = self._model_name  # Ensure cache isolation per model
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()

        # TurboQuant KV cache: propagate bits to scheduler
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_kv_enabled", False)
            if tq_enabled:
                tq_bits = float(getattr(self._model_settings, "turboquant_kv_bits", 4))
                self._engine.engine.scheduler._turboquant_kv_bits = tq_bits

        # SpecPrefill: load draft model and pass to scheduler
        if self._model_settings is not None:
            specprefill_draft = getattr(self._model_settings, "specprefill_draft_model", None)
            specprefill_enabled = getattr(self._model_settings, "specprefill_enabled", False)
            if specprefill_enabled and specprefill_draft:
                try:
                    def _load_draft():
                        draft_model, _ = load(specprefill_draft)
                        return draft_model
                    draft_model = await loop.run_in_executor(get_mlx_executor(), _load_draft)
                    self._engine.engine.scheduler.set_specprefill_draft_model(
                        draft_model, draft_model_name=specprefill_draft
                    )
                    logger.info(f"SpecPrefill: draft model loaded ({specprefill_draft})")
                except Exception as e:
                    logger.error(f"SpecPrefill: draft model load failed: {e}")

        self._dflash_runtime = await self._maybe_build_dflash_runtime(loop)

        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._dflash_runtime = None
        self._loaded = False
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template to messages.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions
            chat_template_kwargs: Optional kwargs passed to tokenizer.apply_chat_template
                (e.g. enable_thinking, reasoning_effort). Overrides global _enable_thinking.
        """
        if hasattr(self._tokenizer, 'apply_chat_template'):
            is_partial = detect_and_strip_partial(messages)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            # Global fallback
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            # Per-model/request kwargs override global
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Tokenizer doesn't support some kwargs, remove them and retry
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                # Template rendering failed (e.g. Jinja2 TemplateError from
                # unsupported roles, invalid message format, etc.)
                logger.error(f"Chat template rendering failed: {e}")
                raise
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """
        Count prompt tokens for chat messages after applying chat template.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions
            chat_template_kwargs: Optional kwargs for chat template

        Returns:
            Number of prompt tokens
        """
        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        dflash_requested = kwargs.pop("dflash", None)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            stop_token_ids=kwargs.get("stop_token_ids", None),
            ignore_eos=bool(kwargs.get("ignore_eos", False)),
            enable_thinking=bool(kwargs.get("enable_thinking", False)),
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
        )

        dflash_runtime = self._dflash_runtime
        self._consume_dflash_telemetry(dflash_runtime)
        dflash_backend = getattr(getattr(dflash_runtime, "config", None), "draft_backend", None)
        use_interleaved_dflash = dflash_backend in ("bstnxbt_mlx", "mirror_sd_mlx")
        dflash_metadata = self._build_dflash_metadata(
            request_override=dflash_requested,
            stream=False,
            sampling_params=sampling_params,
            used=False,
            concurrent_requests=1,
        )
        if dflash_runtime is not None and self._should_use_dflash(dflash_requested):
            if use_interleaved_dflash:
                async with self._track_active_dflash_request():
                    dflash_concurrent_requests = self._active_dflash_requests
                    dflash_metadata = self._build_dflash_metadata(
                        request_override=dflash_requested,
                        stream=False,
                        sampling_params=sampling_params,
                        used=False,
                        concurrent_requests=dflash_concurrent_requests,
                    )
                    supported, reason = dflash_runtime.request_is_supported(
                        stream=False,
                        concurrent_requests=dflash_concurrent_requests,
                        sampling_params=sampling_params,
                    )
                    if supported:
                        logger.info("DFlash requested for %s; runtime path selected", self._model_name)
                        try:
                            dflash_async_generate = getattr(dflash_runtime, "async_generate", None)
                            if callable(dflash_async_generate):
                                output = await dflash_async_generate(
                                    prompt=prompt,
                                    tokenizer=self._tokenizer,
                                    sampling_params=sampling_params,
                                    executor=self._get_dflash_executor(),
                                )
                            else:
                                loop = asyncio.get_running_loop()
                                async with self._get_dflash_request_lock():
                                    output = await loop.run_in_executor(
                                        self._get_dflash_executor(),
                                        lambda: dflash_runtime.generate(
                                            prompt=prompt,
                                            tokenizer=self._tokenizer,
                                            sampling_params=sampling_params,
                                        ),
                                    )
                            dflash_metadata = self._build_dflash_metadata(
                                request_override=dflash_requested,
                                stream=False,
                                sampling_params=sampling_params,
                                used=True,
                                concurrent_requests=dflash_concurrent_requests,
                            )
                            dflash_metadata = self._with_dflash_telemetry(
                                dflash_metadata,
                                self._consume_dflash_telemetry(dflash_runtime),
                            )
                        except Exception as e:
                            if dflash_runtime.config.fallback_to_target_only:
                                logger.warning(
                                    "DFlash execution failed for %s: %s; falling back to stock generation",
                                    self._model_name,
                                    e,
                                )
                                dflash_metadata = self._build_dflash_metadata(
                                    request_override=dflash_requested,
                                    stream=False,
                                    sampling_params=sampling_params,
                                    used=False,
                                    reason_override=f"DFlash execution failed: {e}",
                                    concurrent_requests=dflash_concurrent_requests,
                                )
                                output = None
                            else:
                                raise
                    else:
                        logger.info(
                            "DFlash requested for %s but unavailable: %s; falling back",
                            self._model_name,
                            reason,
                        )
                        dflash_metadata = self._build_dflash_metadata(
                            request_override=dflash_requested,
                            stream=False,
                            sampling_params=sampling_params,
                            used=False,
                            reason_override=reason,
                            concurrent_requests=dflash_concurrent_requests,
                        )
                        output = None
                if output is None:
                    output = await self._engine.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                    )
            else:
                supported, reason = dflash_runtime.request_is_supported(
                    stream=False,
                    concurrent_requests=1,
                    sampling_params=sampling_params,
                )
                if supported:
                    logger.info("DFlash requested for %s; runtime path selected", self._model_name)
                    try:
                        loop = asyncio.get_running_loop()
                        async with self._get_dflash_request_lock():
                            output = await loop.run_in_executor(
                                self._get_dflash_executor(),
                                lambda: dflash_runtime.generate(
                                    prompt=prompt,
                                    tokenizer=self._tokenizer,
                                    sampling_params=sampling_params,
                                ),
                            )
                        dflash_metadata = self._build_dflash_metadata(
                            request_override=dflash_requested,
                            stream=False,
                            sampling_params=sampling_params,
                            used=True,
                            concurrent_requests=1,
                        )
                        dflash_metadata = self._with_dflash_telemetry(
                            dflash_metadata,
                            self._consume_dflash_telemetry(dflash_runtime),
                        )
                    except Exception as e:
                        if dflash_runtime.config.fallback_to_target_only:
                            logger.warning(
                                "DFlash execution failed for %s: %s; falling back to stock generation",
                                self._model_name,
                                e,
                            )
                            dflash_metadata = self._build_dflash_metadata(
                                request_override=dflash_requested,
                                stream=False,
                                sampling_params=sampling_params,
                                used=False,
                                reason_override=f"DFlash execution failed: {e}",
                                concurrent_requests=1,
                            )
                            output = await self._engine.generate(
                                prompt=prompt,
                                sampling_params=sampling_params,
                            )
                        else:
                            raise
                else:
                    logger.info(
                        "DFlash requested for %s but unavailable: %s; falling back",
                        self._model_name,
                        reason,
                    )
                    dflash_metadata = self._build_dflash_metadata(
                        request_override=dflash_requested,
                        stream=False,
                        sampling_params=sampling_params,
                        used=False,
                        reason_override=reason,
                        concurrent_requests=1,
                    )
                    output = await self._engine.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                    )
        else:
            output = await self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
            )

        text = clean_special_tokens(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            tool_calls=output.tool_calls,
            cached_tokens=output.cached_tokens,
            backend_metadata=dflash_metadata,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            stop_token_ids=kwargs.get("stop_token_ids", None),
            ignore_eos=bool(kwargs.get("ignore_eos", False)),
            enable_thinking=bool(kwargs.get("enable_thinking", False)),
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
        )

        dflash_requested = kwargs.pop("dflash", None)

        dflash_runtime = self._dflash_runtime
        self._consume_dflash_telemetry(dflash_runtime)
        dflash_backend = getattr(getattr(dflash_runtime, "config", None), "draft_backend", None)
        use_interleaved_dflash = dflash_backend in ("bstnxbt_mlx", "mirror_sd_mlx")
        dflash_metadata = self._build_dflash_metadata(
            request_override=dflash_requested,
            stream=True,
            sampling_params=sampling_params,
            used=False,
            concurrent_requests=1,
        )
        if dflash_runtime is not None and self._should_use_dflash(dflash_requested):
            if use_interleaved_dflash:
                async with self._track_active_dflash_request():
                    dflash_concurrent_requests = self._active_dflash_requests
                    supported, reason = dflash_runtime.request_is_supported(
                        stream=True,
                        concurrent_requests=dflash_concurrent_requests,
                        sampling_params=sampling_params,
                    )
                    if supported:
                        logger.info("DFlash requested for %s; streaming runtime path selected", self._model_name)
                        dflash_metadata = self._build_dflash_metadata(
                            request_override=dflash_requested,
                            stream=True,
                            sampling_params=sampling_params,
                            used=True,
                            concurrent_requests=dflash_concurrent_requests,
                        )
                        async for output in dflash_runtime.stream_generate(
                            prompt=prompt,
                            tokenizer=self._tokenizer,
                            sampling_params=sampling_params,
                            executor=self._get_dflash_executor(),
                        ):
                            text = clean_special_tokens(output.output_text)
                            cleaned_new_text = SPECIAL_TOKENS_PATTERN.sub("", output.new_text)
                            yield GenerationOutput(
                                text=text,
                                new_text=cleaned_new_text,
                                prompt_tokens=output.prompt_tokens,
                                completion_tokens=output.completion_tokens,
                                finished=output.finished,
                                finish_reason=output.finish_reason,
                                tool_calls=output.tool_calls,
                                cached_tokens=output.cached_tokens,
                                backend_metadata=dflash_metadata,
                            )
                        return
                    logger.info(
                        "DFlash requested for %s but unavailable for streaming: %s; falling back",
                        self._model_name,
                        reason,
                    )
                    dflash_metadata = self._build_dflash_metadata(
                        request_override=dflash_requested,
                        stream=True,
                        sampling_params=sampling_params,
                        used=False,
                        reason_override=reason,
                        concurrent_requests=dflash_concurrent_requests,
                    )
            else:
                supported, reason = dflash_runtime.request_is_supported(
                    stream=True,
                    concurrent_requests=1,
                    sampling_params=sampling_params,
                )
                if supported:
                    logger.info("DFlash requested for %s; streaming runtime path selected", self._model_name)
                    dflash_metadata = self._build_dflash_metadata(
                        request_override=dflash_requested,
                        stream=True,
                        sampling_params=sampling_params,
                        used=True,
                        concurrent_requests=1,
                    )
                    async for output in dflash_runtime.stream_generate(
                        prompt=prompt,
                        tokenizer=self._tokenizer,
                        sampling_params=sampling_params,
                        executor=self._get_dflash_executor(),
                    ):
                        text = clean_special_tokens(output.output_text)
                        cleaned_new_text = SPECIAL_TOKENS_PATTERN.sub("", output.new_text)
                        yield GenerationOutput(
                            text=text,
                            new_text=cleaned_new_text,
                            prompt_tokens=output.prompt_tokens,
                            completion_tokens=output.completion_tokens,
                            finished=output.finished,
                            finish_reason=output.finish_reason,
                            tool_calls=output.tool_calls,
                            cached_tokens=output.cached_tokens,
                            backend_metadata=dflash_metadata,
                        )
                    return
                logger.info(
                    "DFlash requested for %s but unavailable for streaming: %s; falling back",
                    self._model_name,
                    reason,
                )
                dflash_metadata = self._build_dflash_metadata(
                    request_override=dflash_requested,
                    stream=True,
                    sampling_params=sampling_params,
                    used=False,
                    reason_override=reason,
                    concurrent_requests=1,
                )

        # SpecPrefill: pass per-request overrides to engine
        specprefill_kwargs = {}
        if kwargs.get("specprefill") is not None:
            specprefill_kwargs["specprefill"] = kwargs.pop("specprefill")
        if kwargs.get("specprefill_keep_pct") is not None:
            specprefill_kwargs["specprefill_keep_pct"] = kwargs.pop("specprefill_keep_pct")
        if kwargs.get("specprefill_threshold") is not None:
            specprefill_kwargs["specprefill_threshold"] = kwargs.pop("specprefill_threshold")
        if kwargs.get("specprefill_system_end") is not None:
            specprefill_kwargs["specprefill_system_end"] = kwargs.pop("specprefill_system_end")

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            **specprefill_kwargs,
        )

        finished_normally = False
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                # Set finished_normally BEFORE yield, because the consumer
                # may stop iterating after receiving the final output,
                # which triggers GeneratorExit at the yield point -
                # code after yield would never execute.
                if output.finished:
                    finished_normally = True

                cleaned_new_text = SPECIAL_TOKENS_PATTERN.sub("", output.new_text)
                yield GenerationOutput(
                    text=text,
                    new_text=cleaned_new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                    backend_metadata=dflash_metadata,
                )
        except GeneratorExit:
            # Client disconnected
            logger.info(f"[stream_generate] GeneratorExit caught for request {request_id}")
        finally:
            # Abort the request if client disconnected before completion
            if not finished_normally:
                logger.info(f"[stream_generate] Aborting request {request_id} (finished_normally={finished_normally})")
                await self._engine.abort_request(request_id)
            else:
                logger.debug(f"[stream_generate] Request {request_id} finished normally")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        # Preprocess messages for Harmony (gpt-oss) models
        messages = self._preprocess_messages(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )
        kwargs.setdefault("enable_thinking", bool(ct_kwargs and ct_kwargs.get("enable_thinking", False)))

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            presence_penalty: Presence penalty (0.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Preprocess messages for Harmony (gpt-oss) models
        messages = self._preprocess_messages(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )
        kwargs.setdefault("enable_thinking", bool(ct_kwargs and ct_kwargs.get("enable_thinking", False)))

        # SpecPrefill: compute system prompt token count for protection.
        # Can't template system-only messages (most templates require user),
        # so compute by subtracting non-system from full prompt tokens.
        if kwargs.get("specprefill") is not False:
            non_system = [m for m in messages if m.get("role") not in ("system", "developer")]
            if len(non_system) < len(messages) and non_system:
                try:
                    non_system_prompt = self._apply_chat_template(
                        non_system, template_tools, chat_template_kwargs=ct_kwargs
                    )
                    full_tokens = len(self._tokenizer.encode(prompt))
                    non_system_tokens = len(self._tokenizer.encode(non_system_prompt))
                    system_end = full_tokens - non_system_tokens
                    if system_end > 0:
                        kwargs["specprefill_system_end"] = system_end
                except Exception as e:
                    logger.debug(f"SpecPrefill: system_end calc failed: {e}")

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        ):
            yield output

    async def _maybe_build_dflash_runtime(self, loop) -> DFlashRuntime | None:
        if self._model_settings is None:
            return None

        enabled = bool(getattr(self._model_settings, "dflash_enabled", False))
        draft_model_path = getattr(self._model_settings, "dflash_draft_model", None)
        if not enabled and not draft_model_path:
            return None

        use_native = bool(getattr(self._model_settings, "dflash_use_mlx_native_drafter", False))
        use_ddtree_runtime = use_native and os.environ.get("DFLASH_DDTREE_RUNTIME") == "1"
        use_mirror_sd_runtime = (
            use_native
            and os.environ.get("DFLASH_MIRROR_SD_RUNTIME") == "1"
            and not use_ddtree_runtime
        )
        use_bstnxbt_runtime = (
            use_native
            and os.environ.get("DFLASH_BSTNXBT_RUNTIME") == "1"
            and not use_mirror_sd_runtime
            and not use_ddtree_runtime
        )
        draft_backend = (
            "ddtree_mlx"
            if use_ddtree_runtime
            else ("mirror_sd_mlx" if use_mirror_sd_runtime else ("bstnxbt_mlx" if use_bstnxbt_runtime else "zlab_spec_generate"))
        )

        config = DFlashConfig(
            enabled=enabled,
            draft_model_path=draft_model_path,
            target_model_name=self._model_name,
            draft_backend=draft_backend,
            block_size=int(getattr(self._model_settings, "dflash_block_size", 16)),
            verify_kernel=str(getattr(self._model_settings, "dflash_verify_kernel", "metal_batched_gemv")),
            single_batch_only=bool(getattr(self._model_settings, "dflash_single_batch_only", True)),
            report_acceptance_rate=bool(getattr(self._model_settings, "dflash_report_acceptance_rate", True)),
        )

        drafter_model = None
        draft_load_error = None
        if draft_model_path:
            try:
                from ..engine_core import get_mlx_executor
                if use_native:
                    from ..dflash.drafter import load_native_mlx_dflash_model
                    load_fn = load_native_mlx_dflash_model
                else:
                    from ..dflash.drafter import load_zlab_dflash_model
                    load_fn = load_zlab_dflash_model

                drafter_model = await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: load_fn(draft_model_path),
                )
                logger.info("DFlash: loaded drafter (%s, native=%s)", draft_model_path, use_native)
            except Exception as e:
                draft_load_error = str(e)
                logger.warning(
                    "DFlash: draft model load failed (%s): %s",
                    draft_model_path,
                    e,
                )

        runtime = DFlashRuntime(
            config=config,
            target_model=self._model,
            drafter_model=drafter_model,
            verify_kernel=None,
        )
        if drafter_model is None and draft_model_path:
            runtime._draft_load_error = draft_load_error or (
                "Failed to load z-lab DFlash drafter. Ensure torch is installed "
                "for the Transformers backend."
            )
            runtime._availability_reason = runtime._compute_availability_reason()
        logger.info("DFlash runtime status: %s", runtime.status())
        return runtime

    def _should_use_dflash(self, request_override: bool | None) -> bool:
        if request_override is not None:
            return request_override
        return bool(
            self._model_settings is not None
            and getattr(self._model_settings, "dflash_enabled", False)
        )

    def _get_dflash_request_lock(self) -> asyncio.Lock:
        if self._dflash_request_lock is None:
            self._dflash_request_lock = asyncio.Lock()
        return self._dflash_request_lock

    def _get_dflash_state_lock(self) -> asyncio.Lock:
        if self._dflash_state_lock is None:
            self._dflash_state_lock = asyncio.Lock()
        return self._dflash_state_lock

    def _next_dflash_concurrency(self) -> int:
        return self._active_dflash_requests + 1

    @asynccontextmanager
    async def _track_active_dflash_request(self):
        async with self._get_dflash_state_lock():
            self._active_dflash_requests += 1
        try:
            yield
        finally:
            async with self._get_dflash_state_lock():
                self._active_dflash_requests = max(0, self._active_dflash_requests - 1)

    def _get_dflash_executor(self) -> Any | None:
        engine_core = getattr(self, "_engine", None)
        executor = getattr(engine_core, "_mlx_executor", None)
        if executor is not None:
            return executor
        try:
            from ..engine_core import get_mlx_executor
            return get_mlx_executor()
        except Exception:
            return None

    def _consume_dflash_telemetry(self, runtime: DFlashRuntime | None) -> dict[str, Any] | None:
        if runtime is None:
            return None
        consume = getattr(runtime, "consume_last_telemetry", None)
        if not callable(consume):
            return None
        telemetry = consume()
        if isinstance(telemetry, dict) and telemetry:
            return telemetry
        return None

    def _with_dflash_telemetry(
        self,
        metadata: dict[str, Any] | None,
        telemetry: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if metadata is None or not telemetry:
            return metadata
        dflash = metadata.get("dflash")
        if not isinstance(dflash, dict):
            return metadata
        merged = dict(metadata)
        merged_dflash = dict(dflash)
        merged_dflash["telemetry"] = telemetry
        merged["dflash"] = merged_dflash
        return merged

    def _build_dflash_metadata(
        self,
        *,
        request_override: bool | None,
        stream: bool,
        sampling_params,
        used: bool = False,
        reason_override: str | None = None,
        concurrent_requests: int = 1,
    ) -> dict[str, Any] | None:
        model_enabled = bool(
            self._model_settings is not None
            and getattr(self._model_settings, "dflash_enabled", False)
        )
        requested = self._should_use_dflash(request_override)
        if not requested and not model_enabled and request_override is not False:
            return None

        runtime = self._dflash_runtime
        backend = None
        if runtime is not None:
            backend = getattr(runtime.config, "draft_backend", None)

        if reason_override is not None:
            reason = reason_override
        elif request_override is False and model_enabled:
            reason = "DFlash disabled by request override"
        elif requested:
            if used:
                reason = None
            elif runtime is None:
                reason = "DFlash runtime not configured"
            else:
                _supported, reason = runtime.request_is_supported(
                    stream=stream,
                    concurrent_requests=concurrent_requests,
                    sampling_params=sampling_params,
                )
        else:
            reason = None

        return {
            "dflash": {
                "requested": requested,
                "used": used,
                "reason": reason,
                "backend": backend,
            }
        }

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        engine_core = getattr(self, "_engine", None)
        if engine_core is not None:
            inner = getattr(engine_core, "engine", None)
            if inner is not None:
                collectors = getattr(inner, "_output_collectors", {})
                return len(collectors) > 0
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        if self._dflash_runtime is not None:
            stats["dflash"] = self._dflash_runtime.status()
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._engine:
            return self._engine.get_cache_stats()
        return None

    async def abort_all_requests(self) -> int:
        """Abort all active requests without stopping the engine."""
        if self._engine and self._engine.engine:
            return await self._engine.engine.abort_all_requests()
        return 0
