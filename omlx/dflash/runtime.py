from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

try:
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
except ImportError:
    NaiveStreamingDetokenizer = None

from ..request import RequestOutput, SamplingParams
from .bstnxbt_runtime import execute_bstnxbt_mlx_generate, iterate_bstnxbt_mlx_generate_commits
from .ddtree_runtime import ddtree_runtime_availability_reason, execute_ddtree_mlx_generate
from .mirror_sd_runtime import execute_mirror_sd_mlx_generate, iterate_mirror_sd_mlx_generate_commits
from .config import DFlashConfig
from .context_fusion import TargetContextFusion
from .drafter import DFlashDrafter, execute_mlx_hybrid_spec_generate, execute_zlab_spec_generate
from .interfaces import DFlashMetrics
from .target_bridge import MLXQwenDFlashTargetBridge
from .verify import DFlashVerifier

logger = logging.getLogger(__name__)


def _create_streaming_detokenizer(tokenizer: Any) -> Any | None:
    try:
        if hasattr(tokenizer, "detokenizer"):
            detok = tokenizer.detokenizer
        elif NaiveStreamingDetokenizer is not None:
            detok = NaiveStreamingDetokenizer(tokenizer)
        else:
            return None
        detok.reset()
        return detok
    except Exception:
        return None


def _get_think_token_id(tokenizer: Any, attr: str) -> int | None:
    try:
        return getattr(tokenizer, attr, None)
    except (TypeError, ValueError):
        return None


def _resolve_think_end_token_ids(tokenizer: Any) -> list[int] | None:
    think_end_id = _get_think_token_id(tokenizer, "think_end_id")
    if think_end_id is not None:
        return [think_end_id]
    think_end_str = getattr(tokenizer, "think_end", "</think>")
    try:
        ids = tokenizer.encode(think_end_str, add_special_tokens=False)
        if ids:
            return list(ids)
    except Exception:
        pass
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
        if tid != getattr(tokenizer, "unk_token_id", None):
            return [tid]
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def _prompt_needs_think_prefix(tokenizer: Any, prompt_token_ids: list[int]) -> bool:
    think_start_id = _get_think_token_id(tokenizer, "think_start_id")
    if think_start_id is None:
        try:
            think_start_id = tokenizer.convert_tokens_to_ids("<think>")
            if think_start_id == getattr(tokenizer, "unk_token_id", None):
                return False
        except (AttributeError, KeyError, TypeError):
            return False
    if not think_start_id or not prompt_token_ids:
        return False
    last_tokens = list(prompt_token_ids[-3:])
    if think_start_id not in last_tokens:
        return False
    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1:]
    if after_start:
        think_end_ids = _resolve_think_end_token_ids(tokenizer)
        if think_end_ids and think_end_ids[0] in after_start:
            return False
    return True


class DFlashRuntime:
    """Coordinator for native DFlash execution inside oMLX."""

    def __init__(
        self,
        config: DFlashConfig,
        target_model: Any,
        drafter_model: Any | None = None,
        verify_kernel: Any | None = None,
    ):
        config.validate()
        self.config = config
        self.target_model = target_model
        self.drafter_model = drafter_model
        self.verify_kernel = verify_kernel
        self.metrics = DFlashMetrics()
        self.context_fusion = TargetContextFusion(
            layer_indices=tuple(config.conditioning_layer_indices),
            projection_dim=config.conditioning_projection_dim,
        )
        self.drafter = DFlashDrafter(drafter_model, config) if drafter_model is not None else None
        self.verifier = DFlashVerifier(target_model, kernel=verify_kernel)
        self.target_bridge = None
        bridge_reason = MLXQwenDFlashTargetBridge.availability_reason()
        target_layer_ids = None
        if drafter_model is not None:
            draft_inner = getattr(drafter_model, "model", drafter_model)
            target_layer_ids = getattr(draft_inner, "target_layer_ids", None)
        if bridge_reason is None:
            try:
                self.target_bridge = MLXQwenDFlashTargetBridge(target_model, target_layer_ids=target_layer_ids)
            except Exception as e:
                bridge_reason = f"failed to initialize MLX target bridge: {e}"
        self._bridge_reason = bridge_reason
        self._availability_reason = self._compute_availability_reason()
        self._last_telemetry: dict[str, Any] | None = None

    def _compute_availability_reason(self) -> str | None:
        if self.drafter is None:
            draft_error = getattr(self, "_draft_load_error", None)
            if draft_error:
                return f"draft model not loaded: {draft_error}"
            return "draft model not loaded"
        if not self.drafter.ready:
            return "draft model does not expose a runnable DFlash interface"
        if self.config.draft_backend == "zlab_spec_generate":
            if self.target_bridge is None:
                return self._bridge_reason or "MLX target bridge unavailable"
            return None
        if self.config.draft_backend in ("bstnxbt_mlx", "mirror_sd_mlx"):
            return None
        if self.config.draft_backend == "ddtree_mlx":
            return ddtree_runtime_availability_reason()
        if self.verify_kernel is None:
            return "verify kernel not loaded"
        return None

    @property
    def ready(self) -> bool:
        return self._availability_reason is None

    @property
    def availability_reason(self) -> str | None:
        return self._availability_reason

    def _telemetry_enabled(self) -> bool:
        return os.environ.get("DFLASH_BSTNXBT_EMIT_TELEMETRY", "0") == "1"

    def _new_telemetry_buffer(self) -> dict[str, Any] | None:
        if not self._telemetry_enabled():
            return None
        return {}

    def _set_last_telemetry(self, telemetry: dict[str, Any] | None) -> None:
        if telemetry:
            self._last_telemetry = dict(telemetry)
        else:
            self._last_telemetry = None

    def consume_last_telemetry(self) -> dict[str, Any] | None:
        telemetry = self._last_telemetry
        self._last_telemetry = None
        return telemetry

    def request_is_supported(self, *, stream: bool, concurrent_requests: int, sampling_params: SamplingParams | None = None) -> tuple[bool, str | None]:
        if stream and self.config.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx"):
            return False, "DFlash streaming currently requires the bstnxbt_mlx or mirror_sd_mlx backend"
        if (
            self.config.single_batch_only
            and concurrent_requests > 1
            and self.config.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx")
        ):
            return False, "DFlash concurrent requests currently require the bstnxbt_mlx or mirror_sd_mlx interleaving path"
        if sampling_params is not None:
            if float(sampling_params.temperature) > 1e-5:
                return False, "DFlash currently supports greedy decoding only (temperature must be 0)"
            if int(sampling_params.top_k) not in (0, 1):
                return False, "DFlash currently supports greedy decoding only (top_k must be 0 or 1)"
            if float(sampling_params.min_p) > 1e-8:
                return False, "DFlash currently supports greedy decoding only (min_p must be 0)"
            if float(sampling_params.top_p) < 1.0 - 1e-8:
                return False, "DFlash currently supports greedy decoding only (top_p must be 1)"
            if sampling_params.stop:
                return False, "DFlash stop strings are not yet supported"
        if not self.ready:
            return False, self.availability_reason
        return True, None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "ready": self.ready,
            "availability_reason": self.availability_reason,
            "draft_model_path": self.config.draft_model_path,
            "draft_load_error": getattr(self, "_draft_load_error", None),
            "draft_backend": self.config.draft_backend,
            "block_size": self.config.block_size,
            "single_batch_only": self.config.single_batch_only,
            "bf16_only": self.config.bf16_only,
            "target_bridge": "ready" if self.target_bridge is not None else self._bridge_reason,
        }

    def _resolve_stop_and_suppress_token_ids(self, *, tokenizer: Any, sampling_params: SamplingParams) -> tuple[list[int], list[int] | None]:
        stop_token_ids = list(getattr(sampling_params, "stop_token_ids", []) or [])
        suppress_token_ids: list[int] | None = None
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if bool(getattr(sampling_params, "ignore_eos", False)):
            if eos_token_id is not None:
                suppress_token_ids = [int(eos_token_id)]
        else:
            if eos_token_id is not None and eos_token_id not in stop_token_ids:
                stop_token_ids.append(eos_token_id)
        return stop_token_ids, suppress_token_ids

    async def _iterate_bstnxbt_outputs(self, *, prompt: str, tokenizer: Any, sampling_params: SamplingParams, executor: Any | None = None):
        stop_token_ids, suppress_token_ids = self._resolve_stop_and_suppress_token_ids(
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        prompt_token_ids = list(tokenizer.encode(prompt))
        prompt_tokens = len(prompt_token_ids)
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        last_output_text = ""
        detokenizer = _create_streaming_detokenizer(tokenizer)
        prefix_text = ""
        if _prompt_needs_think_prefix(tokenizer, prompt_token_ids):
            prefix_text = f"{getattr(tokenizer, 'think_start', '<think>')}\n"
        prefix_emitted = False

        iterate_fn = iterate_bstnxbt_mlx_generate_commits
        if self.config.draft_backend == "mirror_sd_mlx":
            iterate_fn = iterate_mirror_sd_mlx_generate_commits

        telemetry_out = self._new_telemetry_buffer()

        iterator = await loop.run_in_executor(
            executor,
            lambda: iterate_fn(
                target_model=self.target_model,
                tokenizer=tokenizer,
                drafter_model=self.drafter.model,
                prompt=prompt,
                max_new_tokens=int(sampling_params.max_tokens),
                stop_token_ids=stop_token_ids or None,
                suppress_token_ids=suppress_token_ids,
                enable_thinking=bool(getattr(sampling_params, "enable_thinking", False)),
                thinking_budget=getattr(sampling_params, "thinking_budget", None),
                should_abort=stop_event.is_set,
                telemetry_out=telemetry_out,
            ),
        )

        def next_commit():
            try:
                return next(iterator), False
            except StopIteration:
                return None, True

        try:
            while True:
                item, done = await loop.run_in_executor(executor, next_commit)
                if done:
                    break
                new_token_ids, output_token_ids, finished, finish_reason = item
                if detokenizer is not None:
                    try:
                        new_text = ""
                        for token_id in list(new_token_ids):
                            detokenizer.add_token(token_id)
                            new_text += getattr(detokenizer, "last_segment", "")
                        if not new_text and new_token_ids:
                            detokenizer = None
                        else:
                            output_text = last_output_text + new_text
                    except Exception:
                        detokenizer = None
                if detokenizer is None:
                    output_text = tokenizer.decode(list(output_token_ids))
                    if output_text.startswith(last_output_text):
                        new_text = output_text[len(last_output_text):]
                    else:
                        new_text = tokenizer.decode(list(new_token_ids))
                if prefix_text and not prefix_emitted:
                    new_text = prefix_text + new_text
                    output_text = prefix_text + output_text
                    prefix_emitted = True
                last_output_text = output_text
                yield RequestOutput(
                    request_id="dflash-direct",
                    new_token_ids=list(new_token_ids),
                    new_text=new_text,
                    output_token_ids=list(output_token_ids),
                    output_text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(output_token_ids),
                    finished=bool(finished),
                    finish_reason=finish_reason,
                    cached_tokens=0,
                )
        finally:
            stop_event.set()
            self._set_last_telemetry(telemetry_out)

    def generate(self, *, prompt: str, tokenizer: Any, sampling_params: SamplingParams) -> RequestOutput:
        if not self.ready:
            raise RuntimeError(self.availability_reason or "DFlash runtime not ready")
        if self.drafter is None:
            raise RuntimeError("DFlash runtime missing drafter")

        stop_token_ids, suppress_token_ids = self._resolve_stop_and_suppress_token_ids(
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        prompt_token_ids = list(tokenizer.encode(prompt))

        if self.config.draft_backend in ("bstnxbt_mlx", "mirror_sd_mlx", "ddtree_mlx"):
            execute_fn = execute_bstnxbt_mlx_generate
            if self.config.draft_backend == "mirror_sd_mlx":
                execute_fn = execute_mirror_sd_mlx_generate
            elif self.config.draft_backend == "ddtree_mlx":
                execute_fn = execute_ddtree_mlx_generate

            telemetry_out = self._new_telemetry_buffer()
            output_token_ids = execute_fn(
                target_model=self.target_model,
                tokenizer=tokenizer,
                drafter_model=self.drafter.model,
                prompt=prompt,
                max_new_tokens=int(sampling_params.max_tokens),
                stop_token_ids=stop_token_ids or None,
                suppress_token_ids=suppress_token_ids,
                enable_thinking=bool(getattr(sampling_params, "enable_thinking", False)),
                thinking_budget=getattr(sampling_params, "thinking_budget", None),
                telemetry_out=telemetry_out,
            )
            self._set_last_telemetry(telemetry_out)
            output_text = tokenizer.decode(output_token_ids)
            if _prompt_needs_think_prefix(tokenizer, prompt_token_ids):
                output_text = f"{getattr(tokenizer, 'think_start', '<think>')}\n{output_text}"
            return RequestOutput(
                request_id="dflash-direct",
                output_token_ids=output_token_ids,
                output_text=output_text,
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(output_token_ids),
                finished=True,
                finish_reason="stop",
                cached_tokens=0,
            )

        if self.target_bridge is None:
            raise RuntimeError("DFlash runtime missing target bridge")

        import torch

        hf_drafter = getattr(self.drafter.model, "model", self.drafter.model)
        spec_generate = getattr(hf_drafter, "spec_generate", None)

        input_token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=self.target_bridge.device)

        if getattr(self.drafter.model, 'backend', '') == 'mlx_native_hybrid':
            output_ids = execute_mlx_hybrid_spec_generate(
                hf_drafter,
                target=self.target_bridge,
                input_ids=input_ids,
                max_new_tokens=int(sampling_params.max_tokens),
                stop_token_ids=stop_token_ids or None,
                temperature=float(sampling_params.temperature),
            )
        elif callable(spec_generate):
            output_ids = spec_generate(
                target=self.target_bridge,
                input_ids=input_ids,
                max_new_tokens=int(sampling_params.max_tokens),
                stop_token_ids=stop_token_ids or None,
                temperature=float(sampling_params.temperature),
            )
        else:
            output_ids = execute_zlab_spec_generate(
                hf_drafter,
                target=self.target_bridge,
                input_ids=input_ids,
                max_new_tokens=int(sampling_params.max_tokens),
                stop_token_ids=stop_token_ids or None,
                temperature=float(sampling_params.temperature),
            )

        self._set_last_telemetry(None)
        output_token_ids = output_ids[0, len(input_token_ids):].tolist()
        output_text = tokenizer.decode(output_token_ids)
        return RequestOutput(
            request_id="dflash-direct",
            output_token_ids=output_token_ids,
            output_text=output_text,
            prompt_tokens=len(input_token_ids),
            completion_tokens=len(output_token_ids),
            finished=True,
            finish_reason="stop",
            cached_tokens=0,
        )

    async def async_generate(self, *, prompt: str, tokenizer: Any, sampling_params: SamplingParams, executor: Any | None = None) -> RequestOutput:
        if not self.ready:
            raise RuntimeError(self.availability_reason or "DFlash runtime not ready")
        if self.drafter is None:
            raise RuntimeError("DFlash runtime missing drafter")

        if self.config.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                lambda: self.generate(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                ),
            )

        final_output: RequestOutput | None = None
        async for output in self._iterate_bstnxbt_outputs(
            prompt=prompt,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            executor=executor,
        ):
            final_output = output

        if final_output is not None:
            return final_output

        prompt_tokens = len(tokenizer.encode(prompt))
        return RequestOutput(
            request_id="dflash-direct",
            output_token_ids=[],
            output_text="",
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            finished=True,
            finish_reason="length",
            cached_tokens=0,
        )

    async def stream_generate(self, *, prompt: str, tokenizer: Any, sampling_params: SamplingParams, executor: Any | None = None):
        if not self.ready:
            raise RuntimeError(self.availability_reason or "DFlash runtime not ready")
        if self.drafter is None:
            raise RuntimeError("DFlash runtime missing drafter")
        if self.config.draft_backend not in ("bstnxbt_mlx", "mirror_sd_mlx"):
            raise RuntimeError("DFlash streaming currently requires bstnxbt_mlx or mirror_sd_mlx backend")

        async for output in self._iterate_bstnxbt_outputs(
            prompt=prompt,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            executor=executor,
        ):
            yield output
