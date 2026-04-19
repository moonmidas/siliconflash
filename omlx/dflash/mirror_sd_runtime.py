from __future__ import annotations

import os
from typing import Any

from .bstnxbt_runtime import execute_bstnxbt_mlx_generate, iterate_bstnxbt_mlx_generate_commits
from .mirror_sd_target import mirror_target_forward_with_hidden_states


def _mirror_use_default_target_forward() -> bool:
    raw = os.environ.get("DFLASH_BSTNXBT_MIRROR_USE_DEFAULT_TARGET_FORWARD", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def iterate_mirror_sd_mlx_generate_commits(
    *,
    target_model: Any,
    tokenizer: Any,
    drafter_model: Any,
    prompt: str,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    should_abort: Any | None = None,
    telemetry_out: dict[str, Any] | None = None,
):
    use_default_target_forward = _mirror_use_default_target_forward()
    target_forward_mode = "default" if use_default_target_forward else "mirror_sd_split"
    target_forward_fn_override = None if use_default_target_forward else mirror_target_forward_with_hidden_states

    yield from iterate_bstnxbt_mlx_generate_commits(
        target_model=target_model,
        tokenizer=tokenizer,
        drafter_model=drafter_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        should_abort=should_abort,
        target_forward_mode=target_forward_mode,
        target_forward_fn_override=target_forward_fn_override,
        telemetry_out=telemetry_out,
    )


def execute_mirror_sd_mlx_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    drafter_model: Any,
    prompt: str,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    on_commit: Any | None = None,
    should_abort: Any | None = None,
    telemetry_out: dict[str, Any] | None = None,
) -> list[int]:
    use_default_target_forward = _mirror_use_default_target_forward()
    target_forward_mode = "default" if use_default_target_forward else "mirror_sd_split"
    target_forward_fn_override = None if use_default_target_forward else mirror_target_forward_with_hidden_states

    return execute_bstnxbt_mlx_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        drafter_model=drafter_model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        on_commit=on_commit,
        should_abort=should_abort,
        target_forward_mode=target_forward_mode,
        target_forward_fn_override=target_forward_fn_override,
        telemetry_out=telemetry_out,
    )
