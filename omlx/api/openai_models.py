# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from omlx.api.shared_models import (
    BaseUsage,
    IDPrefix,
    generate_id,
    get_unix_timestamp,
)


# =============================================================================
# Content Types
# =============================================================================

class ImageURL(BaseModel):
    """Image URL or base64 data URI for vision model input."""
    url: str  # "https://..." or "data:image/jpeg;base64,..."
    detail: Optional[str] = "auto"  # "low", "high", "auto"


class ContentPart(BaseModel):
    """
    A part of a message content array.

    Supports:
    - text: Plain text content
    - image_url: Image input for vision models
    """
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


# =============================================================================
# Messages
# =============================================================================

class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Content array messages (role + content list with text parts)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """
    role: str
    content: Optional[Union[str, List[ContentPart], List[dict]]] = None
    # For assistant messages with tool calls
    tool_calls: Optional[List[dict]] = None
    # For tool response messages (role="tool")
    tool_call_id: Optional[str] = None
    # Participant name, rendered into chat template (e.g. Kimi K2/K2.5 named assistants)
    name: Optional[str] = None
    # Continue from this message instead of starting a new turn (prefill / partial mode)
    partial: bool = False


# =============================================================================
# Tool Calling
# =============================================================================

class FunctionCall(BaseModel):
    """A function call with name and arguments."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""
    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""
    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================

class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""
    name: str
    description: Optional[str] = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: Optional[bool] = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """
    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: Optional[ResponseFormatJsonSchema] = None


class StructuredOutputOptions(BaseModel):
    """vLLM-compatible structured output options.

    Exactly one field should be set. When passed via ``extra_body`` in the
    OpenAI client, the key is ``structured_outputs``.

    Supports:
    - json: JSON schema (dict or string) for logit-level enforcement
    - regex: Regular expression the output must match
    - choice: List of allowed string values (output will be exactly one)
    - grammar: EBNF/GBNF context-free grammar string
    """
    model_config = {"populate_by_name": True}

    json_schema: Optional[Union[str, dict]] = Field(None, alias="json")
    regex: Optional[str] = None
    choice: Optional[List[str]] = None
    grammar: Optional[str] = None


# =============================================================================
# Chat Completion
# =============================================================================

class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str
    messages: List[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[List[str]] = None
    min_p: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Tool calling
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None  # "auto", "none", or specific tool
    # Structured output
    response_format: Optional[Union[ResponseFormat, dict]] = None
    # vLLM-compatible structured output (grammar, regex, choice, json)
    structured_outputs: Optional[Union[StructuredOutputOptions, dict]] = None
    # Chat template kwargs (e.g. enable_thinking, reasoning_effort)
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Thinking budget (max thinking tokens, None = unlimited)
    thinking_budget: Optional[int] = None
    # SpecPrefill: per-request enable/disable (None = use model setting)
    specprefill: Optional[bool] = None
    # SpecPrefill: per-request keep percentage (0.1-0.5, None = use model setting)
    specprefill_keep_pct: Optional[float] = None
    # SpecPrefill: per-request threshold override (min tokens to trigger, None = use model setting)
    specprefill_threshold: Optional[int] = None
    # DFlash: per-request enable/disable (None = use model setting)
    dflash: Optional[bool] = None
    # Allow generation to continue past EOS for benchmark parity / diagnostics.
    ignore_eos: Optional[bool] = None
    # Seed for reproducible generation (best-effort)
    seed: Optional[int] = None

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        """Accept stop as a single string (OpenAI compat) and wrap in a list."""
        if isinstance(v, str):
            return [v]
        return v


class AssistantMessage(BaseModel):
    """Response message from the assistant."""
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""
    index: int = 0
    message: AssistantMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseUsage):
    """Token usage statistics for OpenAI API.

    Extends BaseUsage with optional timing metrics (oMLX extension).
    When present, timing values are in seconds.
    """

    cached_tokens: Optional[int] = None
    # Timing metrics (oMLX extension, seconds)
    model_load_duration: Optional[float] = None
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    generation_duration: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    # DFlash backend diagnostics (oMLX extension)
    dflash_requested: Optional[bool] = None
    dflash_used: Optional[bool] = None
    dflash_reason: Optional[str] = None
    dflash_backend: Optional[str] = None
    # Optional per-request speculative telemetry (oMLX extension)
    dflash_draft_steps: Optional[int] = None
    dflash_drafted_tokens: Optional[int] = None
    dflash_accepted_tokens: Optional[int] = None
    dflash_acceptance_rate: Optional[float] = None
    dflash_commit_events: Optional[int] = None
    dflash_mean_commit_tokens: Optional[float] = None
    dflash_full_accept_steps: Optional[int] = None
    dflash_full_accept_rate: Optional[float] = None
    dflash_verify_passes: Optional[int] = None
    dflash_target_forward_passes: Optional[int] = None
    dflash_block_tokens_mean: Optional[float] = None
    dflash_block_tokens_min: Optional[int] = None
    dflash_block_tokens_max: Optional[int] = None
    dflash_prefill_s: Optional[float] = None
    dflash_draft_s: Optional[float] = None
    dflash_verify_s: Optional[float] = None
    dflash_eval_s: Optional[float] = None
    dflash_total_s: Optional[float] = None
    dflash_cache_restore_s: Optional[float] = None
    dflash_cache_rollback_calls: Optional[int] = None
    dflash_cache_trim_calls: Optional[int] = None
    dflash_cache_trim_tokens: Optional[int] = None
    dflash_cache_full_accept_clears: Optional[int] = None
    dflash_split_full_attention_calls: Optional[int] = None
    dflash_split_path_calls: Optional[int] = None
    dflash_split_path_hit_rate: Optional[float] = None
    dflash_split_exact_prefix_calls: Optional[int] = None
    dflash_split_batched_2pass_calls: Optional[int] = None
    dflash_split_batched_2pass_fallback_calls: Optional[int] = None
    dflash_split_query_chunks: Optional[int] = None
    dflash_async_submit_calls: Optional[int] = None
    dflash_async_submit_to_consume_samples: Optional[int] = None
    dflash_async_submit_to_consume_mean_s: Optional[float] = None
    dflash_async_submit_to_consume_max_s: Optional[float] = None
    dflash_async_submit_to_consume_min_s: Optional[float] = None
    dflash_async_submit_unconsumed_steps: Optional[int] = None
    dflash_draft_submit_steps: Optional[int] = None
    dflash_draft_submit_mean_s: Optional[float] = None
    dflash_draft_submit_min_s: Optional[float] = None
    dflash_draft_submit_max_s: Optional[float] = None
    dflash_draft_sync_eval_wait_steps: Optional[int] = None
    dflash_draft_sync_eval_wait_mean_s: Optional[float] = None
    dflash_draft_sync_eval_wait_min_s: Optional[float] = None
    dflash_draft_sync_eval_wait_max_s: Optional[float] = None
    dflash_verify_submit_steps: Optional[int] = None
    dflash_verify_submit_mean_s: Optional[float] = None
    dflash_verify_submit_min_s: Optional[float] = None
    dflash_verify_submit_max_s: Optional[float] = None
    dflash_verify_host_gap_steps: Optional[int] = None
    dflash_verify_host_gap_mean_s: Optional[float] = None
    dflash_verify_host_gap_min_s: Optional[float] = None
    dflash_verify_host_gap_max_s: Optional[float] = None
    dflash_verify_eval_wait_steps: Optional[int] = None
    dflash_verify_eval_wait_mean_s: Optional[float] = None
    dflash_verify_eval_wait_min_s: Optional[float] = None
    dflash_verify_eval_wait_max_s: Optional[float] = None
    dflash_verify_eval_wait_fused_steps: Optional[int] = None
    dflash_verify_eval_wait_fused_mean_s: Optional[float] = None
    dflash_verify_eval_wait_fused_min_s: Optional[float] = None
    dflash_verify_eval_wait_fused_max_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_steps: Optional[int] = None
    dflash_verify_eval_wait_unfused_mean_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_min_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_max_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_target_posterior_steps: Optional[int] = None
    dflash_verify_eval_wait_unfused_target_posterior_mean_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_target_posterior_min_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_target_posterior_max_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_draft_logits_steps: Optional[int] = None
    dflash_verify_eval_wait_unfused_draft_logits_mean_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_draft_logits_min_s: Optional[float] = None
    dflash_verify_eval_wait_unfused_draft_logits_max_s: Optional[float] = None
    dflash_verify_eval_fused_steps: Optional[int] = None
    dflash_verify_eval_unfused_steps: Optional[int] = None
    dflash_ddtree_enabled: Optional[int] = None
    dflash_ddtree_native_runtime: Optional[int] = None
    dflash_ddtree_tree_budget: Optional[int] = None
    dflash_ddtree_cycles_completed: Optional[int] = None
    dflash_ddtree_ddtree_cycles_completed: Optional[int] = None
    dflash_ddtree_dflash_cycles_completed: Optional[int] = None
    dflash_ddtree_dflash_accepted_from_draft: Optional[int] = None
    dflash_ddtree_avg_acceptance: Optional[float] = None
    dflash_ddtree_tokens_per_second: Optional[float] = None
    dflash_ddtree_fast_path_ratio: Optional[float] = None
    dflash_ddtree_fast_path_count: Optional[int] = None
    dflash_ddtree_slow_path_count: Optional[int] = None
    dflash_ddtree_tree_aware_commit_count: Optional[int] = None
    dflash_ddtree_tree_aware_linear: Optional[int] = None
    dflash_ddtree_exact_commit: Optional[int] = None
    dflash_ddtree_dflash_controller_enabled: Optional[int] = None
    dflash_ddtree_dflash_controller_probe_count: Optional[int] = None
    dflash_ddtree_dflash_controller_switch_count: Optional[int] = None
    dflash_ddtree_elapsed_s: Optional[float] = None
    dflash_ddtree_prefill_s: Optional[float] = None
    dflash_ddtree_tree_build_s: Optional[float] = None
    dflash_ddtree_tree_verify_s: Optional[float] = None
    dflash_ddtree_tree_verify_linear_s: Optional[float] = None
    dflash_ddtree_tree_verify_attention_s: Optional[float] = None
    dflash_ddtree_commit_s: Optional[float] = None
    dflash_ddtree_dflash_draft_s: Optional[float] = None
    dflash_ddtree_dflash_verify_s: Optional[float] = None
    dflash_ddtree_dflash_replay_s: Optional[float] = None
    dflash_ddtree_dflash_commit_s: Optional[float] = None
    dflash_thermal_sidecar_enabled: Optional[int] = None
    dflash_thermal_sidecar_sample_interval_s: Optional[float] = None
    dflash_thermal_sidecar_sample_timeout_s: Optional[float] = None
    dflash_thermal_sidecar_samples: Optional[int] = None
    dflash_thermal_sidecar_failures: Optional[int] = None
    dflash_thermal_sidecar_last_sample_age_s: Optional[float] = None
    dflash_thermal_sidecar_thermal_warning_samples: Optional[int] = None
    dflash_thermal_sidecar_performance_warning_samples: Optional[int] = None
    dflash_thermal_sidecar_cpu_power_status_samples: Optional[int] = None
    dflash_thermal_sidecar_thermal_warning_level_max: Optional[int] = None
    dflash_thermal_sidecar_performance_warning_level_max: Optional[int] = None
    dflash_thermal_sidecar_cpu_power_status_max: Optional[int] = None
    dflash_thermal_sidecar_cpu_speed_limit_samples: Optional[int] = None
    dflash_thermal_sidecar_cpu_speed_limit_mean_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_speed_limit_min_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_speed_limit_max_pct: Optional[float] = None
    dflash_thermal_sidecar_gpu_speed_limit_samples: Optional[int] = None
    dflash_thermal_sidecar_gpu_speed_limit_mean_pct: Optional[float] = None
    dflash_thermal_sidecar_gpu_speed_limit_min_pct: Optional[float] = None
    dflash_thermal_sidecar_gpu_speed_limit_max_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_scheduler_limit_samples: Optional[int] = None
    dflash_thermal_sidecar_cpu_scheduler_limit_mean_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_scheduler_limit_min_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_scheduler_limit_max_pct: Optional[float] = None
    dflash_thermal_sidecar_cpu_available_samples: Optional[int] = None
    dflash_thermal_sidecar_cpu_available_mean: Optional[float] = None
    dflash_thermal_sidecar_cpu_available_min: Optional[float] = None
    dflash_thermal_sidecar_cpu_available_max: Optional[float] = None
    dflash_mx_active_mem_mean_bytes: Optional[float] = None
    dflash_mx_active_mem_max_bytes: Optional[int] = None
    dflash_mx_cache_mem_mean_bytes: Optional[float] = None
    dflash_mx_cache_mem_max_bytes: Optional[int] = None
    dflash_mx_peak_mem_max_bytes: Optional[int] = None
    dflash_mx_recommended_working_set_bytes: Optional[int] = None
    dflash_mx_peak_over_recommended_ratio: Optional[float] = None
    dflash_mx_peak_over_recommended_events: Optional[int] = None
    # Collapse watchdog telemetry (oMLX extension)
    dflash_collapse_watchdog_enabled: Optional[int] = None
    dflash_collapse_spike_events: Optional[int] = None
    dflash_collapse_severe_spike_events: Optional[int] = None
    dflash_collapse_safe_mode_active: Optional[int] = None
    dflash_collapse_safe_mode_activations: Optional[int] = None
    dflash_collapse_safe_mode_step: Optional[int] = None
    dflash_collapse_safe_mode_block_tokens: Optional[int] = None
    dflash_collapse_async_drain_calls: Optional[int] = None
    dflash_collapse_async_drain_s: Optional[float] = None
    dflash_collapse_clear_cache_calls: Optional[int] = None
    dflash_collapse_safe_sync_calls: Optional[int] = None
    dflash_collapse_safe_sync_s: Optional[float] = None
    dflash_collapse_eval_step_mean_s: Optional[float] = None
    dflash_collapse_eval_step_max_s: Optional[float] = None
    dflash_collapse_eval_step_min_s: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================

class CompletionRequest(BaseModel):
    """Request for text completion."""
    model: str
    prompt: Union[str, List[str]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[List[str]] = None
    min_p: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Seed for reproducible generation (best-effort)
    seed: Optional[int] = None

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        """Accept stop as a single string (OpenAI compat) and wrap in a list."""
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.COMPLETION))
    object: str = "text_completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================

class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=get_unix_timestamp)
    owned_by: str = "omlx"


class ModelsResponse(BaseModel):
    """Response for listing models."""
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================

class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""
    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""
    tools: List[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""
    name: str
    state: str
    transport: str
    tools_count: int
    error: Optional[str] = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""
    servers: List[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""
    tool_name: str
    content: Optional[Union[str, list, dict]] = None
    is_error: bool = False
    error_message: Optional[str] = None


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================

class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[Usage] = None  # Present on last chunk when include_usage=true
