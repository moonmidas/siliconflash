from __future__ import annotations

from types import SimpleNamespace

from omlx.dflash.bstnxbt_runtime import (
    _CollapseWatchdogState,
    _ExactSmallProjPad,
    _install_exact_small_proj_hooks,
    _observe_eval_step,
    _resolve_block_tokens,
    _resolve_collapse_watchdog_config,
    make_target_cache,
)
from omlx.dflash.recurrent_rollback_cache import RecurrentRollbackCache


class _FakeLinear:
    def __init__(self):
        self.calls = []
        self.weight = "w"
        self.bias = None

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        return x


class _FakeArray:
    def __init__(self, shape, dtype="fake"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self
        shape = list(self.shape)
        for axis, value in enumerate(item):
            if isinstance(value, slice):
                start = 0 if value.start is None else value.start
                stop = self.shape[axis] if value.stop is None else value.stop
                shape[axis] = max(0, stop - start)
        return _FakeArray(shape, dtype=self.dtype)


class _FakeMX:
    @staticmethod
    def zeros(shape, dtype=None):
        return _FakeArray(shape, dtype=dtype)

    @staticmethod
    def concatenate(values, axis=0):
        shape = list(values[0].shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        return _FakeArray(shape, dtype=values[0].dtype)


def test_exact_small_proj_pad_pads_short_sequence_dimension():
    linear = _FakeLinear()
    wrapped = _ExactSmallProjPad(linear, mx_module=_FakeMX(), pad_m=16)

    output = wrapped(_FakeArray((1, 3, 4)))

    assert linear.calls == [(1, 16, 4)]
    assert output.shape == (1, 3, 4)


def test_install_exact_small_proj_hooks_wraps_a_and_b_once():
    linear_attn = SimpleNamespace(in_proj_a=_FakeLinear(), in_proj_b=_FakeLinear())

    _install_exact_small_proj_hooks(linear_attn, mx_module=_FakeMX(), pad_m=8)
    first_a = linear_attn.in_proj_a
    first_b = linear_attn.in_proj_b
    _install_exact_small_proj_hooks(linear_attn, mx_module=_FakeMX(), pad_m=8)

    assert isinstance(first_a, _ExactSmallProjPad)
    assert isinstance(first_b, _ExactSmallProjPad)
    assert linear_attn.in_proj_a is first_a
    assert linear_attn.in_proj_b is first_b


def test_make_target_cache_uses_recurrent_cache_for_linear_layers():
    target_model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(is_linear=True, linear_attn=SimpleNamespace(conv_kernel_size=5)),
                    SimpleNamespace(is_linear=False),
                    SimpleNamespace(is_linear=True, linear_attn=SimpleNamespace(conv_kernel_size=4)),
                ]
            )
        )
    )

    caches = make_target_cache(target_model)

    assert isinstance(caches[0], RecurrentRollbackCache)
    assert caches[0].conv_kernel_size == 5
    assert caches[1].__class__.__name__ == "KVCache"
    assert isinstance(caches[2], RecurrentRollbackCache)
    assert caches[2].conv_kernel_size == 4


def test_make_target_cache_can_use_kernel_tape_replay(monkeypatch):
    target_model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(is_linear=True, linear_attn=SimpleNamespace(conv_kernel_size=5))]
            )
        )
    )

    sentinel = object()
    monkeypatch.setenv("DFLASH_BSTNXBT_RECURRENT_KERNELS", "1")
    monkeypatch.setattr("omlx.dflash.bstnxbt_runtime.tape_replay_kernel", sentinel)

    caches = make_target_cache(target_model)

    assert caches[0]._tape_replay_fn is sentinel


def test_resolve_block_tokens_uses_env_override(monkeypatch):
    monkeypatch.setenv("DFLASH_BLOCK_TOKENS", "12")

    assert _resolve_block_tokens(SimpleNamespace(block_size=16)) == 12


def test_resolve_block_tokens_clamps_to_draft_block_size(monkeypatch):
    monkeypatch.setenv("DFLASH_BLOCK_TOKENS", "32")

    assert _resolve_block_tokens(SimpleNamespace(block_size=16)) == 16


def test_resolve_block_tokens_defaults_to_normal_eos_thinking_value(monkeypatch):
    monkeypatch.delenv("DFLASH_BLOCK_TOKENS", raising=False)

    assert _resolve_block_tokens(
        SimpleNamespace(block_size=16),
        enable_thinking=True,
        ignore_eos=False,
    ) == 11


def test_resolve_block_tokens_defaults_to_no_eos_thinking_value(monkeypatch):
    monkeypatch.delenv("DFLASH_BLOCK_TOKENS", raising=False)

    assert _resolve_block_tokens(
        SimpleNamespace(block_size=16),
        enable_thinking=True,
        ignore_eos=True,
    ) == 9


def test_resolve_block_tokens_clamps_thinking_tuned_value(monkeypatch):
    monkeypatch.delenv("DFLASH_BLOCK_TOKENS", raising=False)

    assert _resolve_block_tokens(
        SimpleNamespace(block_size=8),
        enable_thinking=True,
        ignore_eos=False,
    ) == 8


def test_resolve_collapse_watchdog_config_defaults(monkeypatch):
    monkeypatch.delenv("DFLASH_BSTNXBT_COLLAPSE_WATCHDOG", raising=False)
    monkeypatch.delenv("DFLASH_BSTNXBT_COLLAPSE_SAFE_BLOCK_TOKENS", raising=False)

    config = _resolve_collapse_watchdog_config(15)

    assert config.enabled is True
    assert config.safe_block_tokens == 13
    assert config.consecutive_spikes == 3
    assert config.async_drain_every_steps == 8


def test_resolve_collapse_watchdog_config_clamps_safe_block(monkeypatch):
    monkeypatch.setenv("DFLASH_BSTNXBT_COLLAPSE_SAFE_BLOCK_TOKENS", "99")

    config = _resolve_collapse_watchdog_config(15)

    assert config.safe_block_tokens == 15


def test_observe_eval_step_activates_safe_mode_on_consecutive_spikes(monkeypatch):
    monkeypatch.setenv("DFLASH_BSTNXBT_COLLAPSE_CONSECUTIVE_SPIKES", "2")
    monkeypatch.setenv("DFLASH_BSTNXBT_COLLAPSE_WARMUP_STEPS", "0")
    monkeypatch.setenv("DFLASH_BSTNXBT_COLLAPSE_SPIKE_RATIO", "1.5")
    config = _resolve_collapse_watchdog_config(15)
    state = _CollapseWatchdogState()

    assert _observe_eval_step(state=state, config=config, step_index=1, step_eval_s=0.10) is False
    assert _observe_eval_step(state=state, config=config, step_index=2, step_eval_s=0.20) is False
    assert _observe_eval_step(state=state, config=config, step_index=3, step_eval_s=0.25) is True

    assert state.safe_mode_active is True
    assert state.safe_mode_activations == 1
    assert state.safe_mode_step == 3
    assert state.spike_events >= 2
