from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import omlx.dflash.bstnxbt_runtime as runtime
from omlx.dflash.bstnxbt_runtime import target_forward_with_hidden_states
from omlx.dflash.recurrent_rollback_cache import RecurrentRollbackCache


class _FakeEmbed:
    def __call__(self, input_ids):
        return input_ids.astype(mx.float32)[..., None]

    def as_linear(self, hidden_states):
        return hidden_states


class _FakeLinearLayer:
    def __init__(self):
        self.is_linear = True
        self.calls = 0

    def __call__(self, hidden_states, mask=None, cache=None):
        self.calls += 1
        return hidden_states + 1


class _FakeNorm:
    def __call__(self, x):
        return x


class _FakeWrapper:
    def __init__(self, layer, *, hybrid: bool):
        model = SimpleNamespace(
            embed_tokens=_FakeEmbed(),
            layers=[layer],
            norm=_FakeNorm(),
        )
        if hybrid:
            model.fa_idx = 0
            model.ssm_idx = 0
        self.model = model
        self.args = SimpleNamespace(tie_word_embeddings=True)


class _FakeTarget:
    def __init__(self, layer, *, hybrid: bool):
        self.language_model = _FakeWrapper(layer, hybrid=hybrid)


def test_target_forward_with_hidden_states_delegates_linear_layers_to_layer_call(monkeypatch):
    layer = _FakeLinearLayer()
    target = _FakeTarget(layer, hybrid=True)
    cache = [RecurrentRollbackCache(size=2)]

    monkeypatch.setattr(runtime, "create_attention_mask", lambda hidden_states, cache_entry: None)
    monkeypatch.setattr(runtime, "create_ssm_mask", lambda hidden_states, cache_entry: None)

    logits, captured = target_forward_with_hidden_states(
        target,
        input_ids=mx.array([[1, 2]], dtype=mx.uint32),
        cache=cache,
        capture_layer_ids={1},
    )
    mx.eval(logits, *captured.values())

    assert layer.calls == 1
    assert 1 in captured
