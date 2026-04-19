from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from omlx.dflash.mirror_sd_target import mirror_target_forward_with_hidden_states


class _AddLayer:
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, hidden_states, mask=None, cache=None):
        del mask, cache
        return hidden_states + self.delta


class _InnerModel:
    def __init__(self):
        self.layers = [_AddLayer(1.0), _AddLayer(2.0), _AddLayer(3.0)]

    def embed_tokens(self, input_ids):
        return input_ids.astype(mx.float32)[..., None]

    def norm(self, hidden_states):
        return hidden_states + 0.5


class _TargetModel:
    def __init__(self):
        self.model = _InnerModel()

    def __call__(self, hidden_states):
        return hidden_states * 2.0


def test_mirror_target_forward_with_hidden_states_captures_requested_layers(monkeypatch):
    monkeypatch.setattr("omlx.dflash.mirror_sd_target._target_text_model", lambda target_model: target_model.model)
    monkeypatch.setattr("omlx.dflash.mirror_sd_target._lm_head_logits", lambda target_model, hs: hs * 2.0)

    logits, captured = mirror_target_forward_with_hidden_states(
        _TargetModel(),
        input_ids=mx.array([[1, 2]], dtype=mx.uint32),
        cache=[None, None, None],
        capture_layer_ids={1, 2},
    )

    assert sorted(captured.keys()) == [1, 2]
    assert mx.allclose(captured[1], mx.array([[[2.0], [3.0]]]))
    assert mx.allclose(captured[2], mx.array([[[4.0], [5.0]]]))
    # embed -> +1 -> +2 -> +3 -> +0.5 norm -> *2 lm_head
    assert mx.allclose(logits, mx.array([[[15.0], [17.0]]]))


def test_mirror_target_forward_with_hidden_states_honors_exit_layer_env(monkeypatch):
    monkeypatch.setattr("omlx.dflash.mirror_sd_target._target_text_model", lambda target_model: target_model.model)
    monkeypatch.setattr("omlx.dflash.mirror_sd_target._lm_head_logits", lambda target_model, hs: hs * 2.0)
    monkeypatch.setenv("DFLASH_MIRROR_SD_EXIT_LAYER", "0")

    logits, captured = mirror_target_forward_with_hidden_states(
        _TargetModel(),
        input_ids=mx.array([[1, 2]], dtype=mx.uint32),
        cache=[None, None, None],
        capture_layer_ids={1, 2},
    )

    assert sorted(captured.keys()) == [1, 2]
    assert mx.allclose(logits, mx.array([[[15.0], [17.0]]]))
