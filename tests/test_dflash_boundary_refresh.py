from types import SimpleNamespace

from omlx.dflash.boundary_refresh import (
    restore_linear_boundary,
    set_boundary_capture_enabled,
    trim_attention_reject_suffix,
)


class _LinearCache:
    def __init__(self):
        self.cache = ["conv-live", "ssm-live"]
        self._boundary_snapshots = [
            ("conv-step-1", "ssm-step-1"),
            ("conv-step-2", "ssm-step-2"),
            ("conv-step-3", "ssm-step-3"),
        ]


class _AttentionCache:
    def __init__(self):
        self.trim_calls = []

    def trim(self, amount):
        self.trim_calls.append(amount)


def test_restore_linear_boundary_uses_accepted_prefix_snapshot():
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    caches = [_LinearCache(), _AttentionCache()]

    restore_linear_boundary(caches, layers, accepted_tokens_in_block=2, clone_fn=lambda x: f"cloned:{x}")

    assert caches[0].cache == ["cloned:conv-step-2", "cloned:ssm-step-2"]
    assert caches[1].trim_calls == []


def test_trim_attention_reject_suffix_only_trims_attention_layers():
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    caches = [_LinearCache(), _AttentionCache()]

    trim_attention_reject_suffix(caches, layers, reject_tokens=5, trim_fn=lambda cache, amount: cache.trim(amount))

    assert caches[1].trim_calls == [5]


def test_set_boundary_capture_enabled_marks_only_linear_layers():
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False), SimpleNamespace(is_linear=True)]
    caches = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]

    set_boundary_capture_enabled(caches, layers, enabled=True)

    assert caches[0]._capture_boundaries is True
    assert caches[0]._layer_idx == 0
    assert not hasattr(caches[1], "_capture_boundaries")
    assert caches[2]._capture_boundaries is True
    assert caches[2]._layer_idx == 2
