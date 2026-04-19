from types import SimpleNamespace

from omlx.dflash.sparse_refresh import apply_linear_ssm_refresh


class _FakeCache:
    def __init__(self, conv, ssm):
        self.cache = [conv, ssm]


def test_apply_linear_ssm_refresh_only_updates_selected_linear_layers():
    layers = [
        SimpleNamespace(is_linear=True),
        SimpleNamespace(is_linear=False),
        SimpleNamespace(is_linear=True),
        SimpleNamespace(is_linear=True),
    ]
    live_cache = [
        _FakeCache("live-conv-0", "live-ssm-0"),
        _FakeCache("live-conv-1", "live-ssm-1"),
        _FakeCache("live-conv-2", "live-ssm-2"),
        _FakeCache("live-conv-3", "live-ssm-3"),
    ]
    fresh_cache = [
        _FakeCache("fresh-conv-0", "fresh-ssm-0"),
        _FakeCache("fresh-conv-1", "fresh-ssm-1"),
        _FakeCache("fresh-conv-2", "fresh-ssm-2"),
        _FakeCache("fresh-conv-3", "fresh-ssm-3"),
    ]

    updated = apply_linear_ssm_refresh(
        layers=layers,
        live_cache=live_cache,
        fresh_cache=fresh_cache,
        refresh_layer_ids={0, 3},
        clone_fn=lambda value: f"cloned:{value}",
    )

    assert updated == [0, 3]
    assert live_cache[0].cache == ["live-conv-0", "cloned:fresh-ssm-0"]
    assert live_cache[1].cache == ["live-conv-1", "live-ssm-1"]
    assert live_cache[2].cache == ["live-conv-2", "live-ssm-2"]
    assert live_cache[3].cache == ["live-conv-3", "cloned:fresh-ssm-3"]


def test_apply_linear_ssm_refresh_can_use_linear_cutoff_without_explicit_set():
    layers = [
        SimpleNamespace(is_linear=True),
        SimpleNamespace(is_linear=True),
        SimpleNamespace(is_linear=False),
        SimpleNamespace(is_linear=True),
    ]
    live_cache = [
        _FakeCache("live-conv-0", "live-ssm-0"),
        _FakeCache("live-conv-1", "live-ssm-1"),
        _FakeCache("live-conv-2", "live-ssm-2"),
        _FakeCache("live-conv-3", "live-ssm-3"),
    ]
    fresh_cache = [
        _FakeCache("fresh-conv-0", "fresh-ssm-0"),
        _FakeCache("fresh-conv-1", "fresh-ssm-1"),
        _FakeCache("fresh-conv-2", "fresh-ssm-2"),
        _FakeCache("fresh-conv-3", "fresh-ssm-3"),
    ]

    updated = apply_linear_ssm_refresh(
        layers=layers,
        live_cache=live_cache,
        fresh_cache=fresh_cache,
        linear_cutoff=1,
        clone_fn=lambda value: value,
    )

    assert updated == [0, 1]
    assert live_cache[0].cache[1] == "fresh-ssm-0"
    assert live_cache[1].cache[1] == "fresh-ssm-1"
    assert live_cache[3].cache[1] == "live-ssm-3"
