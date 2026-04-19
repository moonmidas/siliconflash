from __future__ import annotations

from omlx.dflash.recurrent_rollback_cache import RecurrentRollbackCache


class _FakeArray:
    def __init__(self, label, shape):
        self.label = label
        self.shape = shape
        self.dtype = "fake"

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return _FakeArray(f"{self.label}[{item}]", self.shape)
        pieces = []
        new_shape = list(self.shape)
        for axis, value in enumerate(item):
            if isinstance(value, slice):
                start = 0 if value.start is None else value.start
                stop = self.shape[axis] if value.stop is None else value.stop
                pieces.append(f"{'' if value.start is None else value.start}:{'' if value.stop is None else value.stop}")
                if axis < len(new_shape):
                    new_shape[axis] = max(0, stop - start)
            else:
                pieces.append(str(value))
        return _FakeArray(f"{self.label}[{','.join(pieces)}]", tuple(new_shape))

    def __repr__(self):
        return f"_FakeArray({self.label})"


class _FakeMX:
    @staticmethod
    def contiguous(value):
        return value

    @staticmethod
    def zeros(shape, dtype=None):
        return _FakeArray(f"zeros{shape}", shape)

    @staticmethod
    def concatenate(values, axis=0):
        labels = ",".join(getattr(v, "label", repr(v)) for v in values)
        first_shape = list(getattr(values[0], "shape", (1, 1, 1)))
        if axis < len(first_shape):
            first_shape[axis] = sum(getattr(v, "shape", first_shape)[axis] for v in values)
        return _FakeArray(f"concat(axis={axis}:{labels})", tuple(first_shape))

    @staticmethod
    def array(value):
        if isinstance(value, _FakeArray):
            return _FakeArray(f"copy({value.label})", value.shape)
        return value


class _FakeTapeReplay:
    def __call__(self, tape, k, g, state, mask):
        return _FakeArray(
            f"replayed(tape={tape.label},k={k.label},g={g.label},state={state.label})",
            getattr(state, "shape", (1, 1, 1, 1)),
        )


def test_rollback_replays_only_accepted_prefix_and_rebuilds_conv_state():
    fake_mx = _FakeMX()
    fake_replay = _FakeTapeReplay()
    cache = RecurrentRollbackCache(size=2, conv_kernel_size=4, mx_module=fake_mx, tape_replay_fn=fake_replay)
    cache.cache = [
        _FakeArray("conv-live", (1, 3, 8)),
        _FakeArray("ssm-live", (1, 2, 4, 4)),
    ]

    cache.arm_rollback()
    cache.record_tape(
        tape=_FakeArray("tape", (1, 5, 2, 4)),
        k=_FakeArray("k", (1, 5, 1, 4)),
        g=_FakeArray("g", (1, 5, 2)),
        qkv=_FakeArray("qkv", (1, 5, 8)),
    )

    cache.rollback(2)

    assert cache.cache[1].label == "replayed(tape=tape[:,:3],k=k[:,:3],g=g[:,:3],state=copy(ssm-live))"
    assert cache.cache[0].label == "concat(axis=1:copy(conv-live),qkv[:,:3,:])[:,3:6,:]"


def test_rollback_without_tape_restores_snapshot_only():
    fake_mx = _FakeMX()
    cache = RecurrentRollbackCache(size=2, conv_kernel_size=4, mx_module=fake_mx, tape_replay_fn=_FakeTapeReplay())
    cache.cache = [
        _FakeArray("conv-live", (1, 3, 8)),
        _FakeArray("ssm-live", (1, 2, 4, 4)),
    ]

    cache.arm_rollback()
    cache.cache = [
        _FakeArray("conv-mutated", (1, 3, 8)),
        _FakeArray("ssm-mutated", (1, 2, 4, 4)),
    ]

    cache.rollback(1)

    assert cache.cache[0].label == "copy(conv-live)"
    assert cache.cache[1].label == "copy(ssm-live)"
