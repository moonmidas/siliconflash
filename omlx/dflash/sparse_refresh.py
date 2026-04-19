from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable


CloneFn = Callable[[Any], Any]


def apply_linear_ssm_refresh(
    *,
    layers: Iterable[Any],
    live_cache: list[Any],
    fresh_cache: list[Any],
    refresh_layer_ids: set[int] | None = None,
    linear_cutoff: int | None = None,
    clone_fn: CloneFn,
) -> list[int]:
    """Copy recurrent linear SSM state from ``fresh_cache`` into ``live_cache``.

    Only linear layers are considered. When ``refresh_layer_ids`` is provided,
    only those layer indices are refreshed. Otherwise ``linear_cutoff`` can be
    used to refresh all linear layers with index ``<= linear_cutoff``.

    Returns the list of updated layer indices.
    """
    updated: list[int] = []
    for idx, (layer, live_layer_cache, fresh_layer_cache) in enumerate(
        zip(layers, live_cache, fresh_cache)
    ):
        if not getattr(layer, "is_linear", False):
            continue
        if refresh_layer_ids is not None:
            if idx not in refresh_layer_ids:
                continue
        elif linear_cutoff is not None and idx > linear_cutoff:
            continue
        live_layer_cache.cache[1] = clone_fn(fresh_layer_cache.cache[1])
        updated.append(idx)
    return updated
