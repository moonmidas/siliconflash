from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .interfaces import DFlashMetrics


def metrics_to_dict(metrics: DFlashMetrics) -> dict[str, Any]:
    data = asdict(metrics)
    data["mean_acceptance_rate"] = metrics.mean_acceptance_rate
    return data
