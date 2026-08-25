"""Lean production orchestration for the CMU-VLN AI module.

Only contracts and the executable online orchestrator are exported.  Retired
monolithic code, mock stages, and experiment runners are intentionally absent
from the package import graph.
"""

from .contracts import (
    AcquisitionContext,
    Deadline,
    PipelineSummary,
    StageResult,
    TIME_BUDGET,
)


__all__ = [
    "AcquisitionContext",
    "Deadline",
    "PipelineSummary",
    "StageResult",
    "TIME_BUDGET",
]
