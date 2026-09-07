"""Pipelines - use cases that compose services.

A pipeline is just a service whose run() composes other services, so these
subclass BaseService (there is no separate pipeline base).
"""

from claimlens.pipelines.refchecker import RefCheckerPipeline
from claimlens.pipelines.ragchecker import RagCheckerPipeline
from claimlens.pipelines.faithfulness import FaithfulnessPipeline, check_faithfulness
from claimlens.pipelines.directions import run_direction

__all__ = [
    "RefCheckerPipeline",
    "RagCheckerPipeline",
    "FaithfulnessPipeline",
    "check_faithfulness",
    "run_direction",
]
