"""Pipelines - use cases that compose services.

A pipeline is just a service whose run() composes other services, so these
subclass BaseService (there is no separate pipeline base).
"""

from contextchecker.pipelines.refchecker import RefCheckerPipeline
from contextchecker.pipelines.ragchecker import RagCheckerPipeline
from contextchecker.pipelines.directions import run_direction

__all__ = ["RefCheckerPipeline", "RagCheckerPipeline", "run_direction"]
