"""Pipelines - use cases that compose services into one run."""

from contextchecker.pipelines.base import BasePipeline
from contextchecker.pipelines.refchecker import RefCheckerPipeline

__all__ = ["BasePipeline", "RefCheckerPipeline"]
