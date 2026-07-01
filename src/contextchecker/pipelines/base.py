"""
BasePipeline - composes services into one run over one dataset.

Rules (documented here, in code, not in a separate ADR):
  * A pipeline calls services only - never a worker directly.
  * Each worker has exactly one owning service.

Differences from BaseService:
  * Logging is provided but OPTIONAL. A pipeline delegates to services that
    already log, so it may stay silent (RefChecker) or add its own frame
    (the evaluators). BaseService makes logging mandatory; here it is not.
  * Validation is STANDARDIZED. A pipeline only declares _required_keys();
    the base drops items missing them and fails fast. No pipeline writes
    its own validation logic.
"""

import asyncio
from abc import ABC, abstractmethod

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.services.base import BaseService


class BasePipeline(ABC):
    """Blueprint for pipelines (use cases). Subclasses compose services in
    run() and declare their input contract in _required_keys()."""

    quiet: bool = False
    logger = settings.get_logger(__name__)

    # -- Contract (must override) --

    @abstractmethod
    async def run(self, data: list[dict]) -> list[dict]:
        """Compose services over *data*; return the mutated data."""

    @abstractmethod
    def _required_keys(self) -> tuple[str, ...]:
        """Input fields every item must carry - the union of the stages'
        input requirements."""

    # -- Standardized validation (do NOT override) --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing required fields. Return the valid
        subset. Raise InvalidInputError if none remain."""
        required = self._required_keys()
        valid = [item for item in data if all(k in item for k in required)]
        if not valid:
            raise InvalidInputError(
                f"No items contain all required fields: {', '.join(required)}."
            )
        return valid

    # -- Optional logging (provided, no-op by default) --

    def _log_config(self) -> None:
        """Optional pre-run frame. Override to log; default is silent."""

    def _log_summary(self, data: list[dict]) -> None:
        """Optional post-run summary. Override to log; default is silent."""

    # -- Freebies --

    # One definition of key-alias normalization (context->reference, etc.).
    _canonicalize_keys = staticmethod(BaseService._canonicalize_keys)

    def run_sync(self, data: list[dict]) -> list[dict]:
        """Sync wrapper for callers outside an event loop."""
        return asyncio.run(self.run(data))
