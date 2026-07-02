"""
BaseService — abstract blueprint for all service classes.

Every service MUST implement the abstract methods below.
Attempting to instantiate a service without overriding them
raises TypeError at construction time — not at runtime.

Concrete freebies provided by the base:
- run_sync()         → asyncio.run() wrapper, identical for all services
- _require_api_key() → fail-fast config validation, reusable

Note on signatures: @abstractmethod enforces existence, not signature.
Subclasses override with their own parameter/return types.
"""

import asyncio
from abc import ABC, abstractmethod

from contextchecker.exceptions import InvalidInputError


class BaseService(ABC):
    """Blueprint that all services inherit from.

    Subclasses define their own pipeline steps. The base enforces
    that key steps exist and provides universal helpers.
    """

    # ── Contract (must override — TypeError if you forget) ───────

    @abstractmethod
    async def run(self, data: list[dict]) -> list[dict]:
        """Execute the full pipeline.

        Every service's run() should follow this shape:
        1. Validate input data
        2. Filter already-processed items (optional)
        3. Log pre-execution state (validation, config)
        4. Execute (delegate to worker)
        5. Serialize results back into the dicts
        6. Log post-execution results (stats, done line)
        7. Return mutated data
        """

    @abstractmethod
    def _validate(self, data: list[dict]) -> list[dict]:
        """Validate input data. Return the valid subset.

        Raise InvalidInputError if nothing is valid.
        """

    @abstractmethod
    def _filter(self, valid: list[dict]):
        """Filter already-processed items.
        """

    @abstractmethod
    def _log_config(self) -> None:
        """Log the service's configuration (model, prompts, etc).

        Called before execution starts so the user sees what
        they're about to run.
        """

    @abstractmethod
    def _log_results(self, *args, **kwargs) -> None:
        """Log post-execution results.

        This is the "don't forget to report" guard.
        Internal decomposition (_log_bl_results, _log_done, etc.)
        is up to each service.
        """

    @abstractmethod
    def _serialize(self, *args, **kwargs) -> None:
        """Write results back into the source dicts.
        
        If a service does not need serialization, it can override this
        with a simple `pass`.
        """

    @abstractmethod
    def _log_validation(self, *args, **kwargs) -> None:
        """Print the 📂 Validation section."""

    @abstractmethod
    def _log_skip(self, *args, **kwargs) -> None:
        """Print the 🔄 Skip section."""

    @abstractmethod
    def _log_bl_results(self, *args, **kwargs) -> None:
        """Print the domain-specific business logic results.
        
        e.g., "📝 Extraction Result summary".
        """

    @abstractmethod
    def _log_done(self, *args, **kwargs) -> None:
        """Print the ✅ Done summary line."""

    # ── Freebies (inherited as-is) ───────────────────────────────

    def run_sync(self, data: list[dict]) -> list[dict]:
        """Sync wrapper for CLI and facade consumers."""
        return asyncio.run(self.run(data))

    @staticmethod
    def _require_api_key(key: str | None, name: str) -> str:
        """Fail-fast: raise immediately if a required API key is missing. Usually called from constructor.

        Returns the key if present, so callers can do:
            self.api_key = self._require_api_key(settings.EXTRACTOR_API_KEY, "EXTRACTOR_API_KEY")
        """
        if not key:
            raise InvalidInputError(
                f"{name} is required. Set it in your .env file."
            )
        return key

    @staticmethod
    def _canonicalize_keys(data: list[dict]) -> None:
        """Step 0: Normalize common key aliases in-place.

        Different datasets use different names for the same concept.
        This normalizes them to our canonical keys before any service
        logic runs. 

        Mappings:
            context → reference
            query   → question
        """
        for item in data:
            if "context" in item and "reference" not in item:
                item["reference"] = item.pop("context")
            if "query" in item and "question" not in item:
                item["question"] = item.pop("query")
