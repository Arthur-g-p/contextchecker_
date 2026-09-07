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
import copy
import time
from abc import ABC, abstractmethod

from claimlens import settings
from claimlens.exceptions import InvalidInputError
from claimlens.stats import (
    VarianceTracker,
    document_meta,
    log_multi_run_hint,
    log_run_line,
)

logger = settings.get_logger(__name__)

# The one printing knob. Booleans multiply, levels scale:
#   full    — everything, the classic standalone output (default)
#   compact — section rule (with label) + API/BL results; no pre-exec
#             sections, no token table, no done line. For pipeline children.
#   silent  — no log lines at all (progress bars are not logging and remain).
#             For repeated runs and library calls.
VERBOSITY_LEVELS = ("full", "compact", "silent")


class BaseService(ABC):
    """Blueprint that all services inherit from.

    Subclasses define their own pipeline steps. The base enforces
    that key steps exist and provides universal helpers.

    Pipelines use this base too. A "pipeline" is just a service whose run()
    composes other services (e.g. extraction then checking) instead of driving
    a single worker. The caller-facing interface is identical - data in,
    mutated data out - so no separate base is needed; "pipeline" is only a
    label for a service that orchestrates services, whatever the best
    technical term.
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

    def _init_verbosity(self, verbosity: str, section_label: str | None = None) -> None:
        """Validate and store the printing level + optional section label.

        Call from every service/pipeline constructor. Fail-fast on typos —
        a silent fallback to 'full' would hide the mistake in log spam.
        """
        if verbosity not in VERBOSITY_LEVELS:
            raise InvalidInputError(
                f"verbosity must be one of {VERBOSITY_LEVELS}, got '{verbosity}'."
            )
        self.verbosity = verbosity
        self.section_label = section_label

    def run_sync(self, data: list[dict]) -> list[dict]:
        """Sync wrapper for CLI and facade consumers."""
        return asyncio.run(self.run(data))

    # One-line run summary. Presence-filtered: only keys that exist in a
    # pipeline's metrics are shown. Override when a subclass's
    # headline metric is not listed.
    _RUN_SUMMARY_KEYS = ("precision", "recall", "f1", "faithfulness")

    def _log_run_findings(self) -> None:
        """Hook: called after each run in variance mode, before the run
        line. Default prints nothing; pipelines override to show their
        per-run findings (the metrics block) — plumbing, tokens, and the
        done line stay out of runs-mode."""

    async def _run_repeated(self, data: list[dict]) -> list[dict]:
        """Run _run_once N times (N = --runs, default 1) and assemble the two
        documents the CLI writes:

            last_report    {_meta, metrics, variance, runs: [{_meta, metrics, counts, items}]}
            last_findings  {_meta, runs: [{_meta, findings}]}

        The skeleton never changes with --runs: ``metrics`` is the mean over
        runs (the run's own values at N = 1), ``variance`` the spread, ``runs``
        one complete entry per run. Subclass contract: async
        ``_run_once(data, announce, report)`` sets ``self.last_run`` and
        ``self.last_run_findings`` (one run entry each) and reads
        ``self._runs``. Metric-agnostic: the variance roster comes from the
        subclass's ``_VARIANCE_SECTIONS``.
        """
        runs = self._runs
        full = self.verbosity == "full"
        if runs > 1 and full:
            log_multi_run_hint(runs)
        tracker = VarianceTracker(
            getattr(self, "_VARIANCE_SECTIONS", None),
            labels=getattr(self, "_VARIANCE_LABELS", None),
            directions=getattr(self, "_METRIC_DIRECTIONS", None))
        # Run 1 mutates *data* in place (the run() contract); a copy taken
        # after it would carry its results and the skip logic would no-op
        # runs 2..N — so the pristine copy is taken before.
        pristine = copy.deepcopy(data) if runs > 1 else None
        run_docs: list[dict] = []
        finding_docs: list[dict] = []

        for run in range(1, runs + 1):
            if runs > 1 and full:
                logger.info("")
                logger.info(settings.section_rule(f"Run {run}/{runs}"))
            working = data if run == 1 else copy.deepcopy(pristine)
            started = time.perf_counter()
            # One run prints its full results block (report=True); in runs
            # mode the findings hook + run line print instead.
            await self._run_once(working, announce=(run == 1), report=(runs == 1))
            duration = round(time.perf_counter() - started, 1)
            for doc in (self.last_run, self.last_run_findings):
                doc["_meta"]["run"] = run
                doc["_meta"]["duration_seconds"] = duration
            run_docs.append(self.last_run)
            finding_docs.append(self.last_run_findings)
            tracker.add(self.last_run["metrics"], duration)
            if runs > 1 and full:
                self._log_run_findings()
                log_run_line(run, runs, duration, self.last_run["metrics"],
                             self._RUN_SUMMARY_KEYS)

        means, variance = tracker.finish(runs, log=(runs > 1 and full))
        meta = document_meta(run_docs, runs, tracker.total_seconds)
        self.last_report = {"_meta": meta, "metrics": means,
                            "variance": variance, "runs": run_docs}
        self.last_findings = {"_meta": dict(meta), "runs": finding_docs}
        return data

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
