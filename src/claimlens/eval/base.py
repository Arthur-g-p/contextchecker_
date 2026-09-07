"""
Evaluator base — the run loop and document assembly shared by the two
evaluators. Evaluators measure; they do not inherit BaseService (which
mutates data), so their common code lives here.

Subclasses provide the measurement (``_evaluate_once``) and the printing
of one run's Done line; this class owns --runs, the variance tracking and
the two documents:

    record    {_meta, metrics, variance, runs: [{_meta, metrics, counts, items}]}
    findings  {_meta, runs: [{_meta, findings}]}

The skeleton never changes with --runs: ``metrics`` is the mean over runs
(the run's own values at N = 1), ``variance`` the spread, ``runs`` one
complete entry per run. The findings document is a view derived from the
record's items — never a source of its own.
"""

import asyncio
import copy
import time

from claimlens import settings
from claimlens.stats import (
    VarianceTracker,
    document_meta,
    log_multi_run_hint,
    log_run_line,
    log_token_stats,
)

logger = settings.get_logger(__name__)


class Evaluator:
    """Blueprint for evaluators. Subclasses set the class attributes and
    implement ``_evaluate_once`` and ``_log_done``."""

    _runs: int = 1
    _RUN_SUMMARY_KEYS: tuple[str, ...] = ()
    _VARIANCE_SECTIONS: dict | None = None
    _VARIANCE_LABELS: dict | None = None
    _METRIC_DIRECTIONS: dict | None = None

    async def _evaluate_once(
        self, data: list[dict], announce: bool = True
    ) -> tuple[dict, dict]:
        """One measurement pass. Returns (run_doc, run_findings): one
        ``runs[]`` entry of the record — {_meta, metrics, counts, items} —
        and the matching findings entry {_meta, findings}."""
        raise NotImplementedError

    def _log_done(self, run_doc: dict) -> None:
        """Print the ✅ Done line for a single run."""
        raise NotImplementedError

    def _unmeasured(self) -> dict | None:
        """{metric key: reason} for axes this invocation did not run at all
        (rule 2.3: "not measured" is not "not computable"). Default: none."""
        return None

    async def evaluate(self, data: list[dict]) -> tuple[dict, dict]:
        """Run the eval N times (N = --runs, default 1) and assemble the
        record and findings documents the CLI writes verbatim."""
        runs = self._runs
        if runs > 1:
            log_multi_run_hint(runs)
        tracker = VarianceTracker(
            self._VARIANCE_SECTIONS, labels=self._VARIANCE_LABELS,
            unmeasured=self._unmeasured(), directions=self._METRIC_DIRECTIONS,
        )
        run_docs: list[dict] = []
        finding_docs: list[dict] = []
        for run in range(1, runs + 1):
            if runs > 1:
                logger.info("")
                logger.info(settings.section_rule(f"Run {run}/{runs}"))
            started = time.perf_counter()
            # Every run mutates its input (aliasing, verdict stripping,
            # extraction); in variance mode each run starts pristine.
            working = copy.deepcopy(data) if runs > 1 else data
            run_doc, run_findings = await self._evaluate_once(
                working, announce=(run == 1))
            duration = round(time.perf_counter() - started, 1)
            for doc in (run_doc, run_findings):
                doc["_meta"]["run"] = run
                doc["_meta"]["duration_seconds"] = duration
            run_docs.append(run_doc)
            finding_docs.append(run_findings)
            tracker.add(run_doc["metrics"], duration)
            if runs > 1:
                logger.info("")
                log_run_line(run, runs, duration, run_doc["metrics"],
                             self._RUN_SUMMARY_KEYS)

        # One run: the Done line closes the findings; N runs: the VARIANCE
        # block does. The token table is the appendix in both cases.
        means, variance = tracker.finish(runs, log=runs > 1)
        if runs == 1:
            self._log_done(run_docs[0])
            log_token_stats()

        meta = document_meta(run_docs, runs, tracker.total_seconds)
        record = {"_meta": meta, "metrics": means, "variance": variance,
                  "runs": run_docs}
        findings = {"_meta": dict(meta), "runs": finding_docs}
        return record, findings

    def run_sync(self, data: list[dict]) -> tuple[dict, dict]:
        """Sync wrapper — same pattern as BaseService.run_sync."""
        return asyncio.run(self.evaluate(data))
