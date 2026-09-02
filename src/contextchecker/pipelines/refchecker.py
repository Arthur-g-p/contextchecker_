"""
RefChecker - reference-checking use case: extraction + checking in one run.

RefChecker is a *pipeline*: a service whose run() composes other services
(ExtractionService then CheckingService) instead of driving a single worker.
The caller-facing interface is identical, so it subclasses BaseService - there
is no separate pipeline base (see the note in services/base.py).

It talks to services only, never a worker. The two services consolidate their
results into the shared data in place; this class wraps them in the report
envelope (`_meta` + `results`) that every other report-producing command emits,
and the controller (cli.py) persists it from `last_report`.
"""

import time
from datetime import datetime

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.pipelines.directions import unwrap_items
from contextchecker.services.base import BaseService
from contextchecker.services.checking import CheckingService
from contextchecker.services.extraction import ExtractionService
from contextchecker.stats import GLOBAL_STATS
from contextchecker.utils import build_meta

logger = settings.get_logger(__name__)


class RefCheckerPipeline(BaseService):
    """Extraction + checking composed into a single reference-checking run."""

    def __init__(
        self,
        extractor_model: str,
        checker_model: str,
        *,
        extractor_base_url: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        dedup: bool = True,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        verbosity: str = "full",
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model
        self._init_verbosity(verbosity)
        self.last_report: dict | None = None

        # Compose the two services. Each fail-fasts on its own API key here.
        self._extraction = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            verbosity=verbosity,
            dedup=dedup,
        )
        self._checking = CheckingService(
            model=checker_model,
            extractor_model=extractor_model,
            base_url=checker_base_url,
            concurrency=concurrency,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            verbosity=verbosity,
        )

    # -- Pipeline: the BaseService 7-step run() shape --

    async def run(self, data: list[dict]) -> list[dict]:
        """Run extraction then checking over *data*, in place; return data.

        1. Validate     - drop items missing 'response' or 'reference'
        2. Filter       - none (pass); child services filter their own
        3. Log pre-exec - validation + config
        4. Execute      - delegate to ExtractionService then CheckingService
        5. Serialize    - none in place (the services consolidated into data
                          already); the report lands on last_report
        6. Log results  - done line
        7. Return mutated data

        Raises InvalidInputError if no item carries both keys.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        data = unwrap_items(data)                     # Step 0: accept a
        self._canonicalize_keys(data)                 # {"results": [...]} envelope
        valid = self._validate(data)                  # 1
        #self._filter(valid)                          # 2 (no-op)
        self._log_validation(len(data), len(valid))   # 3
        self._log_config()

        await self._extraction.run(valid)             # 4: writes {extractor_model}_response_kg
        await self._checking.run(valid)               #    writes verdicts onto each triplet

        self._serialize()                             # 5 (no-op)
        self.last_report = self.build_report(data)    #    the single artifact
        self._log_results(len(data), len(valid))      # 6
        return data                                   # 7

    # -- Validation: drop items dynamically, like every service --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing 'response' or 'reference'.

        Both stages' inputs are checked up front - extraction needs
        'response', checking needs 'reference' - so the checker never has to
        re-drop. Raises InvalidInputError if nothing is valid.
        """
        valid = []
        for i, item in enumerate(data):
            missing = [k for k in ("response", "reference") if k not in item]
            if missing:
                logger.debug("Item %d missing %s - skipping.", i, ", ".join(missing))
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                "No items contain both 'response' and 'reference'."
            )
        return valid

    def _filter(self, valid):
        """No pipeline-level filtering: child services filter their own
        already-processed items."""
        pass

    # -- Serialization: none; children consolidated into data in place --

    def _serialize(self, *args, **kwargs) -> None:
        pass

    # -- Report --

    def build_report(self, data: list[dict]) -> dict:
        """The single output artifact: `_meta` + the checked items.

        Pure projection - loss-free from the in-memory items, no LLM calls,
        safe to rebuild anytime. Unlike ragcheck/faithcheck there are no
        metrics to aggregate: refcheck produces claim-level verdicts, and the
        items already carry them.
        """
        dropped = sum(
            1 for item in data
            if not isinstance(item, dict)
            or any(k not in item for k in ("response", "reference"))
        )
        timestamp, duration = self._run_timing()
        return {
            "_meta": build_meta(
                "refcheck",
                timestamp=timestamp,
                duration_seconds=duration,
                total_items=len(data),
                evaluated_items=len(data) - dropped,
                dropped_items=dropped,
                request_strategies=GLOBAL_STATS.strategies(),
            ),
            "results": data,
        }

    def _run_timing(self) -> tuple[str, float]:
        """(timestamp, elapsed) for the report envelope; safe before a run."""
        if not hasattr(self, "_started_at"):
            return datetime.now().isoformat(timespec="seconds"), 0.0
        return self._started_at, time.perf_counter() - self._started_perf

    # -- Logging --

    def _log_validation(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        dropped = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if dropped:
            logger.info("    ├─ dropped:  %d  (missing response/reference)", dropped)
        logger.info("    └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, *args, **kwargs) -> None:
        pass

    def _log_config(self) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", self._extractor_model)
        logger.info("    Checker:     %s", self._checker_model)
        logger.info("")

    def _log_results(self, total: int, valid: int) -> None:
        """Step 6: post-execution report. No BL for RefChecker; just the done line."""
        self._log_bl_results()
        self._log_done(total, valid)

    def _log_bl_results(self, *args, **kwargs) -> None:
        pass

    def _log_done(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ✅ Done: ref-checked %d/%d items", valid, total)
