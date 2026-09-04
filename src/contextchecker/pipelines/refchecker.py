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
from contextchecker.pipelines.directions import (
    _location,
    log_pipeline_tree,
    unwrap_items,
    verdict_summary,
)
from contextchecker.services.base import BaseService
from contextchecker.services.checking import CheckingService
from contextchecker.services.extraction import ExtractionService
from contextchecker.stats import GLOBAL_STATS, log_token_stats, usage_since
from contextchecker.utils import build_meta, plural

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
        child_verbosity = "silent" if verbosity == "silent" else "compact"
        self.last_report: dict | None = None

        # Compose the two services. Each fail-fasts on its own API key here.
        self._extraction = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            verbosity=child_verbosity,
            section_label="Extraction: response",
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
            verbosity=child_verbosity,
            section_label="Checking: reference",
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
        6. Log results  - results header, done line, token table once
        7. Return mutated data

        Raises InvalidInputError if no item carries both keys.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        self._usage_at_start = GLOBAL_STATS.snapshot()
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
                usage=usage_since(getattr(self, "_usage_at_start", None)),
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
        logger.info("    Total:        %d items", total)
        if dropped:
            logger.info("     ├─ dropped:  %d  (missing response/reference)", dropped)
        logger.info("     └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, *args, **kwargs) -> None:
        pass

    def _log_config(self) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", _location(self._extraction))
        logger.info("    Checker:     %s", _location(self._checking))
        logger.info("    Mode:        %s", self._checking.mode_label)
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(self, total: int, valid: int) -> None:
        """Step 6: consolidated results — header, done line, tokens once."""
        self._log_bl_results()
        self._log_done(total, valid)
        if self.verbosity == "full":
            log_token_stats()

    def _log_bl_results(self, *args, **kwargs) -> None:
        """══ REFCHECK RESULTS ══ rule + 🔀 Pipeline. Refcheck aggregates no
        metric, so the plumbing tree is its whole results block."""
        if self.verbosity != "full":
            return
        logger.info(settings.section_rule("REFCHECK RESULTS", char="═"))
        logger.info("")
        items = self.last_report["results"]
        kg_key = self._extraction.kg_key
        verdict_key = self._checking.verdict_key
        claims = 0
        counts = {"total": 0, "Entailment": 0, "Contradiction": 0,
                  "Neutral": 0, "unknown": 0}
        for item in items:
            for triplet in item.get(kg_key) or []:
                claims += 1
                if verdict_key not in triplet:
                    continue  # skipped or never checked — no verdict issued
                counts["total"] += 1
                verdict = triplet.get(verdict_key)
                counts[verdict if verdict in counts else "unknown"] += 1
        log_pipeline_tree([
            ("📝", "extract response", self._extraction.last_stats,
             f"{claims} {plural(claims, 'claim')}"),
            ("🔎", "check reference", self._checking.last_stats,
             verdict_summary(counts)),
        ])

    def _log_done(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        kg_key = self._extraction.kg_key
        claims = sum(len(item.get(kg_key) or [])
                     for item in self.last_report["results"])
        logger.info(" ✅ Done: %d %s · %d %s", valid, plural(valid, "item"),
                    claims, plural(claims, "claim"))
