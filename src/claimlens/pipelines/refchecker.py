"""
RefChecker - reference-checking use case: extraction + checking in one run.

RefChecker is a *pipeline*: a service whose run() composes other services
(ExtractionService then CheckingService) instead of driving a single worker.
The caller-facing interface is identical, so it subclasses BaseService - there
is no separate pipeline base (see the note in services/base.py).

It talks to services only, never a worker. The two services consolidate their
results into the shared data in place; this class projects them into the
record + findings documents every other report-producing command emits
(architecture.md, "Output documents"), and the controller (cli.py) persists
them from `last_report` / `last_findings`. RefChecker aggregates no metric,
so its `metrics` and `variance` are empty — the skeleton holds, the roster
is simply empty.
"""

import time
from datetime import datetime

from claimlens import settings
from claimlens.exceptions import InvalidInputError
from claimlens.pipelines.directions import (
    _location,
    log_pipeline_tree,
    pipeline_counts,
    unwrap_items,
    verdict_summary,
)
from claimlens.services.base import BaseService
from claimlens.services.checking import CheckingService
from claimlens.services.extraction import ExtractionService
from claimlens.stats import GLOBAL_STATS, log_token_stats, usage_since
from claimlens.utils import build_meta, findings_view, plural

logger = settings.get_logger(__name__)

_VERDICTS = ("Entailment", "Contradiction", "Neutral")


class RefCheckerPipeline(BaseService):
    """Extraction + checking composed into a single reference-checking run."""

    # No aggregate metric: nothing to variance, no headline on the run line.
    _RUN_SUMMARY_KEYS: tuple[str, ...] = ()

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
        # refcheck has no --runs; the base loop runs once and still assembles
        # the record + findings skeleton.
        self._runs = 1
        # Children narrate compactly under the pipeline's labels; the
        # pipeline owns the preamble, the results header, the done line and
        # the one token table — same contract as ragcheck/faithcheck.
        child_verbosity = "silent" if verbosity == "silent" else "compact"
        # The record and the findings, assembled by the base run loop from
        # the per-run entries _run_once leaves on last_run / last_run_findings.
        self.last_report: dict | None = None
        self.last_findings: dict | None = None
        self.last_run: dict | None = None
        self.last_run_findings: dict | None = None

        # Working-document vocabulary (the services' defaults, spelled out
        # here so the projection never reaches into the children).
        self._response_kg = f"{extractor_model}_response_kg"
        self._response_err = f"{extractor_model}_extraction_error"
        namespace = f"{checker_model}_checker"
        self._verdict_key = f"{namespace}_verdict"
        self._explanation_key = f"{namespace}_explanation"
        self._checker_error_key = f"{namespace}_error"

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
        """Run the pipeline once through the base loop, which assembles
        last_report and last_findings."""
        return await self._run_repeated(data)

    async def _run_once(
        self, data: list[dict], announce: bool = True, report: bool = True,
    ) -> list[dict]:
        """Run extraction then checking over *data*, in place; return data.

        1. Validate     - drop items missing 'response' or 'reference'
        2. Filter       - none (pass); child services filter their own
        3. Log pre-exec - validation + config
        4. Execute      - delegate to ExtractionService then CheckingService
        5. Serialize    - none in place (the services consolidated into data
                          already); the run entry lands on last_run
        6. Log results  - results header, pipeline tree, done line, tokens once
        7. Return mutated data

        Raises InvalidInputError if no item carries both keys.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        self._usage_at_start = GLOBAL_STATS.snapshot()
        data = unwrap_items(data)                     # Step 0: accept a
        self._canonicalize_keys(data)                 # {"results": [...]} envelope
        valid = self._validate(data)                  # 1
        self._filter(valid)                           # 2 (no-op)
        if announce:                                  # 3
            self._log_validation(len(data), len(valid))
            self._log_config()

        await self._extraction.run(valid)             # 4: writes {extractor_model}_response_kg
        await self._checking.run(valid)               #    writes verdicts onto each triplet

        self._serialize()                             # 5 (no-op)
        self.last_run = self._build_run(data)         #    this run's entry
        self.last_run_findings = {
            "_meta": dict(self.last_run["_meta"]),
            "findings": self._build_findings(self.last_run["items"]),
        }
        if report:
            self._log_results()                       # 6
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

    # -- Record --

    @staticmethod
    def _is_valid(item) -> bool:
        return isinstance(item, dict) and all(k in item for k in ("response", "reference"))

    def _build_run(self, data: list[dict]) -> dict:
        """One ``runs[]`` entry of the record: {_meta, metrics, counts, items}.

        Pure projection of the mutated items - loss-free, no LLM calls, safe
        to rebuild anytime. ``metrics`` is empty: refcheck produces claim-level
        verdicts and no aggregate. ``counts`` holds the console's numbers
        (🔀 Pipeline, 📝 Extraction, 🔎 Checking).
        """
        items = [self._build_item(item, i) for i, item in enumerate(data)
                 if self._is_valid(item)]
        dropped = len(data) - len(items)
        timestamp, duration = self._run_timing()
        meta = build_meta(
            "refcheck",
            timestamp=timestamp,
            duration_seconds=duration,
            total_items=len(data),
            evaluated_items=len(items),
            dropped_items=dropped,
            request_strategies=GLOBAL_STATS.strategies(),
            usage=usage_since(getattr(self, "_usage_at_start", None)),
        )
        checking = self._verdict_counts(items)
        counts = {
            "pipeline": pipeline_counts(self._pipeline_phases(items)),
            "extraction": {
                "with_claims": sum(1 for it in items if it["claims"]),
                "abstained": sum(1 for it in items
                                 if not it["claims"] and "extraction_error" not in it),
                "failed": sum(1 for it in items if "extraction_error" in it),
            },
            "checking": checking,
        }
        return {"_meta": meta, "metrics": {}, "counts": counts, "items": items}

    def _build_item(self, item: dict, index: int) -> dict:
        """One item of the record: the input fields, the abstention flag and
        every claim with its verdict, explanation and (sparse) error cause."""
        claims = []
        for triplet in item.get(self._response_kg) or []:
            claim = {
                "claim": f"{triplet.get('subject')} {triplet.get('predicate')} {triplet.get('object')}",
                "verdict": triplet.get(self._verdict_key),
                "explanation": triplet.get(self._explanation_key),
            }
            if triplet.get(self._checker_error_key):
                claim["error"] = triplet[self._checker_error_key]
            claims.append(claim)
        entry = {
            "id": str(item.get("id", index)),
            "question": item.get("question", ""),
            "response": item.get("response", ""),
            "reference": item.get("reference"),
            "is_abstention": bool(item.get("is_abstention", False)),
            "claims": claims,
        }
        if self._response_err in item:
            entry["extraction_error"] = item[self._response_err]
        return entry

    @staticmethod
    def _verdict_counts(items: list[dict]) -> dict:
        counts = {v: 0 for v in _VERDICTS}
        counts["unjudged"] = 0
        for item in items:
            for c in item["claims"]:
                counts[c["verdict"] if c["verdict"] in counts else "unjudged"] += 1
        return counts

    def _pipeline_phases(self, items: list[dict]) -> list[tuple]:
        """The two phases, for the 🔀 tree and ``counts.pipeline`` alike:
        (icon, name, PhaseStats, summary text, tallies)."""
        claims = sum(len(it["claims"]) for it in items)
        c = self._verdict_counts(items)
        tally = {"total": sum(c.values()), **{k: c[k] for k in _VERDICTS},
                 "unknown": c["unjudged"]}
        return [
            ("📝", "extract response", self._extraction.last_stats,
             f"{claims} {plural(claims, 'claim')}", {"claims": claims}),
            ("🔎", "check reference", self._checking.last_stats,
             verdict_summary(tally),
             {"verdicts": tally["total"], **{k: c[k] for k in _VERDICTS},
              "unjudged": c["unjudged"]}),
        ]

    @staticmethod
    def _build_findings(items: list[dict]) -> dict:
        """The review queue: the 🔎 Checking branches opened up plus the
        item-level outcomes — ``unsupported`` (Neutral: the reference does
        not cover the claim), ``contradicted``, ``unjudged`` (no verdict, with
        its cause), ``abstained``, ``extraction_failed``. A pure view over the
        record's items."""
        def classify(item: dict):
            head = {"id": item["id"], "question": item["question"]}
            if "extraction_error" in item:
                yield "extraction_failed", {**head, "cause": item["extraction_error"]}
                return
            if item["is_abstention"]:
                yield "abstained", {**head, "response": item["response"]}
                return
            for c in item["claims"]:
                entry = {**head, "claim": c["claim"]}
                if c["verdict"] is None:
                    yield "unjudged", {**entry, "cause": c.get("error", "checker_failure")}
                elif c["verdict"] == "Contradiction":
                    yield "contradicted", {**entry, "explanation": c["explanation"]}
                elif c["verdict"] == "Neutral":
                    yield "unsupported", {**entry, "explanation": c["explanation"]}

        return findings_view(
            ["unsupported", "contradicted", "unjudged", "abstained", "extraction_failed"],
            items, classify)

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

    def _log_results(self) -> None:
        """Step 6: consolidated results — header, pipeline tree, done line,
        tokens once."""
        self._log_bl_results()
        self._log_done()
        if self.verbosity == "full":
            log_token_stats()

    def _log_bl_results(self, *args, **kwargs) -> None:
        """══ REFCHECK RESULTS ══ rule + 🔀 Pipeline. Refcheck aggregates no
        metric, so the plumbing tree is its whole results block."""
        if self.verbosity != "full":
            return
        logger.info(settings.section_rule("REFCHECK RESULTS", char="═"))
        logger.info("")
        log_pipeline_tree(self._pipeline_phases(self.last_run["items"]))

    def _log_done(self, *args, **kwargs) -> None:
        if self.verbosity != "full":
            return
        run = self.last_run
        n = run["_meta"]["evaluated_items"]
        claims = sum(len(it["claims"]) for it in run["items"])
        logger.info(" ✅ Done: %d %s · %d %s", n, plural(n, "item"),
                    claims, plural(claims, "claim"))
