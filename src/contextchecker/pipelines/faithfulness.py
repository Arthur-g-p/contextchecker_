"""
Faithfulness - real-time faithfulness checking WITHOUT ground truth.

The second killer use case: faithfulness is the only RAGChecker metric that
needs no gt_answer (pure retrieved2response), so it works on live production
traffic where no reference answer exists. One extraction + one matrix
direction - the same building blocks as RagChecker, composed smaller:

    response --extract--> {ext}_response_kg
    retrieved2response:   response claims vs each chunk   (matrix)

Two consumption modes:
- CLI `faithcheck`: batch JSON in, single report out (last_report).
- Library facade `check_faithfulness(...)`: one item in-process, returns the
  report entry - the real-time monitoring hook.

Without GT there is no hallucination/self_knowledge split (both need
correctness); an unfaithful claim here just means "not grounded in the
retrieved context", whether it happens to be true or not.
"""

import time
from datetime import datetime

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.models import Direction
from contextchecker.services.base import BaseService
from contextchecker.services.extraction import ExtractionService
from contextchecker.services.checking import CheckingService
from contextchecker.pipelines.directions import (
    abstention_counts,
    normalize_chunks,
    phase_failure_lines,
    run_direction,
    unwrap_items,
)
from contextchecker.pipelines.ragchecker import _ENTAILMENT, _ratio, _row_entailed
from contextchecker.stats import log_mece_tree, log_rate_rows, log_token_stats
from contextchecker.utils import build_meta

logger = settings.get_logger(__name__)

REQUIRED_KEYS = ("response", "retrieved_context")


class FaithfulnessPipeline(BaseService):
    """1 extraction + 1 checking direction: faithfulness without GT."""

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
        runs: int = 1,
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model
        self._init_verbosity(verbosity)
        self._runs = max(1, runs)
        child_verbosity = (
            "silent" if (verbosity == "silent" or self._runs > 1) else "compact"
        )
        self.last_report: dict | None = None

        self._response_kg = f"{extractor_model}_response_kg"
        self._response_err = f"{extractor_model}_extraction_error"
        self._namespace = f"{checker_model}_retrieved2response"

        # Compose the services. Each fail-fasts on its own API key here.
        self._extract = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            verbosity=child_verbosity,
            section_label="Extraction: response",
            dedup=dedup,
        )
        self._check = CheckingService(
            model=checker_model,
            extractor_model=extractor_model,
            base_url=checker_base_url,
            concurrency=concurrency,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            verbosity=child_verbosity,
            section_label="Direction: retrieved2response",
            kg_key=self._response_kg,
            verdict_namespace=self._namespace,
            extraction_error_key=self._response_err,
        )
        self._direction = Direction(
            name="retrieved2response",
            kg_key=self._response_kg,
            per_chunk=True,
        )

    # -- Pipeline: the BaseService 7-step run() shape --

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the pipeline; with runs > 1, repeat it and report variance."""
        if self._runs <= 1:
            return await self._run_once(data)
        return await self._run_repeated(data)

    async def _run_once(
        self, data: list[dict], announce: bool = True, report: bool = True,
    ) -> list[dict]:
        """One full pass over *data*, in place.

        1. Validate     - hard drop: response + retrieved_context required
                          non-empty; chunks normalized to {doc_id, text}
        2. Filter       - none (no skipping)
        3. Log pre-exec - validation + config
        4. Execute      - extraction, then the matrix direction
        5. Serialize    - none in place; report goes to last_report
        6. Log results  - consolidated results block
        7. Return mutated data
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        data = unwrap_items(data)
        self._canonicalize_keys(data)
        valid = self._validate(data)
        self._filter(valid)
        if announce:
            self._log_validation(len(data), len(valid))
            self._log_config()

        # Children print their own labeled section rules (compact mode).
        await self._extract.run(valid)
        await run_direction(self._check, valid, self._direction)

        self._serialize()
        self.last_report = self.build_report(data)
        if report:
            self._log_results()
        return data

    # _run_repeated inherited from BaseService (variance mode)

    # -- Validation --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Hard drop - response and retrieved_context non-empty."""
        valid = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.debug("Item %d is not an object (%s) - skipping.",
                             i, type(item).__name__)
                continue
            missing = [k for k in REQUIRED_KEYS if not item.get(k)]
            if missing:
                logger.debug("Item %d missing/empty %s - skipping.",
                             i, ", ".join(missing))
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                "No items contain non-empty 'response' and 'retrieved_context'."
            )

        for item in valid:
            item["retrieved_context"] = normalize_chunks(item["retrieved_context"])
        return valid

    def _filter(self, valid):
        """No skipping — a failed run is re-run from the start."""
        pass

    # -- Report (the single output artifact) --

    def build_report(self, data: list[dict]) -> dict:
        """Project the mutated items into the faithfulness report."""
        results = []
        dropped = 0
        for item in data:
            if not isinstance(item, dict) or any(
                not item.get(k) for k in REQUIRED_KEYS
            ):
                dropped += 1
                continue
            results.append(self._build_result_entry(item))

        timestamp, duration = self._run_timing()
        return {
            "_meta": build_meta(
                "faithcheck",
                timestamp=timestamp,
                duration_seconds=duration,
                total_items=len(data),
                evaluated_items=len(results),
                dropped_items=dropped,
            ),
            "overall_metrics": self._compute_overall(results),
            "results": results,
        }

    def _run_timing(self) -> tuple[str, float]:
        """(timestamp, elapsed) for the report envelope; safe before a run."""
        if not hasattr(self, "_started_at"):
            return datetime.now().isoformat(timespec="seconds"), 0.0
        return self._started_at, time.perf_counter() - self._started_perf

    def _build_result_entry(self, item: dict) -> dict:
        claims = item.get(self._response_kg) or []
        chunks = item.get("retrieved_context") or []
        doc_ids = [c["doc_id"] for c in chunks]

        matrix = []
        claim_support = []
        for triplet in claims:
            verdicts = triplet.get(f"{self._namespace}_verdicts") or {}
            explanations = triplet.get(f"{self._namespace}_explanations") or {}
            errors = triplet.get(f"{self._namespace}_errors") or {}
            row = []
            for idx in range(len(doc_ids)):
                cell = {
                    "verdict": verdicts.get(idx),
                    "explanation": explanations.get(idx),
                }
                if errors.get(idx):
                    cell["error"] = errors[idx]
                row.append(cell)
            matrix.append(row)
            claim_support.append(
                [d for idx, d in enumerate(doc_ids)
                 if verdicts.get(idx) == _ENTAILMENT]
            )

        entry = {
            "query_id": str(item.get("query_id", item.get("id", ""))),
            "query": item.get("question", ""),
            "response": item.get("response", ""),
            "is_abstention": bool(item.get("is_abstention", False)),
            "retrieved_context": chunks,
            "response_claims": [
                {"subject": t.get("subject"), "predicate": t.get("predicate"),
                 "object": t.get("object")} for t in claims
            ],
            "retrieved2response": matrix,
            # Per-claim attribution: which chunks ground each claim.
            "claim_support": claim_support,
        }
        if self._response_err in item:
            entry["extraction_errors"] = {"response": item[self._response_err]}

        entry["metrics"] = self._item_metrics(entry)
        return entry

    @staticmethod
    def _item_metrics(entry: dict) -> dict:
        """Faithfulness only - no GT means no correctness split.

        Same gating and None-propagation rules as ragcheck: errors and
        abstentions are null, unknown rows leave both sides of the ratio.
        """
        if entry.get("extraction_errors") or entry.get("is_abstention"):
            return {"faithfulness": None}
        statuses = [_row_entailed(row) for row in entry["retrieved2response"]]
        known = [s for s in statuses if s is not None]
        return {"faithfulness": _ratio(sum(known), len(known))}

    @staticmethod
    def _compute_overall(results: list[dict]) -> dict:
        """Macro faithfulness + the reliability rates."""
        values = [e["metrics"]["faithfulness"] for e in results
                  if e["metrics"]["faithfulness"] is not None]
        overall = {
            "faithfulness": round(sum(values) / len(values), 4) if values else None,
            "support": {"faithfulness": len(values)},
        }

        evaluated = len(results)
        if evaluated == 0:
            return overall

        ab = abstention_counts(results)
        # No-results leave the denominator: the abstention rate is over items
        # the model actually got to answer. A tooling failure is charged
        # exactly once, in extraction_error_rate.
        behavioral = evaluated - ab["errored"]
        overall["abstention_rate"] = _ratio(ab["abstained"], behavioral)
        overall["extraction_error_rate"] = _ratio(ab["errored"], evaluated)

        total_cells, none_cells = _verdict_cell_counts(results)
        overall["checker_failure_rate"] = _ratio(none_cells, total_cells)
        return overall

    # -- Serialization: none in place; last_report is the artifact --

    def _serialize(self, *args, **kwargs) -> None:
        pass

    # -- Logging --

    def _log_validation(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        dropped = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if dropped:
            logger.info("    ├─ dropped:  %d  (missing/empty %s)",
                        dropped, "/".join(REQUIRED_KEYS))
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
        logger.info("    Direction:   retrieved2response (no ground truth)")
        logger.info("")

    def _log_results(self) -> None:
        self._log_bl_results()
        self._log_done()
        if self.verbosity == "full":
            log_token_stats()

    def _log_bl_results(self) -> None:
        """Print ── FAITHFULNESS RESULTS ──: pipeline tree + the score."""
        self._log_pipeline_tree()
        self._log_metrics()
        self._log_abstention()
        self._log_reliability()

    def _log_run_findings(self) -> None:
        """Per-run findings in variance mode: metrics + abstention +
        reliability — the pipeline tree (request plumbing) prints at
        --runs 1 but not per run."""
        logger.info("")
        self._log_metrics()
        self._log_abstention()
        self._log_reliability()

    def _log_pipeline_tree(self) -> None:
        """══ RESULTS ══ rule + 🔀 Pipeline: where the requests went."""
        if self.verbosity != "full":
            return
        report = self.last_report
        results = report["results"]

        logger.info("")
        logger.info(settings.section_rule("FAITHFULNESS RESULTS", char="═"))
        logger.info("")

        # ── 🔀 Pipeline
        claims = sum(len(e["response_claims"]) for e in results)
        cells = {"total": 0, "Entailment": 0, "Contradiction": 0,
                 "Neutral": 0, "unknown": 0}
        for e in results:
            for row in e["retrieved2response"]:
                for cell in row:
                    cells["total"] += 1
                    verdict = cell.get("verdict")
                    if verdict in cells:
                        cells[verdict] += 1
                    else:
                        cells["unknown"] += 1
        verdict_summary = (f"{cells['total']} verdicts "
                           f"(🟢 {cells['Entailment']} · 🔴 {cells['Contradiction']}"
                           f" · ⚪ {cells['Neutral']})")
        if cells["unknown"]:
            verdict_summary += f" · ❓ {cells['unknown']}"

        phases = [
            ("📝", "extract response", self._extract.last_stats, f"{claims} claims"),
            ("🔎", "retrieved2response", self._check.last_stats, verdict_summary),
        ]
        total_requests = sum(s.http_requests for _, _, s, _ in phases if s)
        logger.info(" 🔀 Pipeline")
        logger.info("    %d LLM requests across %d phases", total_requests, len(phases))
        for i, (icon, name, stats, summary) in enumerate(phases):
            last = i == len(phases) - 1
            prefix = "└─" if last else "├─"
            requests = stats.http_requests if stats else 0
            logger.info("    %s %s %-21s %2d reqs → %s",
                        prefix, icon, name + ":", requests, summary)
            continuation = "   " if last else "│ "
            failure_lines = phase_failure_lines(stats)
            for j, sub in enumerate(failure_lines):
                sub_prefix = "└─" if j == len(failure_lines) - 1 else "├─"
                logger.info("    %s      %s %s", continuation, sub_prefix, sub)
        logger.info("")

    def _log_metrics(self) -> None:
        """📊 Metrics: the faithfulness score + reliability."""
        if self.verbosity != "full":
            return
        report = self.last_report
        om = report["overall_metrics"]

        n = report["_meta"]["evaluated_items"]
        support = om.get("support", {})
        faithfulness = om.get("faithfulness")
        value = "n/a" if faithfulness is None else f"{faithfulness:.3f}"
        note = "response claims supported by the retrieved context"
        if support.get("faithfulness") is not None and support["faithfulness"] != n:
            note = f"{support['faithfulness']} of {n} items · {note}"
        logger.info(" 📊 Metrics  (macro over %d items)", n)
        logger.info("    └─ faithfulness:  %s  (%s)", value, note)
        logger.info("")

    def _log_abstention(self) -> None:
        """⚪ Abstention Behavior tree — no ground truth, so abstentions
        cannot be judged justified vs not; the tree says so explicitly.

        Extraction failures branch in only when present — they sit inside
        the rate denominator (docs/holy_data.md rule set 1)."""
        if self.verbosity != "full":
            return
        ab = abstention_counts(self.last_report["results"])
        # Behavior only: extraction-failed items are out of the tree AND
        # out of the rate denominator (charged once, in 💥 Reliability).
        top = ab["evaluated"] - ab["errored"]
        note = None
        if ab["errored"]:
            note = (f"{ab['errored']} extraction-failed excluded"
                    " — see 💥 Reliability")
        log_mece_tree(
            "⚪ Abstention Behavior", top, "evaluated items",
            [
                ("🔬", ab["answered"], "answered",
                 "claims extracted and scored"),
                ("⚪", ab["abstained"], "abstained",
                 "no ground truth — cannot judge justified vs not"),
            ],
            footer=[("abstained", ab["abstained"], top)],
            header_note=note,
        )
        logger.info("")

    def _log_reliability(self) -> None:
        """💥 Reliability rate rows — harness health, always printed
        (docs/holy_data.md rule set 2: hidden ≠ zero)."""
        if self.verbosity != "full":
            return
        report = self.last_report
        ab = abstention_counts(report["results"])
        total_cells, none_cells = _verdict_cell_counts(report["results"])
        causes = {"response": ab["errored"]} if ab["errored"] else None
        log_rate_rows(
            "💥 Reliability",
            [("📝", "Extraction", ab["errored"], ab["evaluated"],
              "items failed", "extraction_error_rate", causes),
             ("🔎", "Checking", none_cells, total_cells,
              "verdicts missing", "checker_failure_rate", None)],
            header_note="tooling — excluded from all metrics,"
                        " counted once here",
        )
        logger.info("")

    def _log_done(self) -> None:
        if self.verbosity != "full":
            return
        report = self.last_report
        logger.info(" ✅ Done: faithfulness-checked %d/%d items",
                    report["_meta"]["evaluated_items"],
                    report["_meta"]["total_items"])


# ── Library facade (real-time, single item) ──────────────────────────────────

def _verdict_cell_counts(results: list[dict]) -> tuple[int, int]:
    """(total, none) verdict cells across the retrieved2response matrix,
    skipping extraction-errored items — shared by compute and display."""
    total = none = 0
    for e in results:
        if e.get("extraction_errors"):
            continue
        for row in e["retrieved2response"]:
            for cell in row:
                total += 1
                none += cell.get("verdict") is None
    return total, none


def check_faithfulness(
    response: str,
    retrieved_context: list,
    *,
    extractor_model: str,
    checker_model: str,
    **pipeline_kwargs,
) -> dict:
    """Score one response against its retrieved context, in process.

    The real-time entry point: no CLI, no files. Returns the report entry
    for the single item — faithfulness score, per-claim chunk attribution
    (claim_support), the full verdict matrix, and abstention/error flags.

    Extra keyword arguments are forwarded to FaithfulnessPipeline
    (base URLs, joint config, retries, ...).
    """
    pipeline = FaithfulnessPipeline(
        extractor_model=extractor_model,
        checker_model=checker_model,
        verbosity="silent",
        **pipeline_kwargs,
    )
    pipeline.run_sync([{"response": response, "retrieved_context": retrieved_context}])
    return pipeline.last_report["results"][0]
