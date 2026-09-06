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
    _location,
    abstention_counts,
    log_pipeline_tree,
    pipeline_counts,
    verdict_summary,
    normalize_chunks,
    run_direction,
    unwrap_items,
)
from contextchecker.pipelines.ragchecker import _ENTAILMENT, _ratio, _row_entailed
from contextchecker.stats import GLOBAL_STATS, format_headline, log_mece_tree, log_rate_rows, log_token_stats, usage_since
from contextchecker.utils import build_meta, findings_view, plural

logger = settings.get_logger(__name__)

REQUIRED_KEYS = ("response", "retrieved_context")


class FaithfulnessPipeline(BaseService):
    """1 extraction + 1 checking direction: faithfulness without GT."""

    # Behavior carries only the abstention rate — the justified/
    # unjustified split is unknowable without GT.
    _METRIC_DIRECTIONS = {"faithfulness": "higher is better",
                          "abstention_rate": "distribution — no direction"}
    _VARIANCE_SECTIONS = {
        "metrics": [(None, ["faithfulness"])],
        "behavior": ["abstention_rate"],
        "health": ["extraction_error_rate", "checker_failure_rate"],
    }

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
        # The record and the findings, assembled by the base run loop from
        # the per-run entries _run_once leaves on last_run / last_run_findings.
        self.last_report: dict | None = None
        self.last_findings: dict | None = None
        self.last_run: dict | None = None
        self.last_run_findings: dict | None = None

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
        """Run the pipeline N times (N = --runs, default 1); the base loop
        assembles last_report and last_findings."""
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
        5. Serialize    - none in place; the run entry goes to last_run
        6. Log results  - consolidated results block
        7. Return mutated data
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        self._usage_at_start = GLOBAL_STATS.snapshot()
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
        self.last_run = self._build_run(data)
        self.last_run_findings = {
            "_meta": dict(self.last_run["_meta"]),
            "findings": self._build_findings(self.last_run["items"]),
        }
        if report:
            self._log_results()
        return data

    # _run_repeated inherited from BaseService (variance mode)

    # -- Validation --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Hard drop - response present (a string; "" is a full
        abstention, not missing data) and retrieved_context non-empty."""
        valid = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.debug("Item %d is not an object (%s) - skipping.",
                             i, type(item).__name__)
                continue
            missing = []
            if not isinstance(item.get("response"), str):
                missing.append("response")
            if not item.get("retrieved_context"):
                missing.append("retrieved_context")
            if missing:
                logger.debug("Item %d missing %s - skipping.",
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

    def _build_run(self, data: list[dict]) -> dict:
        """One ``runs[]`` entry of the record: {_meta, metrics, counts, items}.

        ``metrics`` are the varianced scalars (the console's Metrics roster),
        ``counts`` every number the console blocks print, ``items`` the
        complete per-item record. Pure projection of the mutated items —
        loss-free, no LLM calls, safe to rebuild anytime.
        """
        items = []
        dropped = 0
        for item in data:
            if not isinstance(item, dict) or any(
                not item.get(k) for k in REQUIRED_KEYS
            ):
                dropped += 1
                continue
            items.append(self._build_result_entry(item))

        timestamp, duration = self._run_timing()
        meta = build_meta(
            "faithcheck",
            timestamp=timestamp,
            duration_seconds=duration,
            total_items=len(data),
            evaluated_items=len(items),
            dropped_items=dropped,
            request_strategies=GLOBAL_STATS.strategies(),
            usage=usage_since(getattr(self, "_usage_at_start", None)),
        )
        metrics, support = self._compute_metrics(items)
        counts = {
            # 📊 Metrics brackets
            "support": support,
            # 🔀 Pipeline
            "pipeline": pipeline_counts(self._pipeline_phases(items)),
            # ⚪ Abstention Behavior
            "abstention": abstention_counts(items),
            # 💥 Reliability
            "reliability": self._reliability_counts(items),
        }
        return {"_meta": meta, "metrics": metrics, "counts": counts, "items": items}

    def _pipeline_phases(self, items: list[dict]) -> list[tuple]:
        """The two phases, for the 🔀 tree and ``counts.pipeline`` alike:
        (icon, name, PhaseStats, summary text, tallies)."""
        claims = sum(len(e["response_claims"]) for e in items)
        cells = {"total": 0, "Entailment": 0, "Contradiction": 0,
                 "Neutral": 0, "unknown": 0}
        for e in items:
            for row in e["retrieved2response"]:
                for cell in row:
                    cells["total"] += 1
                    verdict = cell.get("verdict")
                    cells[verdict if verdict in cells else "unknown"] += 1
        tally = {"verdicts": cells["total"], "Entailment": cells["Entailment"],
                 "Contradiction": cells["Contradiction"], "Neutral": cells["Neutral"],
                 "unjudged": cells["unknown"]}
        return [
            ("📝", "extract response", self._extract.last_stats,
             f"{claims} {plural(claims, 'claim')}", {"claims": claims}),
            ("🔎", "retrieved2response", self._check.last_stats,
             verdict_summary(cells), tally),
        ]

    @staticmethod
    def _reliability_counts(items: list[dict]) -> dict:
        """The numbers behind the two 💥 rows."""
        ab = abstention_counts(items)
        total_cells, none_cells = _verdict_cell_counts(items)
        return {
            "extraction": {"failed": ab["errored"], "items": ab["evaluated"],
                           "by_cause": {"response": ab["errored"]}},
            "checking": {"unjudged": none_cells, "issued": total_cells},
        }

    @staticmethod
    def _build_findings(items: list[dict]) -> dict:
        """The review queue: the claim outcomes behind faithfulness plus the
        ⚪ and 💥 branches, one list each — ``ungrounded`` (no chunk entails
        the claim), ``contradicted`` (a chunk contradicts it; the strongest
        signal, named with its explanation), ``undecidable`` (no chunk
        entails it and at least one verdict is missing, so the score
        excludes it), ``abstained``, ``extraction_failed``. A pure view over
        the record's items."""
        def classify(item: dict):
            head = {"query_id": item["query_id"], "query": item["query"]}
            if item.get("extraction_errors"):
                yield "extraction_failed", {**head, "cause": item["extraction_errors"]["response"]}
                return
            if item.get("is_abstention"):
                yield "abstained", {**head, "response": item["response"]}
                return
            doc_ids = [c["doc_id"] for c in item["retrieved_context"]]
            for claim, row in zip(item["response_claims"], item["retrieved2response"]):
                verdicts = [c.get("verdict") for c in row]
                if _ENTAILMENT in verdicts:
                    continue  # grounded — not a finding
                text = f"{claim['subject']} {claim['predicate']} {claim['object']}"
                contradictions = [(doc_ids[d], c.get("explanation"))
                                  for d, c in enumerate(row) if c.get("verdict") == "Contradiction"]
                tally = {v: verdicts.count(v) for v in ("Neutral", "Contradiction") if verdicts.count(v)}
                if contradictions:
                    for doc_id, explanation in contradictions:
                        yield "contradicted", {**head, "claim": text, "doc_id": doc_id,
                                               "explanation": explanation, "verdicts": tally}
                elif None in verdicts:
                    yield "undecidable", {**head, "claim": text, "verdicts": tally,
                                          "unjudged": {doc_ids[d]: c.get("error", "checker_failure")
                                                       for d, c in enumerate(row) if c.get("verdict") is None}}
                else:
                    yield "ungrounded", {**head, "claim": text, "chunks_checked": len(row)}

        return findings_view(
            ["ungrounded", "contradicted", "undecidable", "abstained", "extraction_failed"],
            items, classify)

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
    @staticmethod
    def _compute_metrics(items: list[dict]) -> tuple[dict, dict]:
        """(metrics, support): macro faithfulness, the abstention rate and
        the two reliability rates — exactly the variance roster."""
        values = [e["metrics"]["faithfulness"] for e in items
                  if e["metrics"]["faithfulness"] is not None]
        metrics = {
            "faithfulness": round(sum(values) / len(values), 4) if values else None,
            "abstention_rate": None,
            "extraction_error_rate": None,
            "checker_failure_rate": None,
        }
        support = {"faithfulness": len(values)}
        evaluated = len(items)
        if evaluated == 0:
            return metrics, support

        ab = abstention_counts(items)
        # No-results leave the denominator: the abstention rate is over items
        # the model actually got to answer. A tooling failure is charged
        # exactly once, in extraction_error_rate.
        metrics["abstention_rate"] = _ratio(ab["abstained"], evaluated - ab["errored"])
        metrics["extraction_error_rate"] = _ratio(ab["errored"], evaluated)
        total_cells, none_cells = _verdict_cell_counts(items)
        metrics["checker_failure_rate"] = _ratio(none_cells, total_cells)
        return metrics, support

    # -- Serialization: none in place; last_run is the artifact --

    def _serialize(self, *args, **kwargs) -> None:
        pass

    # -- Logging --

    def _log_validation(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        dropped = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:        %d items", total)
        if dropped:
            logger.info("     ├─ dropped:  %d  (missing/empty %s)",
                        dropped, "/".join(REQUIRED_KEYS))
        logger.info("     └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, *args, **kwargs) -> None:
        pass

    def _log_config(self) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", _location(self._extract))
        logger.info("    Checker:     %s", _location(self._check))
        logger.info("    Mode:        %s", self._check.mode_label)
        logger.info("    Direction:   retrieved2response (no ground truth)")
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
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
        logger.info(settings.section_rule("FAITHFULNESS RESULTS", char="═"))
        logger.info("")
        log_pipeline_tree(self._pipeline_phases(self.last_run["items"]))

    def _log_metrics(self) -> None:
        """📊 Metrics: the faithfulness score + reliability."""
        if self.verbosity != "full":
            return
        run = self.last_run
        om = run["metrics"]

        n = run["_meta"]["evaluated_items"]
        support = run["counts"]["support"]
        faithfulness = om.get("faithfulness")
        value = "n/a" if faithfulness is None else f"{faithfulness:.3f}"
        note = "response claims supported by the retrieved context"
        if support.get("faithfulness") is not None and support["faithfulness"] != n:
            note = f"{support['faithfulness']} of {n} items · {note}"
        if faithfulness is not None:
            note += f" · {self._METRIC_DIRECTIONS['faithfulness']}"
        logger.info(" 📊 Metrics  (macro over %d items)", n)
        logger.info("     └─ faithfulness:  %s  (%s)", value, note)
        logger.info("")

    def _log_abstention(self) -> None:
        """⚪ Abstention Behavior tree — no ground truth, so abstentions
        cannot be judged justified vs not; the tree says so explicitly.

        Extraction failures branch in only when present — they sit
        inside the rate denominator."""
        if self.verbosity != "full":
            return
        ab = self.last_run["counts"]["abstention"]
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
                 "claims extracted — scored"),
                ("⚪", ab["abstained"], "abstained",
                 "no ground truth — cannot judge justified vs not"),
            ],
            footer=[("abstention rate", ab["abstained"], top)],
            header_note=note,
        )
        logger.info("")

    def _log_reliability(self) -> None:
        """💥 Reliability rate rows — harness health, always printed."""
        if self.verbosity != "full":
            return
        rel = self.last_run["counts"]["reliability"]
        ext, chk = rel["extraction"], rel["checking"]
        causes = {k: v for k, v in ext["by_cause"].items() if v} or None
        log_rate_rows(
            "💥 Reliability",
            [("📝", "Extraction", ext["failed"], ext["items"],
              "items failed", "extraction_error_rate", causes),
             ("🔎", "Checking", chk["unjudged"], chk["issued"],
              "verdicts unjudged", "checker_failure_rate", None)],
            header_note="tooling — excluded from all metrics,"
                        " counted once here",
        )
        logger.info("")

    def _log_done(self) -> None:
        if self.verbosity != "full":
            return
        run = self.last_run
        n = run["_meta"]["evaluated_items"]
        logger.info(" ✅ Done: %d %s · %s", n, plural(n, "item"),
                    format_headline(run["metrics"], self._RUN_SUMMARY_KEYS))


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
    return pipeline.last_report["runs"][0]["items"][0]
