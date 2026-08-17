"""
RagChecker - RAGChecker-style RAG evaluation: 2 extractions + 4 checking
directions + a single self-contained report.

Like RefChecker this is a *pipeline*: a BaseService whose run() composes
services (2x ExtractionService, 4x CheckingService via the direction
runner) instead of driving a worker. It talks to services only.

Data flow:
    response  --extract-->  {ext}_response_kg   ┐
    gt_answer --extract-->  {ext}_gt_answer_kg  ┤ in-place on the items
                                                │
    answer2response:    response claims vs gt_answer        (flat)
    response2answer:    gt claims       vs response         (flat)
    retrieved2response: response claims vs each chunk       (matrix)
    retrieved2answer:   gt claims       vs each chunk       (matrix)

The mutated items are the in-memory working document; the only artifact
the CLI writes is build_report() - RAGChecker's structure (results list +
four directional arrays) with modernized leaves (dict claims, verdict
objects). Metrics are stubbed until the formulas are settled.
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
    normalize_chunks,
    phase_failure_lines,
    run_direction,
    unwrap_items,
)
from contextchecker.stats import log_token_stats
from contextchecker.utils import build_meta

logger = settings.get_logger(__name__)

# Hard drop: every metric family needs all three (only precision/recall
# would survive without chunks - those items belong in the faithfulness
# pipeline, not in a degraded ragcheck).
REQUIRED_KEYS = ("response", "gt_answer", "retrieved_context")

_ENTAILMENT = "Entailment"

# Paper metric names, in the original RAGChecker output order.
METRIC_NAMES = (
    "precision", "recall", "claim_recall", "context_precision",
    "faithfulness", "noise_sensitivity_in_relevant",
    "noise_sensitivity_in_irrelevant", "f1", "hallucination",
    "self_knowledge", "context_utilization",
)


# ── Metrics (pure functions - report entries in, numbers out) ────────────────
#
# Formulas follow RAGChecker (Ru et al., 2024); verified against the original
# implementation's reference output (see the reference test in
# tests/unit/test_ragchecker.py).
#
# None-verdict propagation: a failed check is UNKNOWN, never "not entailed" -
# unknown claims leave numerator AND denominator so checker failures cannot
# inflate hallucination. Known cells decide when they can: one Entailment in a
# matrix row makes the claim entailed regardless of unknown cells.


def _ratio(numerator: float, denominator: int) -> float | None:
    """Zero denominators are null, not 0.0 - 'not computable' is not a score."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _row_entailed(row: list[dict]) -> bool | None:
    """Three-valued 'entailed by any chunk' over a matrix row."""
    verdicts = [cell.get("verdict") for cell in row]
    if _ENTAILMENT in verdicts:
        return True
    if None in verdicts:
        return None
    return False


def _chunk_relevance(ret2a: list[list[dict]], n_chunks: int) -> list[bool | None]:
    """Per chunk: does it entail at least one gt claim? (three-valued)."""
    relevance: list[bool | None] = []
    for k in range(n_chunks):
        column = [row[k].get("verdict") for row in ret2a]
        if _ENTAILMENT in column:
            relevance.append(True)
        elif None in column:
            relevance.append(None)
        else:
            relevance.append(False)
    return relevance


def compute_item_metrics(entry: dict) -> dict:
    """All 11 RAGChecker metrics for one report entry; None = not computable.

    Gating per project rulings:
    - extraction error (either side): item fully excluded - every metric None.
    - abstention: generator family None; retrieval metrics still computed
      (retrieved2answer does not involve the response).
    """
    metrics: dict = dict.fromkeys(METRIC_NAMES)

    if entry.get("extraction_errors"):
        return metrics

    ret2a = entry["retrieved2answer"]
    n_chunks = len(entry["retrieved_context"])

    # -- Retriever metrics (computed even for abstentions) --
    gt_chunk_status = [_row_entailed(row) for row in ret2a]
    known_rows = [s for s in gt_chunk_status if s is not None]
    metrics["claim_recall"] = _ratio(sum(known_rows), len(known_rows))

    relevance = _chunk_relevance(ret2a, n_chunks)
    known_chunks = [r for r in relevance if r is not None]
    metrics["context_precision"] = _ratio(sum(known_chunks), len(known_chunks))

    if entry.get("is_abstention"):
        return metrics

    # -- Generator metrics --
    a2r = [cell.get("verdict") for cell in entry["answer2response"]]
    r2a = [cell.get("verdict") for cell in entry["response2answer"]]
    resp_chunk_status = [_row_entailed(row) for row in entry["retrieved2response"]]

    known_a2r = [v for v in a2r if v is not None]
    metrics["precision"] = _ratio(
        sum(v == _ENTAILMENT for v in known_a2r), len(known_a2r))
    known_r2a = [v for v in r2a if v is not None]
    metrics["recall"] = _ratio(
        sum(v == _ENTAILMENT for v in known_r2a), len(known_r2a))

    p, r = metrics["precision"], metrics["recall"]
    if p is not None and r is not None:
        metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    # Faithfulness family shares ONE decidable set (chunk status AND
    # correctness known) so faithfulness = 1 - hallucination - self_knowledge
    # holds exactly.
    family = [
        (chunk_entailed, correct == _ENTAILMENT)
        for chunk_entailed, correct in zip(resp_chunk_status, a2r)
        if chunk_entailed is not None and correct is not None
    ]
    n_family = len(family)
    metrics["faithfulness"] = _ratio(
        sum(entailed for entailed, _ in family), n_family)
    metrics["hallucination"] = _ratio(
        sum((not entailed) and (not correct) for entailed, correct in family),
        n_family)
    metrics["self_knowledge"] = _ratio(
        sum((not entailed) and correct for entailed, correct in family),
        n_family)

    # context_utilization (paper definition, gt-side): of the gt claims
    # answerable from the retrieved chunks, how many made it into the response.
    cu_pairs = [
        (chunk_entailed, v == _ENTAILMENT)
        for chunk_entailed, v in zip(gt_chunk_status, r2a)
        if chunk_entailed is not None and v is not None
    ]
    metrics["context_utilization"] = _ratio(
        sum(entailed and in_response for entailed, in_response in cu_pairs),
        sum(entailed for entailed, _ in cu_pairs))

    # Noise sensitivities: incorrect response claims copied from relevant /
    # only-irrelevant chunks. A claim whose classification hinges on an
    # unknown cell is excluded entirely.
    ns_rel = ns_irrel = ns_known = 0
    for i, correctness in enumerate(a2r):
        if correctness is None:
            continue
        if correctness == _ENTAILMENT:
            ns_known += 1  # correct claims contribute 0 to both numerators
            continue
        entailed_by_relevant = entailed_by_irrelevant = undecidable = False
        for k, cell in enumerate(entry["retrieved2response"][i]):
            verdict = cell.get("verdict")
            if verdict == _ENTAILMENT:
                if relevance[k] is True:
                    entailed_by_relevant = True
                elif relevance[k] is False:
                    entailed_by_irrelevant = True
                else:
                    undecidable = True
            elif verdict is None:
                undecidable = True
        if undecidable and not entailed_by_relevant:
            continue
        ns_known += 1
        if entailed_by_relevant:
            ns_rel += 1
        elif entailed_by_irrelevant:
            ns_irrel += 1
    metrics["noise_sensitivity_in_relevant"] = _ratio(ns_rel, ns_known)
    metrics["noise_sensitivity_in_irrelevant"] = _ratio(ns_irrel, ns_known)

    return metrics


def compute_overall_metrics(results: list[dict]) -> dict:
    """Macro aggregation (per paper): average per-item values, skipping nulls.

    'support' reports how many items actually contributed to each average -
    exclusions shrink N invisibly otherwise. Adds the project's own rates:
    abstention (with the justified/unjustified split from retrieval evidence),
    extraction errors, and checker failures (None verdicts across all arrays).
    """
    overall: dict = {}
    support: dict = {}
    for name in METRIC_NAMES:
        values = [e["metrics"][name] for e in results
                  if e["metrics"].get(name) is not None]
        support[name] = len(values)
        overall[name] = round(sum(values) / len(values), 4) if values else None
    overall["support"] = support

    evaluated = len(results)
    if evaluated == 0:
        return overall

    errored = [e for e in results if e.get("extraction_errors")]
    abstained = [e for e in results
                 if e.get("is_abstention") and not e.get("extraction_errors")]

    # Justified = retrieval gave the generator nothing (no chunk entails any
    # gt claim); unjustified = evidence was retrieved, the generator refused
    # anyway. Unknown claim_recall leaves an abstention uncategorized.
    justified = sum(1 for e in abstained
                    if e["metrics"].get("claim_recall") == 0.0)
    unjustified = sum(1 for e in abstained
                      if (cr := e["metrics"].get("claim_recall")) is not None
                      and cr > 0)
    overall["abstention_rate"] = _ratio(len(abstained), evaluated)
    overall["justified_abstention_rate"] = _ratio(justified, evaluated)
    overall["unjustified_abstention_rate"] = _ratio(unjustified, evaluated)

    overall["extraction_error_rate"] = _ratio(len(errored), evaluated)
    overall["extraction_errors"] = {
        "response": sum(1 for e in errored
                        if e["extraction_errors"].get("response")),
        "gt_answer": sum(1 for e in errored
                         if e["extraction_errors"].get("gt_answer")),
    }

    # Judge reliability: share of None verdicts across every issued check.
    total_cells = none_cells = 0
    for e in results:
        if e.get("extraction_errors"):
            continue
        for key in ("answer2response", "response2answer"):
            for cell in e[key]:
                total_cells += 1
                none_cells += cell.get("verdict") is None
        for key in ("retrieved2response", "retrieved2answer"):
            for row in e[key]:
                for cell in row:
                    total_cells += 1
                    none_cells += cell.get("verdict") is None
    overall["checker_failure_rate"] = _ratio(none_cells, total_cells)

    return overall


class RagCheckerPipeline(BaseService):
    """2 extractions + 4 checking directions composed into one run."""

    def __init__(
        self,
        extractor_model: str,
        checker_model: str,
        *,
        extractor_base_url: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        extractor_max_retries: int | None = 2,
        dedup: bool = True,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        checker_max_retries: int | None = None,
        verbosity: str = "full",
        runs: int = 1,
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model
        self._init_verbosity(verbosity)
        # runs > 1 = variance mode: the pipeline repeats the whole experiment
        # itself (the controller only passes the number through).
        self._runs = max(1, runs)
        # Children narrate compactly under the pipeline's labels; silent when
        # the pipeline is silent (library use) OR in variance mode, where all
        # runs print symmetrically as one summary line each.
        child_verbosity = (
            "silent" if (verbosity == "silent" or self._runs > 1) else "compact"
        )
        # The single output artifact, populated by run() — the CLI reads this
        # instead of calling the pipeline a second time (atomizer precedent).
        self.last_report: dict | None = None

        # Working-document vocabulary. GT keys are separate so a failed GT
        # extraction can never masquerade as a response failure (or vice versa).
        self._response_kg = f"{extractor_model}_response_kg"
        self._response_err = f"{extractor_model}_extraction_error"
        self._gt_kg = f"{extractor_model}_gt_answer_kg"
        self._gt_err = f"{extractor_model}_gt_extraction_error"

        # Compose the services. Each fail-fasts on its own API key here.
        extractor_config = dict(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            max_retries=extractor_max_retries,
            verbosity=child_verbosity,
            dedup=dedup,
        )
        self._extract_response = ExtractionService(
            **extractor_config, section_label="Extraction: response",
        )
        # mark_abstention=False: an empty GT extraction is a data-quality
        # signal, not a response abstention.
        self._extract_gt = ExtractionService(
            **extractor_config,
            section_label="Extraction: gt_answer",
            source_key="gt_answer",
            kg_key=self._gt_kg,
            error_key=self._gt_err,
            mark_abstention=False,
        )

        # The four directions, each with its own namespaced CheckingService
        # so verdicts over the same triplets never collide.
        self._directions: list[tuple[Direction, CheckingService]] = []

        def _add_direction(name: str, kg_key: str, error_key: str,
                           *, reference_key: str | None = None,
                           per_chunk: bool = False) -> None:
            service = CheckingService(
                model=checker_model,
                extractor_model=extractor_model,
                base_url=checker_base_url,
                concurrency=concurrency,
                joint=joint,
                joint_num=joint_num,
                max_words=max_words,
                max_retries=checker_max_retries,
                verbosity=child_verbosity,
                section_label=f"Direction: {name}",
                kg_key=kg_key,
                verdict_namespace=f"{checker_model}_{name}",
                extraction_error_key=error_key,
            )
            self._directions.append((
                Direction(name=name, kg_key=kg_key,
                          reference_key=reference_key, per_chunk=per_chunk),
                service,
            ))

        _add_direction("answer2response", self._response_kg, self._response_err,
                       reference_key="gt_answer")
        _add_direction("response2answer", self._gt_kg, self._gt_err,
                       reference_key="response")
        _add_direction("retrieved2response", self._response_kg, self._response_err,
                       per_chunk=True)
        _add_direction("retrieved2answer", self._gt_kg, self._gt_err,
                       per_chunk=True)

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

        1. Validate     - hard drop: response, gt_answer, retrieved_context
                          all required non-empty; chunks normalized to
                          {doc_id, text} dicts
        2. Filter       - none (v1: no skipping; crash cache covers re-runs)
        3. Log pre-exec - validation + config (announce=False in variance
                          mode after run 1 - they are run-invariant)
        4. Execute      - 2 extractions, then the 4 directions
        5. Serialize    - none in-place (services + runner already did);
                          the report lands on last_report
        6. Log results  - consolidated results (report=False in variance
                          mode - the VARIANCE block reports instead)
        7. Return mutated data
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        data = unwrap_items(data)                     # Step 0: accept the
        self._canonicalize_keys(data)                 # paper's {"results": [...]}
        valid = self._validate(data)                  # envelope; query→question

        self._filter(valid)                           # 2 (no-op)
        if announce:                                  # 3
            self._log_validation(len(data), len(valid))
            self._log_config()

        # Children print their own labeled section rules (compact mode).
        await self._extract_response.run(valid)       # 4
        await self._extract_gt.run(valid)

        for direction, service in self._directions:
            await run_direction(service, valid, direction)

        self._serialize()                             # 5 (no-op)
        self.last_report = self.build_report(data)    # 6: the single artifact
        if report:
            self._log_results()                       # 7: consolidated results
        return data

    # _run_repeated inherited from BaseService (variance mode)

    # -- Validation --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Hard drop - all three required keys must be non-empty.

        Falsy counts as missing: an empty gt_answer or an empty chunk list
        produces meaningless metrics, so those items are dropped too.
        Surviving items get retrieved_context normalized in place to
        [{doc_id, text}] dicts (bare strings get synthesized ids).
        """
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
                "No items contain non-empty 'response', 'gt_answer' "
                "and 'retrieved_context'."
            )

        for item in valid:
            item["retrieved_context"] = normalize_chunks(item["retrieved_context"])
        return valid

    def _filter(self, valid):
        """No skipping in v1 (2.0 feature); LLM crash cache covers re-runs."""
        pass

    # -- Report (the single output artifact) --

    def build_report(self, data: list[dict]) -> dict:
        """Project the mutated working data into the report document.

        RAGChecker structure (results list + four directional arrays,
        parallel to the claims) with modernized leaves: claims as dicts,
        verdict entries as objects. Pure projection - loss-free from the
        in-memory items, no LLM calls, safe to rebuild anytime.
        """
        results = []
        dropped = 0
        for item in data:
            if any(not item.get(k) for k in REQUIRED_KEYS):
                dropped += 1
                continue
            results.append(self._build_result_entry(item))

        timestamp, duration = self._run_timing()
        return {
            "_meta": build_meta(
                "ragcheck",
                timestamp=timestamp,
                duration_seconds=duration,
                total_items=len(data),
                evaluated_items=len(results),
                dropped_items=dropped,
            ),
            "overall_metrics": compute_overall_metrics(results),
            "results": results,
        }

    def _run_timing(self) -> tuple[str, float]:
        """(timestamp, elapsed) for the report envelope; safe before a run."""
        if not hasattr(self, "_started_at"):
            return datetime.now().isoformat(timespec="seconds"), 0.0
        return self._started_at, time.perf_counter() - self._started_perf

    def _build_result_entry(self, item: dict) -> dict:
        response_claims = item.get(self._response_kg) or []
        gt_claims = item.get(self._gt_kg) or []
        chunks = item.get("retrieved_context") or []
        doc_ids = [c["doc_id"] for c in chunks]

        a2r = f"{self._checker_model}_answer2response"
        r2a = f"{self._checker_model}_response2answer"
        ret2r = f"{self._checker_model}_retrieved2response"
        ret2a = f"{self._checker_model}_retrieved2answer"

        entry = {
            "query_id": str(item.get("query_id", item.get("id", ""))),
            "query": item.get("question", ""),
            "gt_answer": item.get("gt_answer", ""),
            "response": item.get("response", ""),
            # Explicit in the report (a view for humans/frontend), sparse in
            # the working data.
            "is_abstention": bool(item.get("is_abstention", False)),
            "retrieved_context": chunks,
            "response_claims": [self._spo(t) for t in response_claims],
            "gt_answer_claims": [self._spo(t) for t in gt_claims],
            "answer2response": [self._flat_cell(t, a2r) for t in response_claims],
            "response2answer": [self._flat_cell(t, r2a) for t in gt_claims],
            "retrieved2response": [
                self._matrix_row(t, ret2r, len(chunks)) for t in response_claims
            ],
            "retrieved2answer": [
                self._matrix_row(t, ret2a, len(chunks)) for t in gt_claims
            ],
        }

        # Tooling failures surface explicitly - never mistakable for abstention.
        errors = {}
        if self._response_err in item:
            errors["response"] = item[self._response_err]
        if self._gt_err in item:
            errors["gt_answer"] = item[self._gt_err]
        if errors:
            entry["extraction_errors"] = errors

        # Load-bearing intermediate, exposed so the frontend never re-derives
        # it from the matrix (drives context_utilization + noise sensitivity).
        relevance = _chunk_relevance(entry["retrieved2answer"], len(doc_ids))
        entry["relevant_chunks"] = [
            doc_id for doc_id, relevant in zip(doc_ids, relevance)
            if relevant is True
        ]

        # Metrics last - gating reads is_abstention and extraction_errors.
        entry["metrics"] = compute_item_metrics(entry)
        return entry

    @staticmethod
    def _spo(triplet: dict) -> dict:
        """Claims in the report are clean s/p/o - verdicts live in the arrays."""
        return {
            "subject": triplet.get("subject"),
            "predicate": triplet.get("predicate"),
            "object": triplet.get("object"),
        }

    @staticmethod
    def _flat_cell(triplet: dict, namespace: str) -> dict:
        """A null verdict is never opaque in the report: the check-failure
        cause rides along, sparsely."""
        cell = {
            "verdict": triplet.get(f"{namespace}_verdict"),
            "explanation": triplet.get(f"{namespace}_explanation"),
        }
        error = triplet.get(f"{namespace}_error")
        if error:
            cell["error"] = error
        return cell

    @staticmethod
    def _matrix_row(triplet: dict, namespace: str, n_chunks: int) -> list[dict]:
        """One row per claim, one cell per chunk, in retrieved_context order.

        Matrix cells match the flat-cell shape: verdict + explanation, plus
        the failure cause when a check errored — sparse "error" key.
        """
        verdicts = triplet.get(f"{namespace}_verdicts") or {}
        explanations = triplet.get(f"{namespace}_explanations") or {}
        errors = triplet.get(f"{namespace}_errors") or {}
        row = []
        for idx in range(n_chunks):
            cell = {
                "verdict": verdicts.get(idx),
                "explanation": explanations.get(idx),
            }
            if errors.get(idx):
                cell["error"] = errors[idx]
            row.append(cell)
        return row

    # -- Serialization: none in place; build_report() is the artifact --

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
        logger.info("    Directions:  %s",
                    ", ".join(d.name for d, _ in self._directions))
        logger.info("")

    def _log_results(self) -> None:
        """Step 7: consolidated results — pipeline tree + metrics + tokens once."""
        self._log_bl_results()
        if self.verbosity == "full":
            log_token_stats()
        self._log_done()

    @staticmethod
    def _verdict_counts(results: list[dict], direction_name: str) -> dict:
        """Tally verdicts for one direction across all report entries."""
        flat = direction_name in ("answer2response", "response2answer")
        counts = {"total": 0, "Entailment": 0, "Contradiction": 0,
                  "Neutral": 0, "unknown": 0}
        for entry in results:
            array = entry[direction_name]
            cells = array if flat else [cell for row in array for cell in row]
            for cell in cells:
                counts["total"] += 1
                verdict = cell.get("verdict")
                if verdict in counts:
                    counts[verdict] += 1
                else:
                    counts["unknown"] += 1
        return counts

    def _log_bl_results(self) -> None:
        """Print ── RAGCHECK RESULTS ──: where requests went + the metrics.

        This block is the user's first impression of the whole system —
        general, not per item."""
        if self.verbosity != "full":
            return
        report = self.last_report
        results = report["results"]
        om = report["overall_metrics"]

        logger.info("")
        logger.info(settings.section_rule("RAGCHECK RESULTS", char="═"))
        logger.info("")

        # ── 🔀 Pipeline: where the requests went
        phases: list[tuple[str, str, object, str]] = []
        response_claims = sum(len(e["response_claims"]) for e in results)
        gt_claims = sum(len(e["gt_answer_claims"]) for e in results)
        phases.append(("📝", "extract response",
                       self._extract_response.last_stats,
                       f"{response_claims} claims"))
        phases.append(("📝", "extract gt_answer",
                       self._extract_gt.last_stats,
                       f"{gt_claims} claims"))
        for direction, service in self._directions:
            c = self._verdict_counts(results, direction.name)
            summary = (f"{c['total']} verdicts "
                       f"(🟢 {c['Entailment']} · 🔴 {c['Contradiction']}"
                       f" · ⚪ {c['Neutral']})")
            if c["unknown"]:
                summary += f" · ❓ {c['unknown']}"
            phases.append(("🔎", direction.name, service.last_stats, summary))

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

        # ── 📊 Metrics: the first-impression overview (macro, from the report)
        n = report["_meta"]["evaluated_items"]
        support = om.get("support", {})

        def fmt(name: str) -> str:
            value = om.get(name)
            text = "n/a" if value is None else f"{value:.3f}"
            if support.get(name) is not None and support[name] != n:
                text += f"  (n={support[name]})"
            return text

        logger.info(" 📊 Metrics  (macro over %d items)", n)
        logger.info("    Overall")
        logger.info("    ├─ precision:            %s", fmt("precision"))
        logger.info("    ├─ recall:               %s", fmt("recall"))
        logger.info("    └─ f1:                   %s", fmt("f1"))
        logger.info("    Retriever")
        logger.info("    ├─ claim_recall:         %s", fmt("claim_recall"))
        logger.info("    └─ context_precision:    %s", fmt("context_precision"))
        logger.info("    Generator")
        logger.info("    ├─ faithfulness:         %s", fmt("faithfulness"))
        logger.info("    ├─ hallucination:        %s", fmt("hallucination"))
        logger.info("    ├─ self_knowledge:       %s", fmt("self_knowledge"))
        logger.info("    ├─ context_utilization:  %s", fmt("context_utilization"))
        logger.info("    ├─ noise sens. relevant: %s", fmt("noise_sensitivity_in_relevant"))
        logger.info("    └─ noise sens. irrelev.: %s", fmt("noise_sensitivity_in_irrelevant"))

        abstained = sum(1 for e in results
                        if e.get("is_abstention") and not e.get("extraction_errors"))
        justified = round((om.get("justified_abstention_rate") or 0) * n)
        unjustified = round((om.get("unjustified_abstention_rate") or 0) * n)
        errors = om.get("extraction_errors", {})
        checker_fail = om.get("checker_failure_rate")
        logger.info("    Reliability")
        logger.info("    ├─ abstentions:          %d  (justified %d / unjustified %d)",
                    abstained, justified, unjustified)
        logger.info("    ├─ extraction errors:    %d response · %d gt_answer",
                    errors.get("response", 0), errors.get("gt_answer", 0))
        logger.info("    └─ checker failures:    %s",
                    "n/a" if checker_fail is None else f"{checker_fail * 100:.1f}%")
        logger.info("")

    def _log_done(self) -> None:
        if self.verbosity != "full":
            return
        report = self.last_report
        logger.info(" ✅ Done: ragchecked %d/%d items → report with %d metrics",
                    report["_meta"]["evaluated_items"],
                    report["_meta"]["total_items"], len(METRIC_NAMES))
