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
    _location,
    abstention_counts,
    log_pipeline_tree,
    verdict_summary,
    normalize_chunks,
    run_direction,
    unwrap_items,
)
from contextchecker.stats import GLOBAL_STATS, format_headline, log_mece_tree, log_rate_rows, log_token_stats, usage_since
from contextchecker.utils import build_meta, plural

logger = settings.get_logger(__name__)

# Hard drop: every metric family needs all three (only precision/recall
# would survive without chunks - those items belong in the faithfulness
# pipeline, not in a degraded ragcheck).
REQUIRED_KEYS = ("response", "gt_answer", "retrieved_context")


def _missing_keys(item: dict) -> list[str]:
    """Required-key check. Absent or null = missing. An empty response is
    data (a full abstention); an explicit "" gt_answer is data (the
    annotated no-answer convention, docs/abstention.md §2); an empty chunk
    list is not — nothing can be evaluated against no context."""
    missing = [k for k in ("response", "gt_answer")
               if not isinstance(item.get(k), str)]
    if not item.get("retrieved_context"):
        missing.append("retrieved_context")
    return missing

_ENTAILMENT = "Entailment"

# Paper metric names, in the original RAGChecker output order.
METRIC_NAMES = (
    "precision", "recall", "claim_recall", "context_precision",
    "faithfulness", "noise_sensitivity_in_relevant",
    "noise_sensitivity_in_irrelevant", "f1", "hallucination",
    "self_knowledge", "context_utilization",
    # refusal calibration — per-item binaries conditioned on the retrieval
    # judge (docs/abstention.md §3): did the generator do the right thing
    # given what retrieval gave it. None where the situation did not arise.
    "answers_with_relevant_context", "abstains_without_relevant_context",
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

    Gating per project rulings (docs/abstention.md §4):
    - extraction error (either side): item fully excluded - every metric None.
    - abstention: nothing is gated. Recall is judged from the response
      text as the paper does (a refusal entails nothing → 0, F1 0);
      precision and the faithfulness family have no response claims →
      0/0 → None on their own; retrieval metrics are computed as always.
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

    # -- Refusal calibration, conditioned on claim_recall (None: no GT
    #    claims or retrieval unjudged → neither applies) --
    abstained = bool(entry.get("is_abstention"))
    cr = metrics["claim_recall"]
    if cr is not None and cr > 0:
        metrics["answers_with_relevant_context"] = 0.0 if abstained else 1.0
    elif cr == 0.0:
        metrics["abstains_without_relevant_context"] = 1.0 if abstained else 0.0

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
    if r == 0.0:
        metrics["f1"] = 0.0  # R = 0 ⇒ F1 = 0 for any P, defined or not
    elif p is not None and r is not None:
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


def _abstention_breakdown(results: list[dict]) -> dict:
    """Item-level abstention partition (docs/abstention.md §3).

    Judge A (annotation): gt_no_answer — the GT is an explicit "", no
    answer exists. Justified = abstained there; unwarranted = answered
    there. Every abstention on a present GT is unjustified (an answer was
    expected). Judge B (retrieval evidence) apportions its CAUSE only:
    claim_recall == 0 → no chunk entails any GT claim; > 0 → a relevant
    chunk existed; None → relevance unknown (GT text extracted to zero
    claims, or retrieval unjudged).
    """
    counts = abstention_counts(results)
    live = [e for e in results if not e.get("extraction_errors")]
    no_answer = [e for e in live if e.get("gt_no_answer")]
    answerable = [e for e in live if not e.get("gt_no_answer")]
    refused = [e for e in answerable if e.get("is_abstention")]

    def cr(e):
        return e["metrics"].get("claim_recall")

    counts.update({
        "unanswerable": len(no_answer),
        "answerable": len(answerable),
        "answered_answerable": sum(
            1 for e in answerable if not e.get("is_abstention")),
        "justified": sum(1 for e in no_answer if e.get("is_abstention")),
        "unwarranted": sum(1 for e in no_answer if not e.get("is_abstention")),
        "unjustified": len(refused),
        "all_chunks_irrelevant": sum(1 for e in refused if cr(e) == 0.0),
        "relevant_chunk_present": sum(
            1 for e in refused if cr(e) is not None and cr(e) > 0),
        "relevance_unknown": sum(1 for e in refused if cr(e) is None),
    })
    return counts


def _verdict_cell_counts(results: list[dict]) -> tuple[int, int]:
    """(total, none) verdict cells across all four direction arrays,
    skipping extraction-errored items — shared by compute and display."""
    total = none = 0
    for e in results:
        if e.get("extraction_errors"):
            continue
        for key in ("answer2response", "response2answer"):
            for cell in e[key]:
                total += 1
                none += cell.get("verdict") is None
        for key in ("retrieved2response", "retrieved2answer"):
            for row in e[key]:
                for cell in row:
                    total += 1
                    none += cell.get("verdict") is None
    return total, none


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
    ab = _abstention_breakdown(results)
    # No-results leave the denominator: a tooling failure is charged exactly
    # once, in extraction_error_rate — never by diluting a behavior rate.
    # justified and unwarranted are rates over the items with no GT answer;
    # unjustified over the items with one.
    overall["abstention_rate"] = _ratio(ab["abstained"], evaluated - ab["errored"])
    overall["justified_abstention_rate"] = _ratio(ab["justified"], ab["unanswerable"])
    overall["unjustified_abstention_rate"] = _ratio(ab["unjustified"], ab["answerable"])
    overall["unwarranted_answer_rate"] = _ratio(ab["unwarranted"], ab["unanswerable"])
    overall["abstention_counts"] = {
        k: ab[k] for k in ("answerable", "unanswerable", "justified",
                           "unjustified", "unwarranted",
                           "all_chunks_irrelevant", "relevant_chunk_present",
                           "relevance_unknown")
    }

    overall["extraction_error_rate"] = _ratio(len(errored), evaluated)
    overall["extraction_errors"] = {
        "response": sum(1 for e in errored
                        if e["extraction_errors"].get("response")),
        "gt_answer": sum(1 for e in errored
                         if e["extraction_errors"].get("gt_answer")),
    }

    # Judge reliability: share of None verdicts across every issued check.
    total_cells, none_cells = _verdict_cell_counts(results)
    overall["checker_failure_rate"] = _ratio(none_cells, total_cells)

    return overall


class RagCheckerPipeline(BaseService):
    """2 extractions + 4 checking directions composed into one run."""

    # Headline of the run line and the Done line (the Overall group).
    _RUN_SUMMARY_KEYS = ("precision", "recall", "f1")

    _VARIANCE_SECTIONS = {
        "metrics": [
            ("Overall", ["precision", "recall", "f1"]),
            ("Retriever", ["claim_recall", "context_precision"]),
            ("Generator", ["faithfulness", "hallucination", "self_knowledge",
                           "answers_with_relevant_context",
                           "abstains_without_relevant_context",
                           "context_utilization",
                           "noise_sensitivity_in_relevant",
                           "noise_sensitivity_in_irrelevant"]),
        ],
        # Footer rates of the ⚪ tree only (rule set 4.1): abstention_rate
        # stays in the JSON as distribution information but has no printed
        # per-run derivation, so it does not enter the variance block.
        "behavior": ["justified_abstention_rate",
                     "unjustified_abstention_rate", "unwarranted_answer_rate"],
        "health": ["extraction_error_rate", "checker_failure_rate"],
    }
    # How to read each row — the round-bracket aid in the Metrics tree
    # and the variance block. Missing key = no direction of its own.
    _METRIC_DIRECTIONS = {
        "precision": "higher is better", "recall": "higher is better", "f1": "higher is better",
        "claim_recall": "higher is better", "context_precision": "higher is better",
        "faithfulness": "higher is better", "hallucination": "lower is better",
        "self_knowledge": "depends on goals",
        "answers_with_relevant_context": "higher is better",
        "abstains_without_relevant_context": "depends on goals",
        "context_utilization": "higher is better",
        "noise_sensitivity_in_relevant": "lower is better",
        "noise_sensitivity_in_irrelevant": "lower is better",
        "justified_abstention_rate": "higher is better",
        "unjustified_abstention_rate": "lower is better",
        "unwarranted_answer_rate": "lower is better",
    }
    _VARIANCE_LABELS = {
        "answers_with_relevant_context": "answers when context is relevant",
        "abstains_without_relevant_context": "abstains when context is irrelevant",
        "noise_sensitivity_in_relevant": "noise sensitivity relevant",
        "noise_sensitivity_in_irrelevant": "noise sensitivity irrelevant",
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
        2. Filter       - none (v1: no skipping)
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
        self._usage_at_start = GLOBAL_STATS.snapshot()
        data = unwrap_items(data)                     # Step 0: accept the
        self._canonicalize_keys(data)                 # {"results": [...]}
        valid = self._validate(data)                  # envelope; query→question

        self._filter(valid)                           # 2 (no-op)
        if announce:                                  # 3
            self._log_validation(len(data), len(valid))
            self._log_config()

        # Children print their own labeled section rules (compact mode).
        await self._extract_response.run(valid)       # 4
        await self._extract_gt.run(valid)

        self._prefill_known_verdicts(valid)
        for direction, service in self._directions:
            await run_direction(service, valid, direction)

        self._serialize()                             # 5 (no-op)
        self.last_report = self.build_report(data)    # 6: the single artifact
        if report:
            self._log_results()                       # 7: consolidated results
        return data

    def _prefill_known_verdicts(self, items: list[dict]) -> None:
        """Verdicts that are known before any request is sent.

        An explicit "" GT has no reference to check response claims
        against: the answer2response check cannot even be built, so
        precision is 0 by necessity. Likewise an empty "" response (a
        full abstention, docs/abstention.md §1) entails no GT claim, so
        response2answer is Neutral throughout and recall is 0 — the
        non-delivery §4 charges. Without this, an empty reference would
        leave those cells unjudged and recall null instead of 0. A
        refusal *text* is still judged by the checker, as the paper does:
        the extractor's abstention call can be wrong, and the text may
        entail a GT claim after all.

        The verdicts are written without a request; the checking service
        then skips those claims as already judged (claim-level
        resumability)."""
        a2r = next(s for d, s in self._directions if d.name == "answer2response")
        r2a = next(s for d, s in self._directions if d.name == "response2answer")
        for item in items:
            if item.get("gt_no_answer"):
                for triplet in item.get(self._response_kg) or []:
                    triplet[a2r.verdict_key] = "Neutral"
                    triplet[a2r.explanation_key] = (
                        "no ground-truth answer exists — not sent to the checker")
            if item.get("response", "").strip() == "":
                for triplet in item.get(self._gt_kg) or []:
                    triplet[r2a.verdict_key] = "Neutral"
                    triplet[r2a.explanation_key] = (
                        "empty response entails nothing — not sent to the checker")

    # _run_repeated inherited from BaseService (variance mode)

    # -- Validation --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Hard drop - required keys must be present (see
        _missing_keys: absent/null is missing, an empty string is data).

        An explicit "" gt_answer marks the item gt_no_answer (annotated:
        no answer exists). Surviving items get retrieved_context
        normalized in place to [{doc_id, text}] dicts (bare strings get
        synthesized ids).
        """
        valid = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.debug("Item %d is not an object (%s) - skipping.",
                             i, type(item).__name__)
                continue
            missing = _missing_keys(item)
            if missing:
                logger.debug("Item %d missing %s - skipping.",
                             i, ", ".join(missing))
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                "No items contain 'response', 'gt_answer' (strings) "
                "and a non-empty 'retrieved_context'."
            )

        for item in valid:
            item["retrieved_context"] = normalize_chunks(item["retrieved_context"])
            if item["gt_answer"].strip() == "":
                item["gt_no_answer"] = True
            else:
                item.pop("gt_no_answer", None)
        return valid

    def _filter(self, valid):
        """No skipping in v1 (2.0 feature)."""
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
            if _missing_keys(item):
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
                request_strategies=GLOBAL_STATS.strategies(),
                usage=usage_since(getattr(self, "_usage_at_start", None)),
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
            "gt_no_answer": bool(item.get("gt_no_answer", False)),
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
        checking = self._directions[0][1]
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", _location(self._extract_response))
        logger.info("    Checker:     %s", _location(checking))
        logger.info("    Mode:        %s", checking.mode_label)
        logger.info("    Directions:  %s",
                    ", ".join(d.name for d, _ in self._directions))
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(self) -> None:
        """Step 7: consolidated results — pipeline tree + metrics + tokens once."""
        self._log_bl_results()
        self._log_done()
        if self.verbosity == "full":
            log_token_stats()

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

        logger.info(settings.section_rule("RAGCHECK RESULTS", char="═"))
        logger.info("")

        # ── 🔀 Pipeline: where the requests went
        phases: list[tuple[str, str, object, str]] = []
        response_claims = sum(len(e["response_claims"]) for e in results)
        gt_claims = sum(len(e["gt_answer_claims"]) for e in results)
        phases.append(("📝", "extract response",
                       self._extract_response.last_stats,
                       f"{response_claims} {plural(response_claims, 'claim')}"))
        phases.append(("📝", "extract gt_answer",
                       self._extract_gt.last_stats,
                       f"{gt_claims} {plural(gt_claims, 'claim')}"))
        for direction, service in self._directions:
            c = self._verdict_counts(results, direction.name)
            phases.append(("🔎", direction.name, service.last_stats,
                           verdict_summary(c)))
        log_pipeline_tree(phases)

    def _log_metrics(self) -> None:
        """📊 Metrics: the first-impression overview (macro, from the report)."""
        if self.verbosity != "full":
            return
        report = self.last_report
        om = report["overall_metrics"]

        n = report["_meta"]["evaluated_items"]
        support = om.get("support", {})

        def fmt(name: str) -> str:
            value = om.get(name)
            text = "n/a" if value is None else f"{value:.3f}"
            notes = []
            if support.get(name) is not None and support[name] != n:
                notes.append(f"{support[name]} of {n} items")
            if value is not None and name in self._METRIC_DIRECTIONS:
                notes.append(self._METRIC_DIRECTIONS[name])
            if notes:
                text += f"  ({' · '.join(notes)})"
            return text

        logger.info(" 📊 Metrics  (macro over %d items)", n)
        logger.info("    Overall — how well the response matches the"
                    " ground-truth answer")
        logger.info("     ├─ %-38s%s", "precision:", fmt("precision"))
        logger.info("     ├─ %-38s%s", "recall:", fmt("recall"))
        logger.info("     └─ %-38s%s", "f1:", fmt("f1"))
        logger.info("    Retriever — did retrieval bring the needed evidence")
        logger.info("     ├─ %-38s%s", "claim recall:", fmt("claim_recall"))
        logger.info("     └─ %-38s%s", "context precision:",
                    fmt("context_precision"))
        logger.info("    Generator — how the response used, ignored, or"
                    " invented beyond the context")
        logger.info("     ├─ %-38s%s", "faithfulness:", fmt("faithfulness"))
        logger.info("     ├─ %-38s%s", "hallucination:", fmt("hallucination"))
        logger.info("     ├─ %-38s%s", "self knowledge:", fmt("self_knowledge"))
        logger.info("     ├─ %-38s%s", "answers when context is relevant:",
                    fmt("answers_with_relevant_context"))
        logger.info("     ├─ %-38s%s", "abstains when context is irrelevant:",
                    fmt("abstains_without_relevant_context"))
        logger.info("     ├─ %-38s%s", "context utilization:",
                    fmt("context_utilization"))
        logger.info("     ├─ %-38s%s", "noise sensitivity relevant:",
                    fmt("noise_sensitivity_in_relevant"))
        logger.info("     └─ %-38s%s", "noise sensitivity irrelevant:",
                    fmt("noise_sensitivity_in_irrelevant"))

        logger.info("")

    def _log_abstention(self) -> None:
        """⚪ Abstention Behavior tree — judge: retrieval evidence.

        Extraction failures branch in only when present: they sit inside
        the rate denominator (evaluated items), so the tree owes the
        reconciliation."""
        if self.verbosity != "full":
            return
        ab = _abstention_breakdown(self.last_report["results"])
        # Behavior only: extraction-failed items are out of the tree AND
        # out of the rate denominators (charged once, in 💥 Reliability).
        top = ab["evaluated"] - ab["errored"]
        note = None
        if ab["errored"]:
            note = (f"{ab['errored']} extraction-failed excluded"
                    " — see 💥 Reliability")

        log_mece_tree(
            "⚪ Abstention Behavior", top, "evaluated items",
            [
                ("🔬", ab["answered_answerable"], "answered",
                 "GT present — scored"),
                ("✅", ab["justified"],
                 plural(ab["justified"], "justified abstention"),
                 "GT empty — correct silence"),
                ("❌", ab["unjustified"],
                 plural(ab["unjustified"], "unjustified abstention"),
                 "GT present — charged in recall",
                 [("refused with relevant chunks", ab["relevant_chunk_present"],
                   "a chunk entails a GT claim — generator fault"),
                  ("refused without relevant chunks", ab["all_chunks_irrelevant"],
                   "no chunk entails any GT claim — retriever fault"),
                  ("refused, relevant chunks unknown", ab["relevance_unknown"],
                   "GT extracted to zero claims, or retrieval unjudged")]),
                ("❌", ab["unwarranted"],
                 plural(ab["unwarranted"], "unwarranted answer"),
                 "GT empty — charged in precision"),
            ],
            footer=[("justified abstention rate", ab["justified"],
                     ab["unanswerable"], "unanswerable"),
                    ("unjustified abstention rate", ab["unjustified"],
                     ab["answerable"], "answerable"),
                    ("unwarranted answer rate", ab["unwarranted"],
                     ab["unanswerable"], "unanswerable")],
            header_note=note,
        )
        logger.info("")

    def _log_reliability(self) -> None:
        """💥 Reliability rate rows — harness health, always printed."""
        if self.verbosity != "full":
            return
        report = self.last_report
        ab = _abstention_breakdown(report["results"])
        total_cells, none_cells = _verdict_cell_counts(report["results"])
        sources = report["overall_metrics"].get("extraction_errors", {})
        causes = {k: v for k, v in sources.items() if v} or None
        log_rate_rows(
            "💥 Reliability",
            [("📝", "Extraction", ab["errored"], ab["evaluated"],
              "items failed", "extraction_error_rate", causes),
             ("🔎", "Checking", none_cells, total_cells,
              "verdicts unjudged", "checker_failure_rate", None)],
            header_note="tooling — excluded from all metrics,"
                        " counted once here",
        )
        logger.info("")

    def _log_done(self) -> None:
        if self.verbosity != "full":
            return
        report = self.last_report
        n = report["_meta"]["evaluated_items"]
        logger.info(" ✅ Done: %d %s · %s", n, plural(n, "item"),
                    format_headline(report.get("overall_metrics", {}),
                                    self._RUN_SUMMARY_KEYS))
