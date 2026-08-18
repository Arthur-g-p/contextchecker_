"""
Extractor Evaluator — triplet-level extraction quality measurement.

Runs extraction live on GT-annotated data, then compares predicted
triplets against ground-truth using set-to-set IR matching.

Architecture:
    - Delegates extraction to ExtractionService (quiet — evaluator owns logging).
    - Matching mode: LLM (online, 2-pass checker).
    - Strict separation: _validate (fail-fast) → extract → _classify (bucket sort).
    - Owns: GT validation, matching orchestration, IR metric computation,
      disagreement collection, and the ── EXTRACTOR EVAL ── logging section.
    - Does NOT inherit BaseService — evaluators measure, services mutate.
"""

import asyncio
import copy
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.models import ExtractorEvalResult
from contextchecker.stats import (
    log_multi_run_hint,
    log_token_stats,
    log_variance_block,
)
from contextchecker.utils import (
    build_meta,
    build_variance,
    canonicalize_triplets,
    find_duplicate_triplets,
)
from contextchecker.services.base import BaseService
from contextchecker.services.checking import CheckingService
from contextchecker.services.extraction import ExtractionService
from contextchecker.services.atomization import AtomizationService

logger = settings.get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Internal model name for the checking service's kg_key.
_INTERNAL_EXT_MODEL = "_exteval"


# ── Item classification ─────────────────────────────────────────────────────

@dataclass
class _ItemBucket:
    """Post-extraction classification into five buckets."""
    to_compare: list[dict]          # has GT + has predictions → normal matching
    wrongful_answer: list[dict]     # no GT but model predicted → FP-like
    wrongful_abstention: list[dict] # has GT but no predictions → FN penalty
    correct_abstention: list[dict]  # no GT, no predictions → ignored
    # extraction failed (tooling) → excluded from ALL metrics, counted in
    # the error rate
    extraction_error: list[dict] = field(default_factory=list)


# ── Matching result per item ─────────────────────────────────────────────────

@dataclass
class _ItemMatchResult:
    """Matching outcome for a single valid item.

    RAGChecker uses two independent coverage counts (no shared TP / no min):
      - tp_recall:    GT triplets entailed by the predictions  (recall numerator)
      - tp_precision: pred triplets entailed by the GT         (precision numerator)

    fp/fn count judged misses only. Claims the checker never returned a
    verdict for (tooling failure) sit in unjudged_* — excluded from both
    numerator and denominator, never charged to the extractor.
    """
    tp_recall: int
    tp_precision: int
    fp: int
    fn: int
    false_positives: list[dict]     # disagreement detail per FP
    false_negatives: list[dict]     # disagreement detail per FN
    unjudged_gt: list[dict] = field(default_factory=list)    # pass 1 checker failures
    unjudged_pred: list[dict] = field(default_factory=list)  # pass 2 checker failures


class ExtractorEvaluator:
    """Evaluates extraction quality by extracting live then matching against GT.

    Flow: validate → extract (API) → classify → match (LLM) → metrics.

    Matching mode:
        - LLM: 2-pass batched checker with semantic equivalence prompt.
          Pass 1: GT→Pred (recall). Pass 2: Pred→GT (precision).

    Returns (summary_doc, disagreements_doc) — CLI writes two files verbatim.
    """

    def __init__(
        self,
        extractor_model: str,
        gt_key: str = "claude2_response_kg",
        # Extraction config
        extractor_base_url: str | None = None,
        extractor_max_retries: int | None = None,
        # LLM matching config
        checker_model: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        max_retries: int | None = None,
        # Atomicity axis (optional, orthogonal to coverage)
        atomizer_model: str | None = None,
        atomizer_base_url: str | None = None,
        runs: int = 1,
    ):
        if not checker_model:
            raise ValueError("checker_model is required for evaluation matching.")

        self._extractor_model = extractor_model
        self._gt_key = gt_key
        # runs > 1 = variance mode: the evaluator repeats itself; the
        # controller only passes the number through.
        self._runs = max(1, runs)
        # Predictions are always written by ExtractionService(model=extractor_model)
        # under this key — it is derived, never an override.
        self._pred_key = f"{extractor_model}_response_kg"
        self._error_key = f"{extractor_model}_extraction_error"

        # Guard: the predicted-triplet key must not collide with the GT key.
        # On collision (e.g. extractor_model="claude2" with the default gt_key)
        # extraction would target the GT slot and the evaluator would match
        # ground truth against itself, silently reporting perfect P/R/F1.
        if self._pred_key == self._gt_key:
            raise InvalidInputError(
                f"Predicted-triplet key collides with GT key: '{self._gt_key}'. "
                f"The predicted key is derived as '{{extractor_model}}_response_kg' "
                f"(here '{self._pred_key}'). Use a different --extractor-model or "
                f"--gt-key so the two do not overlap."
            )

        # Extraction service — always needed (quiet, evaluator owns logging).
        # dedup=False: the eval measures duplicates as an independent dimension,
        # so predictions must stay raw — we never evaluate on deduplicated data.
        self._extraction_service = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            max_retries=extractor_max_retries,
            verbosity="compact",
            dedup=False,
        )

        # LLM config — CheckingService built on demand in _match_all_llm
        self._checker_model = checker_model
        self._checker_base_url = checker_base_url
        self._concurrency = concurrency
        self._joint_num = joint_num
        self._max_words = max_words
        self._max_retries = max_retries

        # Atomicity axis — OPTIONAL. Built only if an atomizer model is given AND
        # the key is configured (AtomizationService.__init__ would otherwise raise
        # on a missing key). Measured on a throwaway copy of the predictions, so
        # coverage numbers and the output file are never affected.
        self._atomizer_model = atomizer_model
        self._atomization_service = None
        self._atomizer_skip_reason = None
        if not atomizer_model:
            self._atomizer_skip_reason = "no --atomizer-model"
        elif not settings.ATOMIZER_API_KEY:
            self._atomizer_skip_reason = "ATOMIZER_API_KEY not set"
        else:
            self._atomization_service = AtomizationService(
                model=atomizer_model,
                source_kg_key=self._pred_key,
                base_url=atomizer_base_url,
                concurrency=concurrency,
                verbosity="compact",
            )

    # ── Public API ───────────────────────────────────────────────

    async def evaluate(self, data: list[dict]) -> tuple[dict, dict]:
        """Run the eval; with runs > 1, repeat it and report variance.

        Returns:
            (summary_doc, disagreements_doc). Multi-run: the summary carries
            flat means + variance sibling + runs = N complete normal
            documents; the disagreements document mirrors with a runs array.
        """
        if self._runs <= 1:
            return await self._evaluate_once(data)

        # Variance mode. Evaluators narrate their full sections every run
        # (evaluator verbosity levels are a later cleanup); the VARIANCE
        # block still lands once at the end.
        log_multi_run_hint(self._runs)
        summaries: list[dict] = []
        disagreements: list[dict] = []
        total_start = time.perf_counter()
        for run in range(1, self._runs + 1):
            logger.info("")
            logger.info(settings.section_rule(f"Run {run}/{self._runs}"))
            started = time.perf_counter()
            summary, disagreement = await self._evaluate_once(copy.deepcopy(data))
            duration = round(time.perf_counter() - started, 1)
            for doc in (summary, disagreement):
                doc["_meta"]["run"] = run
                doc["_meta"]["duration_seconds"] = duration
            summaries.append(summary)
            disagreements.append(disagreement)

        means, variance = build_variance(
            [{k: v for k, v in d.items() if k != "_meta"} for d in summaries])
        durations = [d["_meta"]["duration_seconds"] for d in summaries]
        total = round(time.perf_counter() - total_start, 1)
        log_variance_block(self._runs, means, variance, durations, total)

        def _outer_meta(doc: dict) -> dict:
            meta = {k: v for k, v in doc["_meta"].items()
                    if k not in ("run", "duration_seconds")}
            meta["runs"] = self._runs
            meta["duration_seconds"] = total
            return meta

        summary_doc = {"_meta": _outer_meta(summaries[0]),
                       **means, "variance": variance, "runs": summaries}
        disagreements_doc = {"_meta": _outer_meta(disagreements[0]),
                             "runs": disagreements}
        return summary_doc, disagreements_doc

    async def _evaluate_once(self, data: list[dict]) -> tuple[dict, dict]:
        """Run the full extractor evaluation pipeline once.

        Args:
            data: Pre-loaded list of items (GT-annotated dataset with response text).

        Returns:
            (summary_doc, disagreements_doc) — two ready-to-write JSON
            documents incl. _meta. The CLI only resolves paths and dumps.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()

        # Step 0: Canonicalize key aliases
        BaseService._canonicalize_keys(data)

        # Step 1: Validate — fail-fast on bad input
        valid = self._validate(data)

        for item in valid:
            if item.get(self._gt_key):
                canonicalize_triplets(item[self._gt_key])

        # Pre-extraction logging
        self._log_data_pre(len(data), len(data) - len(valid), len(valid))
        self._log_eval_config()

        # Step 2: Run extraction (quiet — evaluator owns the logging)
        await self._extraction_service.run(valid)

        for item in valid:
            if item.get(self._pred_key):
                canonicalize_triplets(item[self._pred_key])

        # Step 2b: Atomicity axis (optional, orthogonal). Measured on a COPY so
        # coverage and the output file keep the raw predictions untouched.
        atomicity = await self._measure_atomicity(valid)

        # Step 2c: Duplicate-claims axis (orthogonal, read-only). The eval never
        # evaluates on deduplicated data — it only reports the duplicates.
        duplicates = self._measure_duplicates(valid)

        # Step 3: Classify into buckets (now that pred exists)
        buckets = self._classify(valid)

        # Post-extraction logging
        self._log_data_post(buckets)

        # Step 4: Guard — need at least one to_compare item to match
        if not buckets.to_compare:
            logger.warning(
                "No items with both GT and predictions after extraction. "
                "All %d items were wrongful abstentions.",
                len(buckets.wrongful_abstention),
            )

        # Step 5: Match to_compare items (LLM)
        if buckets.to_compare:
            item_results = await self._match_all_llm(buckets.to_compare)
        else:
            item_results = []

        # Step 6: Build result (encapsulated)
        result = self._build_result(item_results, buckets, len(data), atomicity, duplicates)

        # Step 7: Build disagreement list
        disagreements = self._build_disagreements(buckets, item_results)

        # Step 8: Log eval results; the evaluator owns the token table
        self._log_eval_results(result)
        log_token_stats()
        self._log_done(result)

        # Step 9: Assemble the two ready-to-write documents. The CLI only
        # resolves paths and dumps JSON - it never composes content.
        return self._assemble_documents(result, disagreements)

    def run_sync(self, data: list[dict]) -> tuple[dict, dict]:
        """Sync wrapper — same pattern as BaseService.run_sync."""
        return asyncio.run(self.evaluate(data))

    # ── Output documents (serialization owned by the evaluator) ──

    def _assemble_documents(
        self,
        result: ExtractorEvalResult,
        disagreements: list[dict],
    ) -> tuple[dict, dict]:
        """Build the two output documents the CLI writes verbatim.

        Split details are debug material: they move from the summary's
        atomicity block into the disagreements document.
        """
        meta = build_meta(
            "extractor_eval",
            timestamp=self._started_at,
            duration_seconds=time.perf_counter() - self._started_perf,
            total_items=result.total_items,
            evaluated_items=result.to_compare_items,
            dropped_items=result.total_items - result.to_compare_items,
            pred_key=self._pred_key,
            matching="llm-2-pass",
        )

        summary_doc = {"_meta": meta, **asdict(result)}
        atomicity_splits = None
        if summary_doc.get("atomicity"):
            # asdict() deep-copied — the dataclass keeps its splits.
            atomicity_splits = summary_doc["atomicity"].pop("splits", None)

        disagreements_doc = {
            "_meta": meta,
            "total_disagreements": len(disagreements),
            "items": disagreements,
        }
        if atomicity_splits:
            disagreements_doc["atomicity_splits"] = atomicity_splits

        return summary_doc, disagreements_doc

    # ── Atomicity axis (optional, orthogonal to coverage) ────────

    async def _measure_atomicity(self, valid: list[dict]) -> dict | None:
        """Measure how atomic the extractor's predictions are.

        Runs the atomizer on a DEEP COPY of the items so the predictions used
        for coverage matching (and the output file) are never mutated — this is
        a pure measurement side-channel. Skipped (returns None) when no atomizer
        is configured.
        """
        if self._atomization_service is None:
            logger.info(
                " ⏭️  Atomicity skipped (%s)", self._atomizer_skip_reason
            )
            return None

        # Nothing to measure if no item carries predictions yet.
        if not any(item.get(self._pred_key) for item in valid):
            logger.info(" ⏭️  Atomicity skipped (no predictions to measure)")
            return None

        sandbox = copy.deepcopy(valid)
        await self._atomization_service.run(sandbox)
        trace = self._atomization_service.last_trace

        keep = split = failed = atomic_units = 0
        splits: list[dict] = []
        for item in trace:
            for d in item.get("decisions", []):
                dec = d.get("decision")
                if dec == "split":
                    split += 1
                    atomic_units += len(d.get("children", []))
                    # Full split detail — routed to the disagreements file by
                    # the CLI so the summary JSON stays lean.
                    orig = d.get("original", {})
                    splits.append({
                        "id": item.get("id"),
                        "original": self._triplet_to_str(orig) if orig else None,
                        "children": [
                            self._triplet_to_str(c) for c in d.get("children", [])
                        ],
                        "reasoning": d.get("reasoning"),
                    })
                elif dec == "failed":
                    failed += 1
                else:
                    keep += 1
                    atomic_units += 1

        extracted = keep + split + failed
        evaluated = keep + split
        rate = keep / evaluated if evaluated else 1.0
        density = atomic_units / evaluated if evaluated else 0.0
        return {
            "extracted_claims": extracted,
            "evaluated_claims": evaluated,
            "atomic_units": atomic_units,
            "new_claims_from_splits": atomic_units - evaluated,
            "non_atomic": split,
            "failed": failed,
            "atomicity_rate": round(rate, 4),
            "information_density": round(density, 2),
            "splits": splits,
        }

    # ── Duplicate-claims axis (orthogonal, read-only) ────────────

    def _measure_duplicates(self, valid: list[dict]) -> dict | None:
        """Count exact (by string) duplicate claims in the predictions.

        Fully independent of coverage: read-only, never mutates predictions, the
        eval is never run on deduplicated data. The duplicate triplets are listed
        per item so they land in the results file. Returns None when no item
        carries predictions.
        """
        if not any(item.get(self._pred_key) for item in valid):
            return None

        predicted = duplicate = 0
        items_out: list[dict] = []
        for item in valid:
            preds = item.get(self._pred_key) or []
            predicted += len(preds)
            dups = find_duplicate_triplets(preds)
            if dups:
                duplicate += len(dups)
                items_out.append({
                    "id": item.get("id"),
                    "duplicates": [self._triplet_to_str(t) for t in dups],
                })

        rate = duplicate / predicted if predicted else 0.0
        return {
            "predicted_claims": predicted,
            "unique_claims": predicted - duplicate,
            "duplicate_claims": duplicate,
            "duplicate_rate": round(rate, 4),
            "items": items_out,
        }

    # ── Validation (fail-fast) ───────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Fail-fast validation: items must have a response to extract from.

        Items missing 'response' are dropped — we can't extract without text.
        Missing GT is NOT an error — it's the trap for wrongful_answer detection.
        If zero survive → InvalidInputError.

        GT presence is judged on the KEY, not its contents: an empty list is a
        deliberate abstention trap. No item carrying the key at all means wrong
        file or wrong --gt-key — fatal, and caught before any LLM call.
        """
        valid = []
        dropped = 0

        for i, item in enumerate(data):
            if not item.get("response"):
                dropped += 1
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                f"No evaluable items. All {len(data)} items dropped "
                f"(missing_response={dropped})."
            )

        missing_key = sum(1 for item in valid if self._gt_key not in item)

        if missing_key == len(valid):
            raise InvalidInputError(
                f"No ground truth found: none of the {len(valid)} evaluable "
                f"items contain the key '{self._gt_key}'. `eval extractor` "
                f"measures the extractor against labeled GT — check the input "
                f"file, or point --gt-key at the right key."
            )

        self._missing_gt_count = missing_key

        return valid

    # ── Classification (post-extraction) ─────────────────────────

    def _classify(self, data: list[dict]) -> _ItemBucket:
        """Post-extraction classification into four buckets.

        Called AFTER ExtractionService.run() — pred_key may or may not
        be populated per item depending on extraction results.
        """
        to_compare, wrongful_answer, wrongful_abstention, correct_abstention = [], [], [], []
        extraction_error = []

        for item in data:
            # Tooling failure — empty predictions here mean NOTHING about the
            # extractor's abstention behavior. Own bucket, out of all metrics.
            if self._error_key in item:
                extraction_error.append(item)
                continue

            has_gt = bool(item.get(self._gt_key))
            has_pred = bool(item.get(self._pred_key))

            if has_gt and has_pred:
                to_compare.append(item)
            elif not has_gt and has_pred:
                wrongful_answer.append(item)
            elif has_gt and not has_pred:
                wrongful_abstention.append(item)
            else:
                correct_abstention.append(item)

        return _ItemBucket(
            to_compare=to_compare,
            wrongful_answer=wrongful_answer,
            wrongful_abstention=wrongful_abstention,
            correct_abstention=correct_abstention,
            extraction_error=extraction_error,
        )

    # ── Result building (encapsulated) ───────────────────────────

    def _build_result(
        self,
        item_results: list[_ItemMatchResult],
        buckets: _ItemBucket,
        total_items: int,
        atomicity: dict | None = None,
        duplicates: dict | None = None,
    ) -> ExtractorEvalResult:
        """Aggregate judged counts + abstention penalties, compute P/R/F1.

        RAGChecker semantics:
            recall    = entailed_GT   / total_GT      (coverage of ground truth)
            precision = entailed_pred / total_pred    (correctness of predictions)
        Two independent ratios — no shared TP, no min().

        Denominators carry judged claims + penalties. Unjudged claims
        (checker failure) leave numerator AND denominator; an empty
        denominator yields None, never 0.0 — nothing judged is not a score.
        """
        # Accumulate the judged counts + unjudged tallies from matched items
        covered = sum(ir.tp_recall for ir in item_results)
        missed = sum(ir.fn for ir in item_results)
        supported = sum(ir.tp_precision for ir in item_results)
        unsupported = sum(ir.fp for ir in item_results)
        unjudged_gt = sum(len(ir.unjudged_gt) for ir in item_results)
        unjudged_pred = sum(len(ir.unjudged_pred) for ir in item_results)

        # Wrongful abstention: every GT triplet is an uncovered GT claim (FN)
        abstention_fn_penalty = sum(
            len(item[self._gt_key]) for item in buckets.wrongful_abstention
        )

        # Wrongful answer: every predicted triplet is an unsupported prediction (FP)
        answer_fp_penalty = sum(
            len(item[self._pred_key]) for item in buckets.wrongful_answer
        )

        recall_den = covered + missed + abstention_fn_penalty
        precision_den = supported + unsupported + answer_fp_penalty

        recall = round(covered / recall_den, 4) if recall_den > 0 else None
        precision = round(supported / precision_den, 4) if precision_den > 0 else None
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = round(2 * precision * recall / (precision + recall), 4)

        # Exhaustive partitions: total = judged + penalty + unjudged.
        recall_counts = {
            "total_gt_claims": recall_den + unjudged_gt,
            "covered": covered,
            "missed": missed,
            "wrongful_abstention_penalty": abstention_fn_penalty,
            "unjudged": unjudged_gt,
            "denominator": recall_den,
        }
        precision_counts = {
            "total_pred_claims": precision_den + unjudged_pred,
            "supported": supported,
            "unsupported": unsupported,
            "wrongful_answer_penalty": answer_fp_penalty,
            "unjudged": unjudged_pred,
            "denominator": precision_den,
        }

        # Checker failures — eval tooling reliability, co-equal with the
        # extraction error rate but never mixed into P/R/F1. Denominator:
        # every verdict actually asked of the checker (penalty claims were
        # never sent, so they don't count as issued).
        issued = covered + missed + supported + unsupported + unjudged_gt + unjudged_pred
        unjudged_total = unjudged_gt + unjudged_pred
        checker_failures = {
            "count": unjudged_total,
            "issued_verdicts": issued,
            "rate": round(unjudged_total / issued, 4) if issued else 0.0,
            "items_affected": sum(
                1 for ir in item_results if ir.unjudged_gt or ir.unjudged_pred
            ),
            "unjudged_gt": unjudged_gt,
            "unjudged_pred": unjudged_pred,
        }

        # Extraction stats
        gt_triplets = sum(
            len(item[self._gt_key])
            for item in buckets.to_compare + buckets.wrongful_abstention
        )
        pred_triplets = sum(
            len(item[self._pred_key])
            for item in buckets.to_compare + buckets.wrongful_answer
        )
        to_compare_count = len(buckets.to_compare)
        gt_avg = gt_triplets / to_compare_count if to_compare_count > 0 else 0.0
        pred_avg = pred_triplets / to_compare_count if to_compare_count > 0 else 0.0

        # Extraction error rate — tooling reliability, co-equal with P/R/F1
        # but never mixed into it. Denominator: every item that went through
        # the extraction step (all five buckets).
        error_items = buckets.extraction_error
        attempted = (
            to_compare_count + len(buckets.wrongful_answer)
            + len(buckets.wrongful_abstention) + len(buckets.correct_abstention)
            + len(error_items)
        )
        by_cause: dict[str, int] = {}
        for item in error_items:
            cause = item.get(self._error_key, "unknown")
            by_cause[cause] = by_cause.get(cause, 0) + 1
        extraction_errors = {
            "count": len(error_items),
            "rate": round(len(error_items) / attempted, 4) if attempted else 0.0,
            "by_cause": by_cause,
        }

        return ExtractorEvalResult(
            precision=precision,
            recall=recall,
            f1=f1,
            recall_counts=recall_counts,
            precision_counts=precision_counts,
            total_items=total_items,
            to_compare_items=to_compare_count,
            gt_stats={
                "total_triplets": gt_triplets,
                "avg_per_item": round(gt_avg, 2),
            },
            pred_stats={
                "total_triplets": pred_triplets,
                "avg_per_item": round(pred_avg, 2),
            },
            abstention_errors={
                "wrongful_answer": len(buckets.wrongful_answer),
                "wrongful_abstention": len(buckets.wrongful_abstention),
            },
            correct_abstention=len(buckets.correct_abstention),
            checker_failures=checker_failures,
            atomicity=atomicity,
            duplicates=duplicates,
            extraction_errors=extraction_errors,
        )

    # ── LLM matching (2-pass) ────────────────────────────────────

    async def _match_all_llm(
        self, valid_items: list[dict]
    ) -> list[_ItemMatchResult]:
        """Run LLM 2-pass matching across all valid items.

        Pass 1 (GT → Pred): GT triplets as claims, pred triplets as reference.
            Entailment = extractor found this GT claim (contributes to recall).
        Pass 2 (Pred → GT): Pred triplets as claims, GT triplets as reference.
            Non-Entailment = hallucinated/wrong prediction (contributes to FP).

        Uses checker_prompt_EVAL_JOINT with {{response}} context.
        """
        logger.info(settings.section_rule("Matching (LLM 2-pass)"))

        # Build a CheckingService with the eval prompt
        service = CheckingService(
            model=self._checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=self._checker_base_url,
            concurrency=self._concurrency,
            joint=True,
            joint_num=self._joint_num,
            max_words=self._max_words,
            max_retries=self._max_retries,
            verbosity="compact",
            joint_prompt_key="checker_prompt_EVAL_JOINT",
        )

        # ── Pass 1: GT → Pred (recall / FN detection) ──────────
        logger.info("")
        logger.info("  Pass 1: GT → Pred (recall)")
        pass1_items = self._build_pass_items(
            valid_items, claims_key=self._gt_key, ref_key=self._pred_key
        )

        await service.run(pass1_items)

        # ── Pass 2: Pred → GT (precision / FP detection) ───────
        logger.info("")
        logger.info("  Pass 2: Pred → GT (precision)")
        pass2_items = self._build_pass_items(
            valid_items, claims_key=self._pred_key, ref_key=self._gt_key
        )

        # Fresh service to reset per-run stats
        service2 = CheckingService(
            model=self._checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=self._checker_base_url,
            concurrency=self._concurrency,
            joint=True,
            joint_num=self._joint_num,
            max_words=self._max_words,
            max_retries=self._max_retries,
            verbosity="compact",
            joint_prompt_key="checker_prompt_EVAL_JOINT",
        )

        await service2.run(pass2_items)

        # ── Collect results ────────────────────────────────────
        service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"
        verdict_key = f"{self._checker_model}_checker_verdict"
        explanation_key = f"{self._checker_model}_checker_explanation"

        results = []
        for i, item in enumerate(valid_items):
            gt_triplets = item[self._gt_key]
            pred_triplets = item[self._pred_key]

            # Pass 1 verdicts: one per GT triplet
            pass1_triplets = pass1_items[i][service_kg_key]
            # Pass 2 verdicts: one per pred triplet
            pass2_triplets = pass2_items[i][service_kg_key]

            # Pass 1 (recall): GT triplets the predictions failed to cover → FN
            tp_from_recall, fn_list, unjudged_gt = self._count_pass(
                pass1_triplets, gt_triplets,
                verdict_key, explanation_key, "gt_triplet",
            )

            # Pass 2 (precision): predictions the GT does not support → FP
            tp_from_precision, fp_list, unjudged_pred = self._count_pass(
                pass2_triplets, pred_triplets,
                verdict_key, explanation_key, "pred_triplet",
            )

            # RAGChecker: two independent coverage counts, no shared TP, no min().
            #   recall  side: GT triplets entailed by predictions (tp_from_recall)
            #   precision side: pred triplets entailed by GT      (tp_from_precision)
            # FN/FP derive from the judged misses alone — unjudged claims
            # (checker failure) leave numerator and denominator entirely.
            results.append(_ItemMatchResult(
                tp_recall=tp_from_recall,
                tp_precision=tp_from_precision,
                fp=len(fp_list), fn=len(fn_list),
                false_positives=fp_list,
                false_negatives=fn_list,
                unjudged_gt=unjudged_gt,
                unjudged_pred=unjudged_pred,
            ))

        return results

    @staticmethod
    def _count_pass(
        checked: list[dict],
        originals: list[dict],
        verdict_key: str,
        explanation_key: str,
        triplet_field: str,
    ) -> tuple[int, list[dict], list[dict]]:
        """Count one matching pass: checker verdicts → (entailed, misses, unjudged).

        Three-way split:
            "Entailment"       → entailed (TP for this pass)
            other real verdict → miss (a judged disagreement: FN or FP)
            None               → unjudged (the checker never returned a verdict
                                 after all retries — a tooling failure)

        A claim nobody judged is no evidence about the extractor: unjudged
        claims leave numerator AND denominator, same rule as ragcheck's
        None-verdict propagation. They must never be charged as misses.

        Args:
            checked: Triplets carrying the pass's verdicts (parallel to originals).
            originals: The original triplets of this pass's claims side.
            triplet_field: Key name for the triplet string in miss/unjudged
                entries ("gt_triplet" for pass 1, "pred_triplet" for pass 2).
        """
        entailed = 0
        misses: list[dict] = []
        unjudged: list[dict] = []
        for j, t in enumerate(checked):
            verdict = t.get(verdict_key)
            if verdict == "Entailment":
                entailed += 1
            elif verdict is None:
                unjudged.append({
                    triplet_field: ExtractorEvaluator._triplet_to_str(originals[j]),
                    "cause": "checker_failure",
                })
            else:
                misses.append({
                    triplet_field: ExtractorEvaluator._triplet_to_str(originals[j]),
                    "verdict": verdict,
                    "reason": t.get(explanation_key) or verdict,
                })
        return entailed, misses, unjudged

    def _build_pass_items(
        self,
        valid_items: list[dict],
        claims_key: str,
        ref_key: str,
    ) -> list[dict]:
        """Build synthetic items for a checker pass.

        Triplets from claims_key become the claims to check.
        Triplets from ref_key are formatted as the reference text.
        The original response is preserved for the eval prompt's {{response}}.
        """
        service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"
        verdict_key = f"{self._checker_model}_checker_verdict"
        explanation_key = f"{self._checker_model}_checker_explanation"

        pass_items = []
        for item in valid_items:
            claim_triplets = item[claims_key]
            ref_triplets = item[ref_key]

            # Format ref triplets as numbered text passages
            ref_text = "\n".join(
                f"[{i+1}] {self._triplet_to_str(t)}"
                for i, t in enumerate(ref_triplets)
            )

            # Build the synthetic item the service expects
            synth = {
                # Reference = formatted triplets from the "other" set
                "reference": [ref_text],
                # Response text for the eval prompt's {{response}} variable
                "response": item.get("response", ""),
                # Claims = triplets to check, under the service's kg_key
                service_kg_key: [
                    {
                        "subject": t["subject"],
                        "predicate": t["predicate"],
                        "object": t["object"],
                    }
                    for t in claim_triplets
                ],
            }

            # Strip any existing verdicts — force fresh compute
            for triplet in synth[service_kg_key]:
                triplet.pop(verdict_key, None)
                triplet.pop(explanation_key, None)

            pass_items.append(synth)

        return pass_items

    @staticmethod
    def _triplet_to_str(triplet: dict) -> str:
        """Convert a triplet dict to a natural-language string."""
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"

    @staticmethod
    def _fmt_ratio(value: float | None, numerator: int, denominator: int) -> str:
        """Format a metric with its fraction; None = empty denominator."""
        if value is None:
            return f"n/a  ({numerator} / {denominator} — nothing judged)"
        return f"{value:.3f}  ({numerator} / {denominator})"

    # ── Disagreement collection ──────────────────────────────────

    def _build_disagreements(
        self,
        buckets: _ItemBucket,
        item_results: list[_ItemMatchResult],
    ) -> list[dict]:
        """Build the per-item disagreement list for error analysis.

        Includes: valid items with FP/FN, wrongful answers, wrongful
        abstentions, and extraction errors (so failed items are identifiable).
        Skipped items and perfect matches are excluded.
        """
        disagreements = []

        # Extraction errors — tooling failures, listed for identification only
        for i, item in enumerate(buckets.extraction_error):
            disagreements.append({
                "id": item.get("id", f"extraction-error-{i}"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                "error_type": "extraction_error",
                "cause": item.get(self._error_key, "unknown"),
            })

        # to_compare items with disagreements — or with unjudged claims:
        # a checker failure is not a disagreement, but the affected item
        # must stay identifiable in this file.
        for i, (item, ir) in enumerate(zip(buckets.to_compare, item_results)):
            unjudged = ir.unjudged_gt + ir.unjudged_pred
            if ir.fp == 0 and ir.fn == 0 and not unjudged:
                continue  # perfect match, no disagreement

            disagreements.append({
                "id": item.get("id", f"to-compare-{i}"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                "tp_recall": ir.tp_recall,
                "tp_precision": ir.tp_precision,
                "fp": ir.fp,
                "fn": ir.fn,
                "gt_triplets": [
                    self._triplet_to_str(t) for t in item[self._gt_key]
                ],
                "pred_triplets": [
                    self._triplet_to_str(t) for t in item[self._pred_key]
                ],
                "false_positives": ir.false_positives,
                "false_negatives": ir.false_negatives,
                # Checker failures on this item — excluded from fp/fn above.
                "unjudged": unjudged,
            })

        # Wrongful answers — model predicted but shouldn't have
        for i, item in enumerate(buckets.wrongful_answer):
            pred_count = len(item[self._pred_key])
            disagreements.append({
                "id": item.get("id", f"wrongful-answer-{i}"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                "error_type": "wrongful_answer",
                "tp": 0,
                "fp": pred_count,
                "fn": 0,
                "gt_triplets": [],
                "pred_triplets": [
                    self._triplet_to_str(t) for t in item[self._pred_key]
                ],
                "false_positives": [
                    {
                        "pred_triplet": self._triplet_to_str(t),
                        "verdict": "no comparison made.",
                        "reason": "No GT — wrongful answer",
                    }
                    for t in item[self._pred_key]
                ],
                "false_negatives": [],
            })

        # Wrongful abstentions — model missed everything
        for i, item in enumerate(buckets.wrongful_abstention):
            gt_count = len(item[self._gt_key])
            disagreements.append({
                "id": item.get("id", f"wrongful-abstention-{i}"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                "error_type": "wrongful_abstention",
                "tp": 0,
                "fp": 0,
                "fn": gt_count,
                "gt_triplets": [
                    self._triplet_to_str(t) for t in item[self._gt_key]
                ],
                "pred_triplets": [],
                "false_positives": [],
                "false_negatives": [
                    {
                        "gt_triplet": self._triplet_to_str(t),
                        "verdict": "no comparison made.",
                        "reason": "Wrongful abstention — all GT lost",
                    }
                    for t in item[self._gt_key]
                ],
            })

        return disagreements

    # ── Logging (evaluator-owned sections) ───────────────────────

    def _log_data_pre(
        self, total: int, dropped: int, valid: int
    ) -> None:
        """Print 📂 Data section — pre-extraction validation summary."""
        logger.info(" 📂 Data")
        logger.info("    Total:       %d items", total)
        if dropped > 0:
            logger.info("    ├─ dropped:  %d  (missing response)", dropped)
        missing_gt = getattr(self, "_missing_gt_count", 0)
        logger.info("    └─ to extract: %d items", valid)
        if missing_gt:
            logger.info("       ├─ with GT key: %d", valid - missing_gt)
            logger.info("       └─ no GT key:   %d  (⚠️ wrongful-answer traps)",
                        missing_gt)
        logger.info("")

    def _log_data_post(self, buckets: _ItemBucket) -> None:
        """Print post-extraction classification summary."""
        to_compare = len(buckets.to_compare)
        wa = len(buckets.wrongful_answer)
        wab = len(buckets.wrongful_abstention)
        ca = len(buckets.correct_abstention)
        err = len(buckets.extraction_error)

        logger.info("")
        logger.info(" 📂 Post-Extraction Classification")
        if err > 0:
            logger.info("    ├─ extraction error:    %d items  (tooling failure → excluded from metrics)", err)
        if wa > 0:
            logger.info("    ├─ wrongful answer:     %d items  (predicted but no GT)", wa)
        if wab > 0:
            fn_penalty = sum(
                len(item[self._gt_key]) for item in buckets.wrongful_abstention
            )
            logger.info(
                "    ├─ wrongful abstention: %d items  (GT present, 0 predictions → %d FN)",
                wab, fn_penalty,
            )
        if ca > 0:
            logger.info("    ├─ correct abstention:   %d items  (no GT and no predictions)", ca)
        logger.info(
            "    └─ to_compare:    %d items  (GT + predictions present)", to_compare
        )
        logger.info("")

    def _log_eval_config(self) -> None:
        """Print ⚙️  Config section — evaluator-specific."""
        logger.info(" ⚙️  Config")

        ext_location = self._extractor_model
        if self._extraction_service.base_url:
            ext_location += f" @ {self._extraction_service.base_url}"
        logger.info("    Extractor:   %s", ext_location)
        logger.info("    GT key:      %s", self._gt_key)

        location = self._checker_model
        if self._checker_base_url:
            location += f" @ {self._checker_base_url}"
        logger.info("    Matching:    LLM 2-pass (%s)", location)

        if self._atomization_service is not None:
            logger.info("    Atomicity:   %s", self._atomizer_model)
        else:
            logger.info("    Atomicity:   skipped (%s)", self._atomizer_skip_reason)

        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_eval_results(self, result: ExtractorEvalResult) -> None:
        """Print ── EXTRACTOR EVAL ── section: P/R/F1, stats, abstentions."""
        logger.info("")
        logger.info(settings.section_rule("EXTRACTOR EVAL"))
        logger.info("")

        # ── Matching quality: one funnel per side. Every issued claim lands
        # in exactly one branch; the first three sum to the denominator, the
        # 💥 branch is explicitly outside it.
        rc = result.recall_counts
        pc = result.precision_counts
        ae = result.abstention_errors

        logger.info(" 🔎 Matching Quality  (LLM 2-pass)")
        logger.info("    Recall — %d total GT claims", rc["total_gt_claims"])
        logger.info("     ├─ ✅ %d covered by predictions  (judged)", rc["covered"])
        logger.info("     ├─ ❌ %d missed  (judged)", rc["missed"])
        if rc["wrongful_abstention_penalty"]:
            logger.info(
                "     ├─ ⚪ %d wrongful-abstention penalty  (%d items, 0 predictions for %d claims)",
                rc["wrongful_abstention_penalty"], ae["wrongful_abstention"],
                rc["wrongful_abstention_penalty"],
            )
        else:
            logger.info("     ├─ ⚪ 0 wrongful-abstention penalty")
        if rc["unjudged"]:
            logger.info(
                "     ├─ 💥 %d unjudged by checker  (excluded from evaluation — no verdict returned)",
                rc["unjudged"],
            )
        else:
            logger.info("     ├─ 💥 0 unjudged by checker")
        logger.info("     └─ → Recall %s", self._fmt_ratio(result.recall, rc["covered"], rc["denominator"]))

        logger.info("    Precision — %d total predicted claims", pc["total_pred_claims"])
        logger.info("     ├─ ✅ %d supported by GT  (judged)", pc["supported"])
        logger.info("     ├─ ❌ %d unsupported  (judged)", pc["unsupported"])
        if pc["wrongful_answer_penalty"]:
            logger.info(
                "     ├─ ⚪ %d wrongful-answer penalty  (%d items, no GT for %d claims)",
                pc["wrongful_answer_penalty"], ae["wrongful_answer"],
                pc["wrongful_answer_penalty"],
            )
        else:
            logger.info("     ├─ ⚪ 0 wrongful-answer penalty")
        if pc["unjudged"]:
            logger.info(
                "     ├─ 💥 %d unjudged by checker  (excluded from evaluation — no verdict returned)",
                pc["unjudged"],
            )
        else:
            logger.info("     ├─ 💥 0 unjudged by checker")
        logger.info("     └─ → Precision %s", self._fmt_ratio(result.precision, pc["supported"], pc["denominator"]))

        logger.info("    F1: %s", "n/a" if result.f1 is None else f"{result.f1:.3f}")

        # ── Extraction stats
        logger.info("")
        logger.info(" 📊 Extraction Stats")
        logger.info(
            "    GT:    %d claims across %d items  (avg %.1f/item)",
            result.gt_stats["total_triplets"],
            result.to_compare_items,
            result.gt_stats["avg_per_item"],
        )
        logger.info(
            "    Pred:  %d claims across %d items  (avg %.1f/item)",
            result.pred_stats["total_triplets"],
            result.to_compare_items,
            result.pred_stats["avg_per_item"],
        )
        delta = result.pred_stats["avg_per_item"] - result.gt_stats["avg_per_item"]
        if abs(delta) > 0.05:
            direction = "over-extraction" if delta > 0 else "under-extraction"
            logger.info("    Delta: %+.1f/item  (%s)", delta, direction)

        # ── Eval tooling failures — the eval's own reliability, co-equal
        # headline with P/R/F1 but never mixed into it. Extraction and
        # checker failures speak the same language: excluded, counted, rated.
        ee = result.extraction_errors or {"count": 0, "rate": 0.0, "by_cause": {}}
        cf = result.checker_failures
        if ee["count"] > 0 or cf["count"] > 0:
            logger.info("")
            logger.info(" 💥 Eval Tooling Failures  (excluded from all metrics)")
            causes = ", ".join(
                f"{cause}: {n}"
                for cause, n in sorted(ee["by_cause"].items(), key=lambda kv: -kv[1])
            )
            logger.info(
                "     ├─ Extraction:  %d items  (%.1f%%)%s",
                ee["count"], ee["rate"] * 100, f"  → {causes}" if causes else "",
            )
            if cf["count"] > 0:
                # warning level: the measurement is partial — this must
                # survive quieter log configurations.
                logger.warning(
                    "     └─ Checker:     %d of %d verdicts  (%.1f%%, %d items)"
                    " — run `eval checker` to qualify '%s'",
                    cf["count"], cf["issued_verdicts"], cf["rate"] * 100,
                    cf["items_affected"], self._checker_model,
                )
            else:
                logger.info(
                    "     └─ Checker:     0 of %d verdicts  (0.0%%)",
                    cf["issued_verdicts"],
                )

        # ── Atomicity (orthogonal to coverage; only if measured)
        a = result.atomicity
        if a:
            logger.info("")
            logger.info(" 🧬 Atomicity")
            if a.get("failed"):
                logger.info("    ⚠️  %d claims failed atomization (omitted from stats)", a["failed"])
            logger.info(
                "    Evaluated:   %d claims → %d atomic units",
                a.get("evaluated_claims", a["extracted_claims"] - a.get("failed", 0)), 
                a["atomic_units"],
            )
            logger.info(
                "    Non-atomic:  %d  (atomicity %.1f%%)",
                a["non_atomic"], a["atomicity_rate"] * 100,
            )
            logger.info("    Density:     %.2f facts/claim", a["information_density"])

        # ── Duplicates (orthogonal to coverage; read-only, never deduped here)
        d = result.duplicates
        if d:
            logger.info("")
            logger.info(" 🔁 Duplicates")
            logger.info(
                "    Found %d exact duplicates  (%.1f%% of %d predicted)",
                d["duplicate_claims"], d["duplicate_rate"] * 100, d["predicted_claims"],
            )
            for entry in d["items"]:
                logger.info("    item %s:", entry["id"])
                for triplet_str in entry["duplicates"]:
                    logger.info("      • %s", triplet_str)

    def _log_done(self, result: ExtractorEvalResult) -> None:
        """Print ✅ Done summary line."""
        rc = result.recall_counts
        pc = result.precision_counts
        logger.info("")
        logger.info(
            " ✅ Done: %d items compared"
            " · recall %d/%d GT claims (%d missed, %d penalty)"
            " · precision %d/%d predictions (%d unsupported, %d penalty)"
            "%s",
            result.to_compare_items,
            rc["covered"], rc["denominator"],
            rc["missed"], rc["wrongful_abstention_penalty"],
            pc["supported"], pc["denominator"],
            pc["unsupported"], pc["wrongful_answer_penalty"],
            (f" · 💥 {result.checker_failures['count']} unjudged excluded"
             if result.checker_failures["count"] else ""),
        )
