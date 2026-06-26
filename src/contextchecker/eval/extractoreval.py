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
from dataclasses import dataclass

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.models import ExtractorEvalResult
from contextchecker.services.base import BaseService
from contextchecker.services.checking import CheckingService
from contextchecker.services.extraction import ExtractionService

logger = settings.get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Internal model name for the checking service's kg_key.
_INTERNAL_EXT_MODEL = "_exteval"


# ── Item classification ─────────────────────────────────────────────────────

@dataclass
class _ItemBucket:
    """Post-extraction classification into four buckets."""
    to_compare: list[dict]          # has GT + has predictions → normal matching
    wrongful_answer: list[dict]     # no GT but model predicted → FP-like
    wrongful_abstention: list[dict] # has GT but no predictions → FN penalty
    correct_abstention: list[dict]  # no GT, no predictions → ignored


# ── Matching result per item ─────────────────────────────────────────────────

@dataclass
class _ItemMatchResult:
    """Matching outcome for a single valid item.

    RAGChecker uses two independent coverage counts (no shared TP / no min):
      - tp_recall:    GT triplets entailed by the predictions  (recall numerator)
      - tp_precision: pred triplets entailed by the GT         (precision numerator)
    """
    tp_recall: int
    tp_precision: int
    fp: int
    fn: int
    false_positives: list[dict]     # disagreement detail per FP
    false_negatives: list[dict]     # disagreement detail per FN


class ExtractorEvaluator:
    """Evaluates extraction quality by extracting live then matching against GT.

    Flow: validate → extract (API) → classify → match (LLM) → metrics.

    Matching mode:
        - LLM: 2-pass batched checker with semantic equivalence prompt.
          Pass 1: GT→Pred (recall). Pass 2: Pred→GT (precision).

    Returns (ExtractorEvalResult, disagreements) — CLI writes two files.
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
    ):
        if not checker_model:
            raise ValueError("checker_model is required for evaluation matching.")

        self._extractor_model = extractor_model
        self._gt_key = gt_key
        # Predictions are always written by ExtractionService(model=extractor_model)
        # under this key — it is derived, never an override.
        self._pred_key = f"{extractor_model}_response_kg"
        self._method = "llm"

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

        # Extraction service — always needed (quiet, evaluator owns logging)
        self._extraction_service = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            max_retries=extractor_max_retries,
            quiet=True,
        )

        # LLM config — CheckingService built on demand in _match_all_llm
        self._checker_model = checker_model
        self._checker_base_url = checker_base_url
        self._concurrency = concurrency
        self._joint_num = joint_num
        self._max_words = max_words
        self._max_retries = max_retries

    # ── Public API ───────────────────────────────────────────────

    async def evaluate(
        self, data: list[dict]
    ) -> tuple[ExtractorEvalResult, list[dict]]:
        """Run the full extractor evaluation pipeline.

        Args:
            data: Pre-loaded list of items (GT-annotated dataset with response text).

        Returns:
            (ExtractorEvalResult, disagreements) — result dataclass + per-item details.
        """
        # Step 0: Canonicalize key aliases
        BaseService._canonicalize_keys(data)

        # Step 1: Validate — fail-fast on bad input
        valid = self._validate(data)

        # Pre-extraction logging
        self._log_data_pre(len(data), len(data) - len(valid), len(valid))
        self._log_eval_config()

        # Step 2: Run extraction (quiet — evaluator owns the logging)
        await self._extraction_service.run(valid)

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
        result = self._build_result(item_results, buckets, len(data))

        # Step 7: Build disagreement list
        disagreements = self._build_disagreements(buckets, item_results)

        # Step 8: Log eval results
        self._log_eval_results(result)
        self._log_done(result)

        return result, disagreements

    def run_sync(
        self, data: list[dict]
    ) -> tuple[ExtractorEvalResult, list[dict]]:
        """Sync wrapper — same pattern as BaseService.run_sync."""
        return asyncio.run(self.evaluate(data))

    # ── Validation (fail-fast) ───────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Fail-fast validation: items must have a response to extract from.

        Items missing 'response' are dropped — we can't extract without text.
        Missing GT is NOT an error — it's the trap for wrongful_answer detection.
        If zero survive → InvalidInputError.
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

        return valid

    # ── Classification (post-extraction) ─────────────────────────

    def _classify(self, data: list[dict]) -> _ItemBucket:
        """Post-extraction classification into four buckets.

        Called AFTER ExtractionService.run() — pred_key may or may not
        be populated per item depending on extraction results.
        """
        to_compare, wrongful_answer, wrongful_abstention, correct_abstention = [], [], [], []

        for item in data:
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
        )

    # ── Result building (encapsulated) ───────────────────────────

    def _build_result(
        self,
        item_results: list[_ItemMatchResult],
        buckets: _ItemBucket,
        total_items: int,
    ) -> ExtractorEvalResult:
        """Aggregate the two coverage counts + abstention penalties, compute P/R/F1.

        RAGChecker semantics:
            recall    = entailed_GT   / total_GT      (coverage of ground truth)
            precision = entailed_pred / total_pred    (correctness of predictions)
        Two independent ratios — no shared TP, no min().
        """
        # Accumulate the two numerators + the two error counts from matched items
        tp_recall, tp_precision, fp, fn = 0, 0, 0, 0
        for ir in item_results:
            tp_recall += ir.tp_recall
            tp_precision += ir.tp_precision
            fp += ir.fp
            fn += ir.fn

        # Wrongful abstention: every GT triplet is an uncovered GT claim (FN)
        abstention_fn_penalty = sum(
            len(item[self._gt_key]) for item in buckets.wrongful_abstention
        )
        fn += abstention_fn_penalty

        # Wrongful answer: every predicted triplet is an unsupported prediction (FP)
        answer_fp_penalty = sum(
            len(item[self._pred_key]) for item in buckets.wrongful_answer
        )
        fp += answer_fp_penalty

        # Two independent ratios. Denominators:
        #   tp_recall + fn    == total GT claims    (incl. wrongful-abstention GT)
        #   tp_precision + fp == total pred claims  (incl. wrongful-answer preds)
        recall = tp_recall / (tp_recall + fn) if (tp_recall + fn) > 0 else 0.0
        precision = tp_precision / (tp_precision + fp) if (tp_precision + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

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

        return ExtractorEvalResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            tp_recall=tp_recall,
            tp_precision=tp_precision,
            fp=fp,
            fn=fn,
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
                "wrongful_abstention_fn_penalty": abstention_fn_penalty,
                "wrongful_answer_fp_penalty": answer_fp_penalty,
            },
            correct_abstention=len(buckets.correct_abstention),
            method=self._method,
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
        logger.info("── Matching (LLM 2-pass) ─────────────────────────────────────")

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
            quiet=True,
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
            quiet=True,
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

            # Count matches from Pass 1 (recall)
            fn_list = []
            tp_from_recall = 0
            for j, t in enumerate(pass1_triplets):
                verdict = t.get(verdict_key)
                if verdict == "Entailment":
                    tp_from_recall += 1
                else:
                    fn_list.append({
                        "gt_triplet": self._triplet_to_str(gt_triplets[j]),
                        "verdict": verdict,
                        "reason": t.get(explanation_key) or verdict or "Parse error",
                    })

            # Count FPs from Pass 2 (precision)
            fp_list = []
            tp_from_precision = 0
            for j, t in enumerate(pass2_triplets):
                verdict = t.get(verdict_key)
                if verdict == "Entailment":
                    tp_from_precision += 1
                else:
                    fp_list.append({
                        "pred_triplet": self._triplet_to_str(pred_triplets[j]),
                        "verdict": verdict,
                        "reason": t.get(explanation_key) or verdict or "Parse error",
                    })

            # RAGChecker: two independent coverage counts, no shared TP, no min().
            #   recall  side: GT triplets entailed by predictions (tp_from_recall)
            #   precision side: pred triplets entailed by GT      (tp_from_precision)
            # FN = GT not covered; FP = predictions not supported.
            fn = len(gt_triplets) - tp_from_recall
            fp = len(pred_triplets) - tp_from_precision

            results.append(_ItemMatchResult(
                tp_recall=tp_from_recall,
                tp_precision=tp_from_precision,
                fp=fp, fn=fn,
                false_positives=fp_list,
                false_negatives=fn_list,
            ))

        return results

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
        verdicts_list_key = f"{self._checker_model}_checker_verdicts"

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
                        "subject": t.get("subject", t.get("triplet", [""])[0] if "triplet" in t else ""),
                        "predicate": t.get("predicate", t.get("triplet", ["", ""])[1] if "triplet" in t else ""),
                        "object": t.get("object", t.get("triplet", ["", "", ""])[2] if "triplet" in t else ""),
                    }
                    for t in claim_triplets
                ],
            }

            # Strip any existing verdicts — force fresh compute
            for triplet in synth[service_kg_key]:
                triplet.pop(verdict_key, None)
                triplet.pop(explanation_key, None)
            synth.pop(verdicts_list_key, None)

            pass_items.append(synth)

        return pass_items

    @staticmethod
    def _triplet_to_str(triplet: dict) -> str:
        """Convert a triplet dict to a natural-language string.

        Supports both legacy (triplet: [s,p,o]) and canonical formats.
        """
        if "subject" in triplet:
            return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"
        t = triplet.get("triplet", [])
        if t and len(t) >= 3:
            return f"{t[0]} {t[1]} {t[2]}"
        return str(triplet)

    # ── Disagreement collection ──────────────────────────────────

    def _build_disagreements(
        self,
        buckets: _ItemBucket,
        item_results: list[_ItemMatchResult],
    ) -> list[dict]:
        """Build the per-item disagreement list for error analysis.

        Includes: valid items with FP/FN, wrongful answers, wrongful abstentions.
        Skipped items and perfect matches are excluded.
        """
        disagreements = []

        # to_compare items with disagreements
        for i, (item, ir) in enumerate(zip(buckets.to_compare, item_results)):
            if ir.fp == 0 and ir.fn == 0:
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
        logger.info("    └─ to_extract: %d items", valid)
        logger.info("")

    def _log_data_post(self, buckets: _ItemBucket) -> None:
        """Print post-extraction classification summary."""
        to_compare = len(buckets.to_compare)
        wa = len(buckets.wrongful_answer)
        wab = len(buckets.wrongful_abstention)
        ca = len(buckets.correct_abstention)

        logger.info("")
        logger.info(" 📂 Post-Extraction Classification")
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

        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_eval_results(self, result: ExtractorEvalResult) -> None:
        """Print ── EXTRACTOR EVAL ── section: P/R/F1, stats, abstentions."""
        logger.info("")
        logger.info(
            "── EXTRACTOR EVAL ──────────────────────────────────────────"
        )
        logger.info("")

        # ── Matching quality
        logger.info(" 🔎 Matching Quality")
        logger.info(
            "    Precision:  %.3f   (%d / %d)",
            result.precision, result.tp_precision, result.tp_precision + result.fp,
        )
        logger.info(
            "    Recall:     %.3f   (%d / %d)",
            result.recall, result.tp_recall, result.tp_recall + result.fn,
        )
        logger.info("    F1:         %.3f", result.f1)

        # ── Extraction stats
        logger.info("")
        logger.info(" 📊 Extraction Stats")
        logger.info(
            "    GT:    %d triplets across %d items  (avg %.1f/item)",
            result.gt_stats["total_triplets"],
            result.to_compare_items,
            result.gt_stats["avg_per_item"],
        )
        logger.info(
            "    Pred:  %d triplets across %d items  (avg %.1f/item)",
            result.pred_stats["total_triplets"],
            result.to_compare_items,
            result.pred_stats["avg_per_item"],
        )
        delta = result.pred_stats["avg_per_item"] - result.gt_stats["avg_per_item"]
        if abs(delta) > 0.05:
            direction = "over-extraction" if delta > 0 else "under-extraction"
            logger.info("    Delta: %+.1f/item  (%s)", delta, direction)

        # ── Abstention errors (only if any)
        wa = result.abstention_errors["wrongful_answer"]
        wab = result.abstention_errors["wrongful_abstention"]
        if wa > 0 or wab > 0:
            logger.info("")
            logger.info(" ⚠️  Abstention Errors")
            if wa > 0:
                logger.info(
                    "    Wrongful answers:     %d items → %d FP added",
                    wa, result.abstention_errors["wrongful_answer_fp_penalty"],
                )
            if wab > 0:
                logger.info(
                    "    Wrongful abstentions: %d items → %d FN added",
                    wab, result.abstention_errors["wrongful_abstention_fn_penalty"],
                )

    def _log_done(self, result: ExtractorEvalResult) -> None:
        """Print ✅ Done summary line."""
        logger.info("")
        logger.info(
            " ✅ Done: %d items matched (%d GT covered, %d preds correct, %d FP, %d FN)",
            result.to_compare_items, result.tp_recall, result.tp_precision,
            result.fp, result.fn,
        )
