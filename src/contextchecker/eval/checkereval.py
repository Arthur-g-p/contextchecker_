"""
Checker Evaluator — triplet-level entailment classification accuracy.

Runs the checker on GT triplets (with human_label) and compares
the checker's predicted verdicts 1:1 against the human annotations.

Architecture:
    - Delegates ALL checking to CheckingService (zero duplication).
    - Owns only: GT validation, verdict comparison, metric computation,
      and the ── CHECKER EVAL ── logging section.
    - Does NOT inherit BaseService — evaluators measure, services mutate.
"""

import asyncio

from contextchecker import settings
from contextchecker.eval.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from contextchecker.models import CheckerEvalResult
from contextchecker.utils import canonicalize_triplets
from contextchecker.services.base import BaseService
from contextchecker.services.checking import CheckingService

logger = settings.get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

LABELS = ["Entailment", "Contradiction", "Neutral"]

# Internal extractor model name used to build the service's kg_key.
# This decouples the user's gt_key from the _response_kg convention.
_INTERNAL_EXT_MODEL = "_gt_eval"


class CheckerEvaluator:
    """Evaluates checker quality by running GT triplets through CheckingService
    and comparing predicted verdicts against human_label annotations.

    The evaluator is a thin orchestrator:
        1. Validates GT data (has human_label? has reference?)
        2. Aliases gt_key → service-compatible key
        3. Strips existing verdicts (force full recompute)
        4. Delegates checking to CheckingService (logs CHECKER RESULTS)
        5. Compares verdicts vs human_labels (1:1 zip)
        6. Computes and logs eval metrics (CHECKER EVAL section)
    """

    def __init__(
        self,
        checker_model: str,
        gt_key: str = "claude2_response_kg",
        checker_base_url: str | None = None,
        concurrency: int = 10,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        max_retries: int | None = None,
    ):
        self._checker_model = checker_model
        self._gt_key = gt_key
        self._service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"
        self._verdict_key = f"{checker_model}_checker_verdict"
        self._explanation_key = f"{checker_model}_checker_explanation"

        # The service owns all checking logic. quiet=True suppresses its
        # pre-execution logging (validation/skip/config) since the
        # evaluator logs its own 📂 Data and ⚙️ Config sections.
        self._service = CheckingService(
            model=checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=checker_base_url,
            concurrency=concurrency,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            max_retries=max_retries,
            quiet=True,
        )

    # ── Public API ───────────────────────────────────────────────

    async def evaluate(self, data: list[dict]) -> CheckerEvalResult:
        """Run the full checker evaluation pipeline.

        Args:
            data: Pre-loaded list of items (eval dataset with GT triplets).

        Returns:
            CheckerEvalResult with accuracy, per-label report, confusion matrix.
        """
        # Step 0: Canonicalize key aliases (context→reference, query→question)
        BaseService._canonicalize_keys(data)

        for item in data:
            if item.get(self._gt_key):
                canonicalize_triplets(item[self._gt_key])

        # Steps 1-2: Validate GT data + extract human labels
        evaluable, gt_labels_map, skip_info = self._prepare_gt(data)
        total_claims = sum(
            len(labels) for labels in gt_labels_map.values()
        )

        # Pre-execution logging
        self._log_data(len(data), skip_info, len(evaluable), total_claims)
        self._log_eval_config()

        # Step 3: Alias GT key → service key, filter to labeled only,
        #         strip existing verdicts, remap indices
        self._prepare_for_service(evaluable, gt_labels_map)

        # Step 4: Delegate to CheckingService
        # This produces: ── Checking ──, ── CHECKER RESULTS ──
        await self._service.run(evaluable)

        # Step 5: Compare verdicts vs human labels
        gt_flat, pred_flat, parse_errors = self._compare(
            evaluable, gt_labels_map
        )

        # Step 6: Build result
        result = self._build_result(
            gt_flat, pred_flat, parse_errors, len(evaluable), skip_info
        )

        # Step 7: Log eval-specific results
        self._log_eval_results(result, gt_flat, pred_flat)
        self._log_done(result)

        return result

    def run_sync(self, data: list[dict]) -> CheckerEvalResult:
        """Sync wrapper — same pattern as BaseService.run_sync."""
        return asyncio.run(self.evaluate(data))

    # ── Pipeline steps (private) ─────────────────────────────────

    def _prepare_gt(
        self, data: list[dict]
    ) -> tuple[list[dict], dict[int, dict[int, str]], dict]:
        """Classify items and extract human labels.

        Items need: gt_key present + reference present + at least one
        triplet with both a valid triplet array AND a human_label.

        Returns:
            evaluable:     Items that pass all checks.
            gt_labels_map: {evaluable_index: {claim_index: human_label}}.
                           Only triplets WITH human_label are included.
            skip_info:     Counts by skip reason (item-level + claim-level).
        """
        evaluable: list[dict] = []
        gt_labels_map: dict[int, dict[int, str]] = {}

        missing_gt = 0
        missing_context = 0
        empty_gt = 0
        total_triplets = 0

        for item in data:
            # 1. Need GT triplets
            if self._gt_key not in item or not item[self._gt_key]:
                missing_gt += 1
                continue

            # 2. Need reference
            reference = item.get("reference", "")
            if not reference:
                missing_context += 1
                continue

            total_triplets += len(item[self._gt_key])

            # 3. Extract triplets with human_label — track by index
            labels: dict[int, str] = {}
            for j, t in enumerate(item[self._gt_key]):
                label = t.get("human_label")
                if label:
                    labels[j] = label

            if not labels:
                empty_gt += 1
                continue

            idx = len(evaluable)
            gt_labels_map[idx] = labels
            evaluable.append(item)

        total_labeled = sum(len(l) for l in gt_labels_map.values())

        skip_info = {
            "missing_gt": missing_gt,
            "missing_context": missing_context,
            "empty_gt": empty_gt,
            "unlabeled_claims": total_triplets - total_labeled,
        }

        if not evaluable:
            from contextchecker.exceptions import InvalidInputError
            total_dropped = missing_gt + missing_context + empty_gt
            raise InvalidInputError(
                f"No evaluable items found. All {total_dropped} items were "
                f"dropped (missing_gt={missing_gt}, missing_context="
                f"{missing_context}, empty_gt={empty_gt})."
            )

        return evaluable, gt_labels_map, skip_info

    def _prepare_for_service(
        self,
        evaluable: list[dict],
        gt_labels_map: dict[int, dict[int, str]],
    ) -> None:
        """Filter to labeled triplets only, alias to service key, strip verdicts.

        Only triplets with human_label are sent to the checker — unlabeled
        triplets are stripped to avoid wasting API tokens on claims we
        can't evaluate.

        Also remaps gt_labels_map indices to sequential 0, 1, 2... to
        stay aligned with the filtered triplet list.
        """
        for i, item in enumerate(evaluable):
            labeled_indices = sorted(gt_labels_map[i].keys())
            all_triplets = item[self._gt_key]

            # Filter: only labeled triplets go to the service
            labeled_triplets = [all_triplets[j] for j in labeled_indices]
            item[self._service_kg_key] = labeled_triplets

            # Remap indices: old sparse {2: "Ent", 5: "Con"} → {0: "Ent", 1: "Con"}
            old_labels = gt_labels_map[i]
            gt_labels_map[i] = {
                new_idx: old_labels[old_idx]
                for new_idx, old_idx in enumerate(labeled_indices)
            }

            # Strip existing verdicts — force recompute ALL
            for triplet in item[self._service_kg_key]:
                triplet.pop(self._verdict_key, None)
                triplet.pop(self._explanation_key, None)

    def _compare(
        self,
        evaluable: list[dict],
        gt_labels_map: dict[int, dict[int, str]],
    ) -> tuple[list[str], list[str], int]:
        """Index-aligned comparison of human_labels vs predicted verdicts.

        Only triplets that have a human_label are compared — the rest are
        skipped (they were checked by the service but have no GT to compare).

        Claims where the checker returned None verdict (parse error)
        are excluded from metrics and counted separately.

        Returns:
            gt_flat:      Human labels for successfully compared claims.
            pred_flat:    Predicted verdicts for successfully compared claims.
            parse_errors: Count of claims with None verdict.
        """
        gt_flat: list[str] = []
        pred_flat: list[str] = []
        parse_errors = 0

        for i, item in enumerate(evaluable):
            if i not in gt_labels_map:
                continue

            labeled_claims = gt_labels_map[i]       # {claim_idx: label}
            triplets = item[self._service_kg_key]

            for claim_idx, label in labeled_claims.items():
                triplet = triplets[claim_idx]
                pred_verdict = triplet.get(self._verdict_key)

                if pred_verdict is None:
                    parse_errors += 1
                    continue

                gt_flat.append(label)
                pred_flat.append(pred_verdict)

        return gt_flat, pred_flat, parse_errors

    def _build_result(
        self,
        gt_flat: list[str],
        pred_flat: list[str],
        parse_errors: int,
        total_items: int,
        skip_info: dict,
    ) -> CheckerEvalResult:
        """Compute accuracy, classification report, and confusion matrix."""
        acc = accuracy_score(gt_flat, pred_flat)

        report = classification_report(
            gt_flat,
            pred_flat,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(gt_flat, pred_flat, labels=LABELS)

        return CheckerEvalResult(
            accuracy=round(acc, 4),
            total_claims=len(gt_flat),
            total_items=total_items,
            parse_errors=parse_errors,
            report=report,
            confusion_matrix={"labels": LABELS, "matrix": cm},
            skipped=skip_info,
        )

    # ── Logging (evaluator-owned sections) ───────────────────────

    def _log_data(
        self,
        total: int,
        skip_info: dict,
        evaluable_count: int,
        total_claims: int,
    ) -> None:
        """Print 📂 Data section — GT-specific validation summary."""
        logger.info(" 📂 Data")
        logger.info("    Total:       %d items", total)

        dropped_gt = skip_info["missing_gt"]
        dropped_ctx = skip_info["missing_context"]
        dropped_empty = skip_info["empty_gt"]
        unlabeled = skip_info.get("unlabeled_claims", 0)

        if dropped_gt > 0:
            logger.info(
                "    ├─ dropped:  %d  (missing GT)", dropped_gt
            )
        if dropped_ctx > 0:
            logger.info(
                "    ├─ dropped:  %d  (missing reference)", dropped_ctx
            )
        if dropped_empty > 0:
            logger.info(
                "    ├─ dropped:  %d  (empty GT after filtering)",
                dropped_empty,
            )
        if unlabeled > 0:
            logger.info(
                "    ├─ skipped:  %d claims  (no human_label)", unlabeled
            )
        logger.info(
            "    └─ valid:    %d items → %s labeled claims",
            evaluable_count,
            f"{total_claims:,}",
        )
        logger.info("")

    def _log_eval_config(self) -> None:
        """Print ⚙️  Config section — evaluator-specific."""
        location = self._checker_model
        if self._service.base_url:
            location += f" @ {self._service.base_url}"

        logger.info(" ⚙️  Config")
        logger.info("    Checker:     %s", location)
        logger.info("    GT key:      %s", self._gt_key)

        if self._service.joint:
            logger.info(
                "    Mode:        joint (max %d claims/call, %d max words)",
                self._service.joint_num,
                self._service.max_words or 0,
            )
        else:
            logger.info("    Mode:        single")

        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_eval_results(
        self,
        result: CheckerEvalResult,
        gt_flat: list[str],
        pred_flat: list[str],
    ) -> None:
        """Print ── CHECKER EVAL ── section: accuracy, report, matrix."""
        logger.info("")
        logger.info(settings.section_rule("CHECKER EVAL"))
        logger.info("")

        # ── Accuracy
        correct = int(result.accuracy * result.total_claims)
        logger.info(
            " 🔎 Accuracy: %.1f%%  (%d / %s)",
            result.accuracy * 100,
            correct,
            f"{result.total_claims:,}",
        )

        if result.parse_errors > 0:
            logger.info(
                "    ⚠️  %d parse errors excluded from metrics",
                result.parse_errors,
            )

        # ── Per-label report (string format for terminal)
        logger.info("")
        logger.info(" 📊 Per-Label Report:")
        report_str = classification_report(
            gt_flat,
            pred_flat,
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
        for line in report_str.splitlines():
            if line.strip():
                logger.info("    %s", line)

        # ── Confusion matrix
        logger.info("")
        logger.info(
            " 📉 Confusion Matrix (rows = GT, cols = Predicted):"
        )
        cm = result.confusion_matrix["matrix"]
        short_labels = ["Ent", "Con", "Neu"]

        header = "   ".join(f"{sl:>8}" for sl in short_labels)
        logger.info("    %16s %s", "", header)
        for i, label in enumerate(LABELS):
            short = short_labels[i]
            row = "   ".join(f"{v:>8}" for v in cm[i])
            logger.info("    %-16s %s", short, row)

    def _log_done(self, result: CheckerEvalResult) -> None:
        """Print ✅ Done summary line."""
        total_skipped = sum(result.skipped.values())
        logger.info("")
        logger.info(
            " ✅ Done: %s claims evaluated (%d claims skipped, "
            "%d parse errors)",
            f"{result.total_claims:,}",
            total_skipped,
            result.parse_errors,
        )
