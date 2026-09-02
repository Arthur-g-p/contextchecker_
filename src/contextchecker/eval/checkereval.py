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
import copy
import time
from dataclasses import asdict
from datetime import datetime

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.eval.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from contextchecker.models import CheckerEvalResult
from contextchecker.stats import (
    GLOBAL_STATS,
    VarianceTracker,
    log_mece_tree,
    log_multi_run_hint,
    log_rate_rows,
    log_run_line,
    log_token_stats,
)
from contextchecker.utils import build_meta, canonicalize_triplets
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

    _RUN_SUMMARY_KEYS = ("accuracy", "macro_f1")

    # No behavior section: abstention is a foreign concept here
    # (pre-labeled GT triplets in, verdicts out).
    _VARIANCE_SECTIONS = {
        "metrics": [
            ("Overall", ["accuracy", "macro_f1"]),
            ("Per label", ["entailment_f1", "contradiction_f1",
                           "neutral_f1"]),
        ],
        "behavior": [],
        "health": ["checker_failure_rate"],
    }

    def __init__(
        self,
        checker_model: str,
        gt_key: str = "claude2_response_kg",
        checker_base_url: str | None = None,
        concurrency: int = 10,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        runs: int = 1,
    ):
        self._checker_model = checker_model
        # runs > 1 = variance mode: the evaluator repeats itself; the
        # controller only passes the number through.
        self._runs = max(1, runs)
        self._gt_key = gt_key
        self._service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"
        self._verdict_key = f"{checker_model}_checker_verdict"
        self._explanation_key = f"{checker_model}_checker_explanation"

        # The service owns all checking logic. compact verbosity keeps its
        # per-phase API/BL blocks but leaves the pre-exec sections and the
        # token table to the evaluator. In variance mode the service runs
        # silent: per-run plumbing is cut, the eval blocks + VARIANCE
        # report instead (progress bars are not logging and remain).
        self._service = CheckingService(
            model=checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=checker_base_url,
            concurrency=concurrency,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            verbosity="silent" if self._runs > 1 else "compact",
        )

    # ── Public API ───────────────────────────────────────────────

    async def evaluate(self, data: list[dict]) -> dict:
        """Run the eval; with runs > 1, repeat it and report variance.

        Returns:
            Ready-to-write JSON document incl. _meta. Multi-run: flat means
            + variance sibling + runs = N complete normal documents.
        """
        if self._runs <= 1:
            return await self._evaluate_once(data)

        # Variance mode. Evaluators narrate their result sections every run
        # (evaluator verbosity levels are a later cleanup); Data/Config are
        # announced once, and the VARIANCE block + token table land once at
        # the end.
        log_multi_run_hint(self._runs)
        docs: list[dict] = []
        tracker = VarianceTracker(self._VARIANCE_SECTIONS)
        for run in range(1, self._runs + 1):
            logger.info("")
            logger.info(settings.section_rule(f"Run {run}/{self._runs}"))
            started = time.perf_counter()
            doc = await self._evaluate_once(
                copy.deepcopy(data), announce=(run == 1))
            doc["_meta"]["run"] = run
            doc["_meta"]["duration_seconds"] = round(
                time.perf_counter() - started, 1)
            docs.append(doc)
            tracker.add({k: v for k, v in doc.items() if k != "_meta"},
                        doc["_meta"]["duration_seconds"])
            logger.info("")
            log_run_line(run, self._runs, doc["_meta"]["duration_seconds"],
                         doc, self._RUN_SUMMARY_KEYS)

        means, variance = tracker.finish(self._runs)
        total_seconds = tracker.total_seconds

        outer_meta = {k: v for k, v in docs[0]["_meta"].items()
                      if k not in ("run", "duration_seconds")}
        outer_meta["runs"] = self._runs
        outer_meta["duration_seconds"] = total_seconds
        return {"_meta": outer_meta, **means, "variance": variance, "runs": docs}

    async def _evaluate_once(self, data: list[dict], announce: bool = True) -> dict:
        """Run the full checker evaluation pipeline once.

        Args:
            data: Pre-loaded list of items (eval dataset with GT triplets).
            announce: Print the Data/Config sections (False for runs 2..N
                in variance mode — they repeat run 1 verbatim).

        Returns:
            Ready-to-write JSON document incl. _meta — accuracy, per-label
            report, confusion matrix. The CLI only resolves paths and dumps.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()

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
        if announce:
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

        # Step 7: Log eval-specific results; the evaluator owns the token
        # table, printed once per invocation — in variance mode it is the
        # cumulative total and belongs after the VARIANCE block.
        self._log_eval_results(result, gt_flat, pred_flat)
        if self._runs <= 1:
            self._log_done(result)
            log_token_stats()

        # Step 8: Assemble the ready-to-write document. The CLI only
        # resolves paths and dumps JSON - it never composes content.
        return self._assemble_document(result)

    def run_sync(self, data: list[dict]) -> dict:
        """Sync wrapper — same pattern as BaseService.run_sync."""
        return asyncio.run(self.evaluate(data))

    def _assemble_document(self, result: CheckerEvalResult) -> dict:
        """Build the output document the CLI writes verbatim."""
        return {
            "_meta": build_meta(
                "checker_eval",
                timestamp=self._started_at,
                duration_seconds=time.perf_counter() - self._started_perf,
                total_items=result.total_items,
                evaluated_items=result.total_items,
                dropped_items=sum(result.skipped.values()),
                request_strategies=GLOBAL_STATS.strategies(),
            ),
            **asdict(result),
        }

    # ── Pipeline steps (private) ─────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1 — fail-fast gate: which items can be evaluated at all.

        An item survives only with all three: gt_key present and non-empty,
        a reference to check against, and at least one triplet carrying a
        human_label. Skip reasons are counted onto self._skip_counts for the
        report; zero survivors is fatal.
        """
        evaluable: list[dict] = []

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

            # 3. Need at least one human_label to compare against
            if not any(t.get("human_label") for t in item[self._gt_key]):
                empty_gt += 1
                continue

            evaluable.append(item)

        self._skip_counts = {
            "missing_gt": missing_gt,
            "missing_context": missing_context,
            "empty_gt": empty_gt,
            "total_triplets": total_triplets,
        }

        if not evaluable:
            total_dropped = missing_gt + missing_context + empty_gt
            raise InvalidInputError(
                f"No evaluable items found. All {total_dropped} items were "
                f"dropped (missing_gt={missing_gt}, missing_context="
                f"{missing_context}, empty_gt={empty_gt})."
            )

        return evaluable

    def _prepare_gt(
        self, data: list[dict]
    ) -> tuple[list[dict], dict[int, dict[int, str]], dict]:
        """Step 2 — extract human labels from the validated items.

        Returns:
            evaluable:     Items that passed _validate.
            gt_labels_map: {evaluable_index: {claim_index: human_label}}.
                           Only triplets WITH human_label are included.
            skip_info:     Counts by skip reason (item-level + claim-level).
        """
        evaluable = self._validate(data)
        gt_labels_map: dict[int, dict[int, str]] = {}

        for idx, item in enumerate(evaluable):
            labels: dict[int, str] = {}
            for j, t in enumerate(item[self._gt_key]):
                label = t.get("human_label")
                if label:
                    labels[j] = label
            gt_labels_map[idx] = labels

        total_labeled = sum(len(l) for l in gt_labels_map.values())

        skip_info = {
            "missing_gt": self._skip_counts["missing_gt"],
            "missing_context": self._skip_counts["missing_context"],
            "empty_gt": self._skip_counts["empty_gt"],
            "unlabeled_claims": self._skip_counts["total_triplets"] - total_labeled,
        }

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

        def label_f1(label: str) -> float | None:
            d = report[label]
            return round(d["f1-score"], 4) if d["support"] else None

        issued = len(gt_flat) + parse_errors
        return CheckerEvalResult(
            accuracy=round(acc, 4),
            total_claims=len(gt_flat),
            total_items=total_items,
            parse_errors=parse_errors,
            report=report,
            confusion_matrix={"labels": LABELS, "matrix": cm},
            skipped=skip_info,
            macro_f1=round(report["macro avg"]["f1-score"], 4),
            checker_failure_rate=(
                round(parse_errors / issued, 4) if issued else None
            ),
            entailment_f1=label_f1("Entailment"),
            contradiction_f1=label_f1("Contradiction"),
            neutral_f1=label_f1("Neutral"),
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
        correct = sum(g == p for g, p in zip(gt_flat, pred_flat))
        wrong = result.total_claims - correct
        labeled = result.total_claims + result.parse_errors
        log_mece_tree(
            "🔎 Verdicts", labeled, "labeled claims",
            [
                ("✅", correct, "correct",
                 "verdict matches the human label"),
                ("❌", wrong, "wrong",
                 "verdict differs — see disagreements file"),
                ("💥", result.parse_errors, "unjudged",
                 "no verdict — see 💥 Reliability"),
            ],
            footer=[("accuracy", correct, result.total_claims,
                     "judged" if result.parse_errors else "")],
        )
        # Majority-class baseline: on an imbalanced slice a constant
        # answer scores this — accuracy below it is worse than a rock.
        if gt_flat:
            counts = {label: gt_flat.count(label) for label in LABELS}
            top_label = max(counts, key=counts.get)
            logger.info(
                "    ℹ️  majority baseline %.3f  (a constant '%s' checker"
                " scores this)",
                counts[top_label] / len(gt_flat), top_label,
            )

        # ── Per-label report (from the stored sklearn dict; the accuracy
        # row stays out — it lives in the tree footer, one home)
        logger.info("")
        logger.info(" 📊 Per-Label Report:")
        col_rule = "    " + "─" * 52
        logger.info("    %-14s%10s%9s%9s%10s",
                    "label", "precision", "recall", "f1", "total")
        logger.info(col_rule)

        def _row(name: str, d: dict) -> None:
            logger.info("    %-14s%10.3f%9.3f%9.3f%10d",
                        name, d["precision"], d["recall"],
                        d["f1-score"], int(d["support"]))

        for label in LABELS:
            _row(label, result.report[label])
        logger.info(col_rule)
        _row("macro avg", result.report["macro avg"])
        _row("weighted avg", result.report["weighted avg"])
        zero_support = [l for l in LABELS
                        if int(result.report[l]["support"]) == 0]
        if zero_support:
            logger.info(
                "    ⚠️  %s: 0 labeled claims — the zeros dilute the"
                " macro avg", ", ".join(zero_support),
            )

        # ── Confusion matrix, with marginals. The grand total is the
        # judged count — it reconciles against the tree's ✅ + ❌.
        logger.info("")
        logger.info(" 📉 Confusion Matrix  (rows = GT, cols = predicted)")
        cm = result.confusion_matrix["matrix"]
        widths = [max(len(label), 6) + 2 for label in LABELS]
        header = "".join(f"{label:>{w}}" for label, w in zip(LABELS, widths))
        logger.info("    %-14s%s%8s", "label", header, "total")
        mat_rule = "    " + "─" * (14 + sum(widths) + 8)
        logger.info(mat_rule)
        col_totals = [0] * len(LABELS)
        for i, label in enumerate(LABELS):
            cells = "".join(f"{int(v):>{w}}" for v, w in zip(cm[i], widths))
            logger.info("    %-14s%s%8d", label, cells, int(sum(cm[i])))
            for j, v in enumerate(cm[i]):
                col_totals[j] += int(v)
        logger.info(mat_rule)
        cells = "".join(f"{v:>{w}}" for v, w in zip(col_totals, widths))
        logger.info("    %-14s%s%8d", "total", cells, sum(col_totals))

        # ── Reliability — the checker under test failing to produce a
        # verdict IS a finding about the checker: excluded from accuracy,
        # charged exactly once, here.
        issued = result.total_claims + result.parse_errors
        logger.info("")
        log_rate_rows(
            "💥 Reliability",
            [("🔎", "Checker (subject)", result.parse_errors, issued,
              "claims unjudged", "checker_failure_rate", None)],
            header_note="subject tooling — excluded from accuracy,"
                        " counted once here",
        )

    def _log_done(self, result: CheckerEvalResult) -> None:
        """Print ✅ Done summary line."""
        total_skipped = sum(result.skipped.values())
        logger.info("")
        logger.info(
            " ✅ Done: %s claims evaluated (%d skipped, "
            "%d unjudged)",
            f"{result.total_claims:,}",
            total_skipped,
            result.parse_errors,
        )
