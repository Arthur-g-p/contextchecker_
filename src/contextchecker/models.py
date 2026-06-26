"""
Dataclass payloads — the typed contracts between architectural layers.

Every payload flowing from CLI → Service → Worker is defined here.
This file is a leaf dependency — it imports nothing from contextchecker.
"""

from dataclasses import dataclass, field


# ── Extraction ───────────────────────────────────────────────────────────────

@dataclass
class ExtractionPayload:
    """Contract: CLI → Extraction Service."""

    text: str
    require_ground_truth: bool = False


# ── Checking ─────────────────────────────────────────────────────────────────

@dataclass
class CheckingPayload:
    """Contract: CheckingService → Checker worker.

    One payload per claim. The worker checks if this claim
    is entailed by the reference (Entailment / Contradiction / Neutral).
    """

    claim: str              # flattened triplet: "subject predicate object"
    reference: list[str]    # list of reference passages to check against
    item_index: int         # which item in the dataset this claim belongs to
    claim_index: int        # which claim within that item


# ── Evaluation ───────────────────────────────────────────────────────────────

@dataclass
class CheckerEvalResult:
    """Result of a checker evaluation run.

    Returned by CheckerEvaluator.evaluate() and serialized to JSON by CLI.
    The report dict mirrors sklearn's classification_report(output_dict=True).
    """

    accuracy: float                        # overall fraction correct
    total_claims: int                      # claims actually compared (excl. parse errors)
    total_items: int                       # items evaluated
    parse_errors: int                      # claims with None verdict (excluded from metrics)
    report: dict                           # per-label P/R/F1 + macro avg
    confusion_matrix: dict                 # {"labels": [...], "matrix": [[...]]}
    skipped: dict                          # {"missing_gt": N, "missing_context": N, "empty_gt": N}


@dataclass
class ExtractorEvalResult:
    """Result of an extractor evaluation run.

    Returned by ExtractorEvaluator.evaluate() alongside a disagreement list.
    Uses IR metrics (Precision/Recall/F1) rather than classification metrics.
    """

    precision: float
    recall: float
    f1: float
    tp_recall: int                         # GT triplets entailed by predictions (recall numerator)
    tp_precision: int                      # pred triplets entailed by GT (precision numerator)
    fp: int                                # predictions not supported by GT
    fn: int                                # GT triplets not covered by predictions
    total_items: int                       # items in dataset
    to_compare_items: int                  # items with GT + predictions to compare
    gt_stats: dict                         # {"total_triplets": N, "avg_per_item": float}
    pred_stats: dict                       # {"total_triplets": N, "avg_per_item": float}
    abstention_errors: dict                # {"wrongful_answer": N, "wrongful_abstention": N,
                                           #  "wrongful_abstention_fn_penalty": N,
                                           #  "wrongful_answer_fp_penalty": N}
    correct_abstention: int                # items with neither GT nor predictions
    method: str                            # "llm"

