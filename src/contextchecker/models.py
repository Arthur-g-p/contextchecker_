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

