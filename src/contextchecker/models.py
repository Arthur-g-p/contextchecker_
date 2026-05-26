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

