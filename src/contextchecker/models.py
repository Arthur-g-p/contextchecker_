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


# ── Evaluation ───────────────────────────────────────────────────────────────
