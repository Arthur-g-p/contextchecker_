"""
Checking service — orchestrates the claim entailment pipeline.

Pipeline steps:
0. Canonicalize: Normalize key aliases (context→reference, query→question).
1. Validation:   Ensures data has 'reference' and non-empty '{model}_response_kg'.
   1.5. Normalize: Convert triplet arrays to canonical {subject, predicate, object} dicts.
   1.6. Context warning: Warn if reference exceeds max_words (advisory in single mode).
2. Filtering:    Skips items already checked or with zero claims (abstentions).
3. Payloading:   Flattens triplets into one CheckingPayload per claim.
4. Execution:    Delegates to the Checker worker (async).
                 Joint mode: groups claims per item, chunks by context budget.
                 Single mode: one LLM call per claim.
5. Serialization: Writes verdicts back into the dicts.
6. Reporting:    Logs validation, skip, config, results, and done line.
"""

import asyncio
import math
from pathlib import Path

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError, FilterError
from contextchecker.models import CheckingPayload
from contextchecker.services.base import BaseService
from contextchecker.workers.checker import Checker, Verdict, _reference_word_count
from contextchecker.stats import GLOBAL_STATS

logger = settings.get_logger(__name__)

# Default maximum claims per joint LLM call.
DEFAULT_JOINT_NUM = 10

# Context budget: if estimated word count exceeds this fraction of
# max_words, we reduce the chunk size. Word count is a proxy for
# tokens since not all endpoints support tokenization.
CONTEXT_BUDGET_RATIO = 0.75

# Rough word limit for a joint prompt. Conservative default that
# works for most models (~8k tokens ≈ 6000 words).
DEFAULT_MAX_WORDS = 6000


# ── Normalization helpers ────────────────────────────────────────────────────

def _normalize_triplets(kg_list: list[dict]) -> None:
    """Normalize triplet format in-place to canonical {subject, predicate, object}.

    Handles the legacy format where triplets are stored as:
        {"triplet": [subject, predicate, object], "human_label": "Entailment"}

    Converts to:
        {"subject": subject, "predicate": predicate, "object": object, "human_label": "Entailment"}

    Already-canonical dicts (with 'subject' key) are left untouched.
    """
    for claim in kg_list:
        if "triplet" in claim and "subject" not in claim:
            t = claim.pop("triplet")
            claim["subject"] = t[0]
            claim["predicate"] = t[1]
            claim["object"] = t[2]


# ── Context budget helpers ───────────────────────────────────────────────────

def _effective_joint_num(
    reference: list[str],
    claims: list[str],
    joint_num: int,
    max_words: int = DEFAULT_MAX_WORDS,
) -> int:
    """Compute the effective chunk size for a joint call.

    If the full prompt (reference + all claims) fits within the budget,
    returns joint_num unchanged. Otherwise reduces it proportionally.

    Args:
        reference: list of reference passages.
        claims: list of claim strings.
        joint_num: user-requested max claims per call.
        max_words: word budget for the entire prompt.

    Returns:
        Effective chunk size (≥ 1).
    """
    ref_words = _reference_word_count(reference)
    budget = int(max_words * CONTEXT_BUDGET_RATIO)

    # Words available for claims after reference + prompt overhead (~100 words)
    available = budget - ref_words - 100
    if available <= 0:
        # Reference alone is too long — still try with 1 claim at a time
        return 1

    # Average words per claim
    if not claims:
        return joint_num
    avg_claim_words = sum(len(c.split()) for c in claims) / len(claims)
    if avg_claim_words <= 0:
        return joint_num

    # How many claims fit in the budget?
    fits = int(available / avg_claim_words)
    fits = max(1, fits)  # at least 1

    return min(joint_num, fits)


class CheckingService(BaseService):
    """Orchestrates checking: validate → filter → flatten → check → serialize.

    Takes extracted data (with {model}_response_kg triplets) and checks
    each claim against the item's reference passage. Writes verdicts
    back as {model}_checker_verdicts.

    Supports two modes:
    - Joint (default): groups claims per item into a single LLM call.
      Controlled by joint_num (max claims per call) and max_words (context budget).
    - Single: one LLM call per claim (--no-joint).

    The service owns all validation and filtering logic. The Checker
    worker is a dumb execution unit that the service delegates to.
    """

    def __init__(
        self,
        model: str,
        extractor_model: str,
        base_url: str | None = None,
        concurrency: int = 10,
        joint: bool = True,
        joint_num: int = DEFAULT_JOINT_NUM,
        max_words: int | None = None,
    ):
        api_key = self._require_api_key(
            settings.CHECKER_API_KEY, "CHECKER_API_KEY"
        )

        self.model = model
        self.base_url = base_url
        self.joint = joint
        self.joint_num = joint_num
        # max_words: default only applies in joint mode
        self.max_words = max_words if max_words is not None else (
            DEFAULT_MAX_WORDS if joint else None
        )
        self._extractor_model = extractor_model
        self._kg_key = f"{extractor_model}_response_kg"
        self._verdict_key = f"{model}_checker_verdicts"
        self._checker = Checker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
        )

    # ── Public API ───────────────────────────────────────────────

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the full checking pipeline.

        Mutates and returns *data* with verdicts written
        into each dict under the ``{model}_checker_verdicts`` key.

        Raises:
            InvalidInputError: No items have extracted triplets.
            FilterError:       All items already have verdicts.
        """
        # Step 0: Normalize key aliases (context→reference, query→question)
        self._canonicalize_keys(data)

        valid = self._validate(data)
        pending, skipped = self._filter(valid)

        self._log_validation(len(data), len(valid), abstained=0)
        self._log_skip(len(valid), skipped, len(pending))
        self._log_config()

        # Execute
        mode = "joint" if self.joint else "single"
        logger.info("── Checking (%s) ─────────────────────────────────────", mode)

        if self.joint:
            verdicts_map = await self._execute_joint(pending)
        else:
            payloads = self._build_payloads(pending)
            flat_verdicts = await self._checker.check_batch(payloads)
            verdicts_map = self._flat_to_map(payloads, flat_verdicts)

        # Serialize verdicts back into dicts
        self._serialize(pending, verdicts_map)

        # TODO: Count results for reporting
        # TODO: self._log_results(...)

        return data

    # ── Pipeline steps (private) ─────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Keep items that have extracted triplets AND a reference.

        Items without the kg_key (not yet extracted) or without 'reference'
        are dropped. Then step 1.5 normalizes triplet format in-place.
        Step 1.6 warns about oversized references.

        Global gate: at least one item must have a non-empty _response_kg.
        """
        valid = []
        for i, item in enumerate(data):
            if self._kg_key not in item:
                logger.warning(
                    "Item %d has no '%s' key — skipping (run extraction first).",
                    i, self._kg_key,
                )
                continue
            if "reference" not in item:
                logger.warning("Item %d has no 'reference' key — skipping.", i)
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                f"No items contain both '{self._kg_key}' and 'reference'."
            )

        # Step 1.5: Normalize triplet format to canonical {subject, predicate, object}
        for item in valid:
            if item[self._kg_key]:
                _normalize_triplets(item[self._kg_key])

        # Global gate: at least one item must have actual claims
        has_claims = any(len(item[self._kg_key]) > 0 for item in valid)
        if not has_claims:
            raise InvalidInputError(
                f"All {len(valid)} items have empty '{self._kg_key}'. "
                "No claims to check."
            )

        # Step 1.6: Context size warning
        self._warn_oversized_references(valid)

        return valid

    def _warn_oversized_references(self, items: list[dict]) -> None:
        """Step 1.6: Warn if any reference exceeds max_words.

        Only fires when max_words is set (always in joint mode,
        opt-in in single mode). Advisory only — does not drop items.
        """
        if self.max_words is None:
            return

        budget = int(self.max_words * CONTEXT_BUDGET_RATIO)
        oversized = []
        for i, item in enumerate(items):
            ref = item["reference"]
            wc = _reference_word_count(ref) if isinstance(ref, list) else len(ref.split())
            if wc > budget:
                oversized.append((i, wc))

        if oversized:
            for idx, wc in oversized:
                logger.warning(
                    "⚠️  Item %d: reference is %d words (budget: %d). "
                    "Context may be too large for reliable results.",
                    idx, wc, budget,
                )

    def _filter(self, valid: list[dict]) -> tuple[list[dict], int]:
        """Step 2: Filter items that already have verdicts or zero claims.

        Returns (pending, skipped_count).
        """
        pending: list[dict] = []
        skipped = 0

        for item in valid:
            if self._verdict_key in item:
                skipped += 1
                continue
            # Skip items with zero claims (abstentions) — nothing to check
            if not item[self._kg_key]:
                skipped += 1
                continue
            pending.append(item)

        if not pending:
            raise FilterError(
                f"All {len(valid)} items already have '{self._verdict_key}' "
                "or have zero claims. Nothing to check."
            )
        return pending, skipped

    def _build_payloads(self, pending: list[dict]) -> list[CheckingPayload]:
        """Step 3 (single mode): Flatten triplets into one payload per claim."""
        payloads = []
        for item_idx, item in enumerate(pending):
            reference = item["reference"]
            for claim_idx, triplet in enumerate(item[self._kg_key]):
                claim_text = self._triplet_to_text(triplet)
                payloads.append(CheckingPayload(
                    claim=claim_text,
                    reference=reference,
                    item_index=item_idx,
                    claim_index=claim_idx,
                ))
        return payloads

    @staticmethod
    def _triplet_to_text(triplet: dict) -> str:
        """Flatten a canonical triplet dict to a claim string."""
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"

    # ── Joint execution ──────────────────────────────────────────

    async def _execute_joint(
        self, pending: list[dict]
    ) -> dict[int, dict[int, Verdict | None]]:
        """Execute checking in joint mode: group claims per item, chunk by budget.

        Returns nested dict: {item_index: {claim_index: Verdict | None}}.
        """
        all_verdicts: dict[int, dict[int, Verdict | None]] = {}

        for item_idx, item in enumerate(pending):
            claims = item[self._kg_key]
            reference = item["reference"]

            # Flatten all claim texts for this item
            claim_texts = [self._triplet_to_text(t) for t in claims]

            # Compute effective chunk size based on context budget
            effective_num = _effective_joint_num(
                reference, claim_texts, self.joint_num, self.max_words,
            )

            # Chunk claims into groups of effective_num
            item_verdicts: dict[int, Verdict | None] = {}
            for chunk_start in range(0, len(claims), effective_num):
                chunk_end = min(chunk_start + effective_num, len(claims))

                # Build numbered claims: (claim_id, claim_text)
                # claim_id is 1-based for the LLM prompt
                numbered = [
                    (claim_idx + 1, claim_texts[claim_idx])
                    for claim_idx in range(chunk_start, chunk_end)
                ]

                # Call worker
                id_verdicts = await self._checker.check_joint(
                    numbered_claims=numbered,
                    reference=reference,
                )

                # Map 1-based claim_ids back to 0-based claim indices
                for claim_idx in range(chunk_start, chunk_end):
                    claim_id = claim_idx + 1  # 1-based
                    item_verdicts[claim_idx] = id_verdicts.get(claim_id)

            all_verdicts[item_idx] = item_verdicts

        return all_verdicts

    @staticmethod
    def _flat_to_map(
        payloads: list[CheckingPayload],
        verdicts: list[Verdict | None],
    ) -> dict[int, dict[int, Verdict | None]]:
        """Convert flat payload+verdict lists to the nested dict format.

        Used by single mode to produce the same shape as joint mode
        so _serialize can handle both.
        """
        result: dict[int, dict[int, Verdict | None]] = {}
        for payload, verdict in zip(payloads, verdicts):
            if payload.item_index not in result:
                result[payload.item_index] = {}
            result[payload.item_index][payload.claim_index] = verdict
        return result

    # ── Serialization ────────────────────────────────────────────

    def _serialize(
        self,
        items: list[dict],
        verdicts_map: dict[int, dict[int, Verdict | None]],
    ) -> None:
        """Step 5: Write verdicts back into the source dicts.

        verdicts_map: {item_index: {claim_index: Verdict | None}}
        """
        for item_idx, item in enumerate(items):
            num_claims = len(item[self._kg_key])
            item_verdicts = verdicts_map.get(item_idx, {})
            item[self._verdict_key] = [
                item_verdicts.get(ci, None)
                for ci in range(num_claims)
            ]
            # Convert Verdict enums to strings for JSON serialization
            item[self._verdict_key] = [
                v.value if isinstance(v, Verdict) else v
                for v in item[self._verdict_key]
            ]

    # ── Logging (service-owned sections) ─────────────────────────

    def _log_validation(self, total: int, valid: int, abstained: int) -> None:
        """Print 📂 Validation section."""
        invalid = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if invalid > 0:
            logger.info("    ├─ dropped:  %d  (missing '%s' or 'reference')", invalid, self._kg_key)
        logger.info("    └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, valid: int, skipped: int, pending: int) -> None:
        """Print 🔄 Skip section. Hidden entirely when nothing was skipped."""
        if skipped == 0:
            return
        logger.info(" 🔄 Skip: items already checked or with zero claims")
        logger.info("    Total:      %d valid items", valid)
        logger.info("    ├─ skipped: %d items", skipped)
        logger.info("    └─ pending: %d items", pending)
        logger.info("")

    def _log_config(self) -> None:
        """Print ⚙️  Config section."""
        location = f"{self.model}"
        if self.base_url:
            location += f" @ {self.base_url}"
        logger.info(" ⚙️  Config")
        logger.info("    Model:       %s", location)
        logger.info("    Extractor:   %s (reading '%s')", self._extractor_model, self._kg_key)
        if self.joint:
            logger.info("    Mode:        joint (max %d claims/call, %d max words)", self.joint_num, self.max_words)
        else:
            if self.max_words:
                logger.info("    Mode:        single (1 claim/call, %d max words)", self.max_words)
            else:
                logger.info("    Mode:        single (1 claim/call)")
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(self, *args, **kwargs) -> None:
        """TODO: Print results block after checking completes."""
        pass

    def _log_bl_results(self, *args, **kwargs) -> None:
        """TODO: Print 📝 Checking Result summary."""
        pass

    def _log_done(self, *args, **kwargs) -> None:
        """TODO: Print ✅ Done summary line."""
        pass
