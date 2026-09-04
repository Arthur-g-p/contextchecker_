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
                 Joint mode: pre-computes all chunks, sends via check_joint_batch.
                 Single mode: one LLM call per claim via check_batch.
5. Serialization: Writes verdict + explanation into each triplet dict.
6. Reporting:    Logs validation, skip, config, results, and done line.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from contextchecker import settings
from contextchecker.settings import DEFAULT_MAX_WORDS
from contextchecker.exceptions import InvalidInputError, FilterError
from contextchecker.models import CheckingPayload
from contextchecker.utils import canonicalize_triplets, plural
from contextchecker.services.base import BaseService
from contextchecker.workers.checker import (
    Checker, ClaimVerdict, _reference_word_count,
)
from contextchecker.stats import GLOBAL_STATS, log_api_parsing, log_mece_tree, log_token_stats

logger = settings.get_logger(__name__)

# Context budget: if estimated word count exceeds this fraction of
# max_words, we reduce the chunk size. Word count is a proxy for
# tokens since not all endpoints support tokenization.
CONTEXT_BUDGET_RATIO = 0.75


def mode_label(joint: bool, joint_num: int, max_words: int | None) -> str:
    """The Config "Mode:" string, resolved the way the service resolves it
    (joint mode falls back to DEFAULT_MAX_WORDS)."""
    if joint:
        words = max_words if max_words is not None else settings.DEFAULT_MAX_WORDS
        return f"joint (max {joint_num} claims/call, {words} max words)"
    if max_words:
        return f"single (1 claim/call, {max_words} max words)"
    return "single (1 claim/call)"


# ── Context budget helpers ───────────────────────────────────────────────────

def _effective_joint_num(
    reference: list[str],
    claims: list[str],
    joint_num: int,
    max_words: int = settings.DEFAULT_MAX_WORDS,
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


# ── Joint chunk metadata ────────────────────────────────────────────────────

@dataclass
class _JointChunk:
    """Internal bookkeeping for a joint LLM call.

    Maps a chunk of numbered claims back to its source item and
    claim indices so we can write verdicts to the correct triplets.
    """
    numbered_claims: list[tuple[int, str]]   # (local id 1..n, claim text)
    reference: list[str]
    item_index: int
    # Item-level index per local id. Not derivable by offset: already-checked
    # claims are skipped, so the indices are not contiguous.
    orig_indices: list[int]
    extra_vars: dict | None = None  # additional template variables (e.g. {{response}})


class CheckingService(BaseService):
    """Orchestrates checking: validate → filter → flatten → check → serialize.

    Takes extracted data (with {model}_response_kg triplets) and checks
    each claim against the item's reference passage. Writes verdict +
    explanation directly into each triplet dict.

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
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        verbosity: str = "full",
        section_label: str | None = None,
        joint_prompt_key: str | None = None,
        kg_key: str | None = None,
        verdict_namespace: str | None = None,
        extraction_error_key: str | None = None,
    ):
        """Defaults reproduce classic behavior: read {extractor_model}_response_kg,
        write {model}_checker_verdict/_explanation/_error. Pipelines override
        kg_key + verdict_namespace so multiple checking directions over the
        same triplets never collide (e.g. namespace "{model}_answer2response").
        """
        api_key = self._require_api_key(
            settings.CHECKER_API_KEY, "CHECKER_API_KEY"
        )

        self.model = model
        self.base_url = base_url
        self.joint = joint
        self.joint_num = joint_num
        self._init_verbosity(verbosity, section_label)
        # max_words: default only applies in joint mode
        self.max_words = max_words if max_words is not None else (
            settings.DEFAULT_MAX_WORDS if joint else None
        )
        self._extractor_model = extractor_model
        self._kg_key = kg_key or f"{extractor_model}_response_kg"
        self._extraction_error_key = (
            extraction_error_key or f"{extractor_model}_extraction_error"
        )
        namespace = verdict_namespace or f"{model}_checker"
        self._verdict_key = f"{namespace}_verdict"
        self._explanation_key = f"{namespace}_explanation"
        self._checker_error_key = f"{namespace}_error"
        self._checker = Checker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
            joint_prompt_key=joint_prompt_key,
        )

    # ── Output contract (read by pipelines/direction runner) ─────

    @property
    def kg_key(self) -> str:
        return self._kg_key

    @property
    def verdict_key(self) -> str:
        return self._verdict_key

    @property
    def explanation_key(self) -> str:
        return self._explanation_key

    @property
    def checker_error_key(self) -> str:
        return self._checker_error_key

    @property
    def extraction_error_key(self) -> str:
        return self._extraction_error_key

    @property
    def mode_label(self) -> str:
        """Human-readable execution mode for Config blocks (service, evals,
        pipelines print the same string)."""
        return mode_label(self.joint, self.joint_num, self.max_words)

    @property
    def last_stats(self):
        """Read-only view of the worker's last PhaseStats — lets composing
        pipelines report per-phase requests/failures without reaching into
        the worker."""
        return self._checker.last_stats

    # ── Public API ───────────────────────────────────────────────

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the full checking pipeline.

        Mutates and returns *data* with verdicts + explanations written
        into each triplet dict.

        Raises:
            InvalidInputError: No items have extracted triplets.
            FilterError:       All items already have verdicts.
        """
        # Step 0: Normalize key aliases (context→reference, query→question)
        self._canonicalize_keys(data)

        valid = self._validate(data)

        pending, skip_stats = self._filter(valid)

        if self.verbosity == "full":
            self._log_validation(len(data), len(valid), abstained=0)
            self._log_skip(len(valid), skip_stats, len(pending))
            self._log_config()

        # Execute
        mode = "joint" if self.joint else "single"
        if self.verbosity != "silent":
            logger.info(settings.section_rule(
                self.section_label or f"Checking ({mode})"))

        if self.joint:
            verdicts_map = await self._execute_joint(pending)
        else:
            payloads = self._build_payloads(pending)
            flat_verdicts = await self._checker.check_batch(
                payloads, description=self.section_label,
            )
            verdicts_map = self._flat_to_map(payloads, flat_verdicts)

        # Serialize verdicts + explanations into each triplet dict
        self._serialize(pending, verdicts_map)

        # Count results for reporting (only from the active execution)
        total_triplets = 0
        entailment = 0
        contradiction = 0
        neutral = 0
        unjudged = 0

        for item_idx, claim_verdicts in verdicts_map.items():
            for claim_idx, cv in claim_verdicts.items():
                verdict_val = cv.verdict.value if cv.verdict else None
                if verdict_val is None:
                    unjudged += 1
                else:
                    total_triplets += 1
                    if verdict_val == "Entailment":
                        entailment += 1
                    elif verdict_val == "Contradiction":
                        contradiction += 1
                    elif verdict_val == "Neutral":
                        neutral += 1

        skipped = (
            skip_stats["already_checked"]
            + skip_stats["abstained"]
            + skip_stats["extraction_failed"]
        )

        self._log_results(
            unjudged=unjudged,
            total=len(pending),
            total_triplets=total_triplets,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            skipped=skipped,
        )

        return data

    def _is_claim_checked(self, item: dict, claim_idx: int) -> bool:
        """A claim counts as checked when its triplet carries a non-None
        verdict for this checker model."""
        return item[self._kg_key][claim_idx].get(self._verdict_key) is not None

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

        for item in valid:
            if item[self._kg_key]:
                canonicalize_triplets(item[self._kg_key])

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

    def _filter(self, valid: list[dict]) -> tuple[list[dict], dict]:
        """Step 2: Filter items that already have verdicts or zero claims.

        Returns (pending, skip_stats) where skip_stats breaks down
        the reasons items were skipped:
            already_checked:   item already has verdict for this checker model
            abstained:         kg is empty without an extraction error — per
                               project semantics an abstention (justified or not)
            extraction_failed: kg is empty because extraction errored
                               ({extractor}_extraction_error present)
        """
        pending: list[dict] = []
        already_checked = 0
        abstained = 0
        extraction_failed = 0

        for item in valid:
            # Empty claims — nothing to check; the marker tells us why
            if not item[self._kg_key]:
                if self._extraction_error_key in item:
                    extraction_failed += 1
                else:
                    abstained += 1
                continue

            # Item is skipped if all of its claims are already checked
            has_unchecked = any(not self._is_claim_checked(item, claim_idx) for claim_idx in range(len(item[self._kg_key])))
            if not has_unchecked:
                already_checked += 1
                continue

            pending.append(item)

        if not pending:
            raise FilterError(
                f"All {len(valid)} items already checked or have zero claims. "
                "Nothing to check."
            )
        return pending, {
            "already_checked": already_checked,
            "abstained": abstained,
            "extraction_failed": extraction_failed,
        }

    def _build_payloads(self, pending: list[dict]) -> list[CheckingPayload]:
        """Step 3 (single mode): Flatten triplets into one payload per claim."""
        payloads = []
        for item_idx, item in enumerate(pending):
            reference = item["reference"]
            for claim_idx, triplet in enumerate(item[self._kg_key]):
                if self._is_claim_checked(item, claim_idx):
                    continue
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

    async def _execute_joint(
        self, pending: list[dict]
    ) -> dict[int, dict[int, ClaimVerdict]]:
        """Execute checking in joint mode.

        Pre-computes ALL chunks across all items, sends them as a single
        batch through check_joint_batch (→ tqdm progress bar + concurrency).
        Maps results back to {item_index: {claim_index: ClaimVerdict}}.
        """
        # Pre-compute all chunks
        chunks: list[_JointChunk] = []
        for item_idx, item in enumerate(pending):
            claims = item[self._kg_key]
            reference = item["reference"]

            # Filter unchecked claims and keep their original 0-based indices
            unchecked = [
                (claim_idx, self._triplet_to_text(t))
                for claim_idx, t in enumerate(claims)
                if not self._is_claim_checked(item, claim_idx)
            ]

            if not unchecked:
                continue

            unchecked_texts = [text for _, text in unchecked]

            # Compute effective chunk size based on context budget
            effective_num = _effective_joint_num(
                reference, unchecked_texts, self.joint_num, self.max_words,
            )

            for chunk_start in range(0, len(unchecked), effective_num):
                chunk_end = min(chunk_start + effective_num, len(unchecked))
                slice_ = unchecked[chunk_start:chunk_end]
                numbered = [
                    (local_id, text)
                    for local_id, (_, text) in enumerate(slice_, start=1)
                ]
                orig_indices = [orig_idx for orig_idx, _ in slice_]
                # Only the eval matching prompt declares {{response}}.
                # Only the eval matching prompt declares {{response}}; templates
                # without it ignore the key.
                ev = {"response": item.get("response") or "No response text available"}
                chunks.append(_JointChunk(
                    numbered_claims=numbered,
                    reference=reference,
                    item_index=item_idx,
                    orig_indices=orig_indices,
                    extra_vars=ev or None,
                ))

        # Send all chunks as a single batch (progress bar + concurrency)
        batch_input = [
            (c.numbered_claims, c.reference) for c in chunks
        ]
        extra_vars_list = [c.extra_vars for c in chunks]
        batch_results = await self._checker.check_joint_batch(
            batch_input, extra_vars_list=extra_vars_list,
            description=(
                f"{self.section_label} (joint)" if self.section_label else None
            ),
        )

        # Map results back: {item_index: {claim_index: ClaimVerdict}}
        all_verdicts: dict[int, dict[int, ClaimVerdict]] = {}
        for chunk_meta, id_verdicts in zip(chunks, batch_results):
            item_idx = chunk_meta.item_index
            if item_idx not in all_verdicts:
                all_verdicts[item_idx] = {}

            for local_id, orig_idx in enumerate(chunk_meta.orig_indices, start=1):
                all_verdicts[item_idx][orig_idx] = id_verdicts.get(
                    local_id, ClaimVerdict(verdict=None)
                )

        return all_verdicts

    @staticmethod
    def _flat_to_map(
        payloads: list[CheckingPayload],
        verdicts: list[ClaimVerdict],
    ) -> dict[int, dict[int, ClaimVerdict]]:
        """Convert flat payload+verdict lists to the nested dict format.

        Used by single mode to produce the same shape as joint mode
        so _serialize can handle both.
        """
        result: dict[int, dict[int, ClaimVerdict]] = {}
        for payload, verdict in zip(payloads, verdicts):
            if payload.item_index not in result:
                result[payload.item_index] = {}
            result[payload.item_index][payload.claim_index] = verdict
        return result

    # ── Serialization ────────────────────────────────────────────

    def _serialize(
        self,
        items: list[dict],
        verdicts_map: dict[int, dict[int, ClaimVerdict]],
    ) -> None:
        """Step 5: Write verdict + explanation into each triplet dict.

        Each triplet gets:
        - ``{model}_checker_verdict``: "Entailment" | "Contradiction" | "Neutral" | None
        - ``{model}_checker_explanation``: chain-of-thought reasoning | None

        Triplets absent from verdicts_map (already checked in a prior run)
        are left untouched.
        """
        for item_idx, item in enumerate(items):
            item_verdicts = verdicts_map.get(item_idx, {})
            for claim_idx, triplet in enumerate(item[self._kg_key]):
                if claim_idx not in item_verdicts:
                    continue
                cv = item_verdicts[claim_idx]
                triplet.pop(self._checker_error_key, None)
                triplet[self._verdict_key] = cv.verdict.value if cv.verdict else None
                triplet[self._explanation_key] = cv.explanation
                # Null verdict is never left uninterpretable: persist WHY
                if cv.verdict is None and cv.error:
                    triplet[self._checker_error_key] = cv.error

    # ── Logging (service-owned sections) ─────────────────────────

    def _log_validation(self, total: int, valid: int, abstained: int) -> None:
        """Print 📂 Validation section (full only)."""
        if self.verbosity != "full":
            return
        invalid = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:        %d items", total)
        if invalid > 0:
            logger.info("     ├─ dropped:  %d  (missing '%s' or 'reference')", invalid, self._kg_key)
        logger.info("     └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, valid: int, skip_stats: dict, pending: int) -> None:
        """Print 🔄 Skip section with breakdown. Hidden when nothing was skipped."""
        if self.verbosity != "full":
            return
        already = skip_stats["already_checked"]
        abstained = skip_stats["abstained"]
        failed = skip_stats["extraction_failed"]
        skipped = already + abstained + failed

        if skipped == 0:
            return

        logger.info(" 🔄 Skip")
        logger.info("    %-24s%d valid items", "Total:", valid)
        if already > 0:
            logger.info("     ├─ %-20s%d  (verdict exists for this model)", "already checked:", already)
        if abstained > 0:
            logger.info("     ├─ %-20s%d  (empty claims, no error)", "abstained:", abstained)
        if failed > 0:
            logger.info("     ├─ %-20s%d  ('%s' present)", "extraction failed:", failed, self._extraction_error_key)
        logger.info("     └─ %-20s%d items", "pending:", pending)
        logger.info("")

    def _log_config(self) -> None:
        """Print ⚙️  Config section."""
        if self.verbosity != "full":
            return
        location = f"{self.model}"
        if self.base_url:
            location += f" @ {self.base_url}"
        logger.info(" ⚙️  Config")
        logger.info("    Model:       %s", location)
        logger.info("    Extractor:   %s (reading '%s')", self._extractor_model, self._kg_key)
        logger.info("    Mode:        %s", self.mode_label)
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(
        self,
        unjudged: int,
        total: int,
        total_triplets: int,
        entailment: int,
        contradiction: int,
        neutral: int,
        skipped: int,
    ) -> None:
        """Print the full results block: API summary, BL results, tokens, done line.

        compact: API + BL only (pipeline owns tokens + done);
        silent: nothing."""
        if self.verbosity == "silent":
            return
        phase_stats = self._checker.last_stats

        logger.info("")
        if self.verbosity == "full":
            logger.info(settings.section_rule("CHECKER RESULTS", char="═"))
            logger.info("")
        if phase_stats:
            log_api_parsing(phase_stats.first_pass_count, phase_stats)

        self._log_bl_results(total_triplets, unjudged, entailment, contradiction, neutral)
        self._log_done(total, total_triplets, skipped)
        if self.verbosity == "full":
            log_token_stats()

    def _log_bl_results(
        self, judged: int, unjudged: int, entailment: int, contradiction: int, neutral: int
    ) -> None:
        """Print the 🔎 Checking tree: every issued claim lands in exactly
        one branch, null verdicts included."""
        log_mece_tree(
            "🔎 Checking", judged + unjudged, "claims",
            [
                ("🟢", entailment, "Entailment", None),
                ("🔴", contradiction, "Contradiction", None),
                ("⚪", neutral, "Neutral", None),
                ("💥", unjudged, "unjudged", "no verdict — see 🌐 API & Parsing"),
            ],
        )
        logger.info("")

    def _log_done(
        self, total: int, total_triplets: int, skipped: int
    ) -> None:
        """Print ✅ Done summary line (full only)."""
        if self.verbosity != "full":
            return
        parts = [f"{total} {plural(total, 'item')}",
                 f"{total_triplets} {plural(total_triplets, 'claim')}"]
        if skipped > 0:
            parts.append(f"{skipped} {plural(skipped, 'item')} skipped")
        logger.info(" ✅ Done: %s", " · ".join(parts))
