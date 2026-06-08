"""
Extraction service — orchestrates the extract pipeline.

Pipeline steps:
0. Canonicalize: Normalize key aliases (context→reference, query→question).
1. Validation: Ensures API key is configured and input data has 'response' keys.
2. Filtering:  Skips items that already contain a {model}_response_kg key.
3. Pre-filter: Detects full abstentions to avoid wasting LLM calls.
4. Payloading: Builds ExtractionPayload list for the worker.
5. Execution:  Delegates to the Extractor worker (async).
6. Serialization: Writes extracted triplets back into the dicts.
7. Reporting:  Logs validation, skip, config, results, and done line.
"""


from pathlib import Path

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError, FilterError
from contextchecker.services.base import BaseService
from contextchecker.models import ExtractionPayload
from contextchecker.workers.extractor import Extractor, Triplet
from contextchecker.stats import log_api_parsing, log_token_stats

logger = settings.get_logger(__name__)


# ── Pre-filter helpers ───────────────────────────────────────────────────────

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_REMOVE_PUNCTUATION = str.maketrans("", "", PUNCTUATION)

REFUSAL_PHRASES = (
    "i dont know",
    "i cannot answer",
    "not provided in the context",
    "i dont have enough information",
    "information not provided",
)


def _is_full_abstention(text: str, threshold: float = 0.85) -> bool:
    """Detect responses that are pure refusals / abstentions.

    Returns True when the response is empty or consists almost entirely
    of a known refusal phrase (measured by character coverage).  These
    items produce zero triplets by definition, so we skip the LLM call.
    """
    if not text or not text.strip():
        return True
    clean = text.lower().translate(_REMOVE_PUNCTUATION)
    clean = " ".join(clean.split())
    if not clean:
        return True
    for phrase in REFUSAL_PHRASES:
        if phrase in clean and (len(phrase) / len(clean)) >= threshold:
            return True
    return False


# ── Service ──────────────────────────────────────────────────────────────────

class ExtractionService(BaseService):
    """Orchestrates extraction: validate → filter → extract → serialize.

    The service owns all validation and filtering logic.  The Extractor
    worker is a dumb execution unit that the service delegates to.

    Logging: This service logs validation, skip, config, results, and
    the done line via logger.info(). The CLI prints the box header and
    output path. The worker/llmclient log extraction progress.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        concurrency: int = 10,
        max_retries: int | None = None,
        quiet: bool = False,
    ):
        # Fail-fast: validate config before creating any workers.
        api_key = self._require_api_key(
            settings.EXTRACTOR_API_KEY, "EXTRACTOR_API_KEY"
        )

        self.model = model
        self.base_url = base_url
        self._kg_key = f"{model}_response_kg"
        self._extractor = Extractor(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
            max_retries=max_retries,
        )

        self.quiet = quiet
    # ── Public API ───────────────────────────────────────────────

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the full extraction pipeline.

        Mutates and returns *data* with extracted triplets written
        into each dict under the ``{model}_response_kg`` key.

        Raises:
            InvalidInputError: No items contain a 'response' key.
            FilterError:       All items are already processed.
        """
        # Step 0: Normalize key aliases (context→reference, query→question)
        self._canonicalize_keys(data)

        valid = self._validate(data)
        pending, abstained, skipped = self._filter(valid)

        self._log_validation(len(data), len(valid), len(abstained))
        self._log_skip(len(valid), skipped, len(pending))

        if not pending and not abstained:
            logger.info(" ✨ All items already processed. Nothing to extract.")
            return data

        self._log_config()

        # Build payloads only for pending items
        payloads = [ExtractionPayload(text=item["response"]) for item in pending]

        # Execute — fatal errors (auth, connection) propagate to CLI
        logger.info("── Extraction ────────────────────────────────────────")
        batch_results = await self._extractor.extract_batch(payloads)
        phase_stats = self._extractor.last_stats

        # Serialize results back into dicts
        self._serialize(pending, batch_results)

        # Abstained items get empty triplets
        for item in abstained:
            item[self._kg_key] = []

        # Log results using PhaseStats from the worker
        self._log_results(
            pending_count=len(pending),
            abstained_count=len(abstained),
            total=len(data),
            total_claims=phase_stats.total_items,
            successful=phase_stats.success,
            failed=phase_stats.total_errors,
            skipped=skipped,
        )

        return data

    # run_sync() inherited from BaseService

    # ── Pipeline steps (private) ─────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing 'response' key."""
        valid = []
        for i, item in enumerate(data):
            if "response" not in item:
                logger.debug("Item %d has no 'response' key — skipping.", i)
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError("No items contain a 'response' key.")
        return valid

    def _filter(self, valid: list[dict]) -> tuple[list[dict], list[dict], int]:
        """Step 2+3: Filter already-processed and pre-filter abstentions.

        Returns (pending, abstained, skipped_count).
        """
        pending: list[dict] = []
        abstained: list[dict] = []
        skipped = 0

        for item in valid:
            if self._kg_key in item:
                skipped += 1
                continue  # already processed
            if _is_full_abstention(item["response"]):
                abstained.append(item)
            else:
                pending.append(item)

        return pending, abstained, skipped

    def _serialize(
        self,
        items: list[dict],
        results: list[list[Triplet]],
    ) -> None:
        """Step 6: Write triplets back into the source dicts."""
        for item, triplets in zip(items, results):
            item[self._kg_key] = [t.model_dump() for t in triplets]

    # ── Logging ─────────────────────────────────────────────────

    def _log_validation(self, total: int, valid: int, abstained: int) -> None:
        """Print 📂 Validation section."""
        if self.quiet:
            return
        invalid = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if invalid > 0:
            logger.info("    ├─ dropped:  %d  (no 'response' key)", invalid)
        if abstained > 0:
            logger.info("    ├─ abstain:  %d  (empty response)", abstained)
        logger.info("    └─ valid:  %d items", valid - abstained)
        logger.info("")

    def _log_skip(self, valid: int, skipped: int, pending: int) -> None:
        """Print 🔄 Skip section. Hidden entirely when nothing was skipped."""
        if self.quiet:
            return
        if skipped == 0:
            return
        logger.info(" 🔄 Skip: items with existing extraction (≥1 claim)")
        logger.info("    Total:      %d valid items", valid)
        logger.info("    ├─ skipped: %d items (already extracted)", skipped)
        logger.info("    └─ pending: %d items", pending)
        logger.info("")

    def _log_config(self) -> None:
        """Print ⚙️  Config section."""
        if self.quiet:
            return
        location = f"{self.model}"
        if self.base_url:
            location += f" @ {self.base_url}"
        logger.info(" ⚙️  Config")
        logger.info("    Model:       %s", location)
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(
        self,
        pending_count: int,
        abstained_count: int,
        total: int,
        total_claims: int,
        successful: int,
        failed: int,
        skipped: int,
    ) -> None:
        """Print the full results block: API summary, BL results, tokens, done line."""
        phase_stats = self._extractor.last_stats

        logger.info("")
        logger.info("── EXTRACTOR RESULTS ────────────────────────────────────────")
        logger.info("")
        log_api_parsing(pending_count, phase_stats)
        worker_empty = phase_stats.empty if phase_stats else 0
        total_output = successful + worker_empty + abstained_count
        self._log_bl_results(total_output, total_claims, successful, worker_empty + abstained_count)
        log_token_stats()
        self._log_done(total, total_claims, abstained_count, skipped)

    def _log_bl_results(
        self, valid_items: int, claims: int, with_claims: int, empty_count: int
    ) -> None:
        """Print 📝 Extraction summary.

        Shows items with output: how many generated claims vs empty.
        """
        logger.info(" 📝 Extraction:")
        logger.info("    %d items with output", valid_items)
        has_empty = empty_count > 0
        # Items with claims
        prefix = "├─" if has_empty else "└─"
        logger.info("     %s 💎 %d items → %d triplets", prefix, with_claims, claims)
        # Empty items (abstentions / model returned 0 claims)
        if has_empty:
            logger.info("     └─ ⚪ %d items → 0 claims (empty)", empty_count)
        logger.info("")

    def _log_done(
        self, total: int, claims: int, abstentions: int, skipped: int
    ) -> None:
        """Print ✅ Done summary line."""
        if self.quiet:
            return
        parts = [f"{claims} claims"]
        if abstentions > 0:
            parts.append(f"{abstentions} abstentions")
        if skipped > 0:
            parts.append(f"{skipped} skipped")
        logger.info(
            " ✅ Done: %d items extracted → %s", total, ", ".join(parts)
        )

    # TODO: _run_validation pass (optional, percentage-based)
    # TODO: _run_add_missing pass (optional, percentage-based)
