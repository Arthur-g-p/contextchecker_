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
from contextchecker.utils import deduplicate_triplets
from contextchecker.workers.extractor import Extractor, Triplet
from contextchecker.stats import PhaseStats, log_api_parsing, log_token_stats

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
        verbosity: str = "full",
        section_label: str | None = None,
        dedup: bool = True,
        source_key: str = "response",
        kg_key: str | None = None,
        error_key: str | None = None,
        mark_abstention: bool = True,
    ):
        """Defaults reproduce classic behavior: extract from 'response' into
        {model}_response_kg. Pipelines override source_key/kg_key/error_key to
        run the same service against other fields (e.g. gt_answer).

        mark_abstention: write is_abstention/abstention_source on empty
        results. Disable when the source is not the response (an empty GT
        extraction says nothing about response abstention).
        """
        # Fail-fast: validate config before creating any workers.
        api_key = self._require_api_key(
            settings.EXTRACTOR_API_KEY, "EXTRACTOR_API_KEY"
        )

        self.model = model
        self.base_url = base_url
        self.source_key = source_key
        self._kg_key = kg_key or f"{model}_response_kg"
        # Model-prefixed like the kg key: multi-extractor files must not
        # mix up whose attempt failed.
        self._error_key = error_key or f"{model}_extraction_error"
        self._mark_abstention = mark_abstention
        self._extractor = Extractor(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
            max_retries=max_retries,
        )

        self._init_verbosity(verbosity, section_label)
        self._dedup = dedup

    @property
    def last_stats(self):
        """Read-only view of the worker's last PhaseStats — lets composing
        pipelines report per-phase requests/failures without reaching into
        the worker."""
        return self._extractor.last_stats

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
            if self.verbosity != "silent":
                logger.info(" ✨ All items already processed. Nothing to extract.")
            return data

        self._log_config()

        # Build payloads only for pending items
        payloads = [ExtractionPayload(text=item[self.source_key]) for item in pending]

        # Execute — fatal errors (auth, connection) propagate to CLI
        if self.verbosity != "silent":
            logger.info(settings.section_rule(self.section_label or "Extraction"))
        batch_results = await self._extractor.extract_batch(
            payloads, description=self.section_label,
        )
        phase_stats = self._extractor.last_stats

        # Serialize results back into dicts (dedups when enabled)
        dups_removed = self._serialize(pending, batch_results, phase_stats.error_causes)

        # Abstained items get empty triplets + the explicit abstention flag
        for item in abstained:
            self._clear_markers(item)
            item[self._kg_key] = []
            if self._mark_abstention:
                item["is_abstention"] = True
                item["abstention_source"] = "heuristic"

        # Log results using PhaseStats from the worker
        self._log_results(
            pending_count=len(pending),
            abstained_count=len(abstained),
            total=len(data),
            total_claims=phase_stats.total_items,
            successful=phase_stats.success,
            failed=phase_stats.total_errors,
            skipped=skipped,
            phase_stats=phase_stats,
            dups_removed=dups_removed,
        )

        return data

    # run_sync() inherited from BaseService

    # ── Pipeline steps (private) ─────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing the source key (default 'response')."""
        valid = []
        for i, item in enumerate(data):
            if self.source_key not in item:
                logger.debug("Item %d has no '%s' key — skipping.", i, self.source_key)
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(f"No items contain a '{self.source_key}' key.")
        return valid

    def _filter(self, valid: list[dict]) -> tuple[list[dict], list[dict], int]:
        """Step 2+3: Filter already-processed and pre-filter abstentions.

        Returns (pending, abstained, skipped_count).
        """
        pending: list[dict] = []
        abstained: list[dict] = []
        skipped = 0

        for item in valid:
            # An error marker is not a result: errored items re-enter the
            # pending pool on re-runs instead of being skipped forever.
            if self._kg_key in item and self._error_key not in item:
                skipped += 1
                continue  # already processed
            if _is_full_abstention(item[self.source_key]):
                abstained.append(item)
            else:
                pending.append(item)

        return pending, abstained, skipped

    def _clear_markers(self, item: dict) -> None:
        """Drop stale outcome markers before writing a fresh result.

        Matters on re-runs: a previously errored item that now succeeds must
        not keep its old error/abstention keys. A service that is not allowed
        to WRITE the abstention flags (mark_abstention=False, e.g. gt_answer
        extraction) must not DELETE them either — they describe the response
        and belong to the response extraction alone."""
        item.pop(self._error_key, None)
        if self._mark_abstention:
            item.pop("is_abstention", None)
            item.pop("abstention_source", None)

    def _serialize(
        self,
        items: list[dict],
        results: list[list[Triplet]],
        error_causes: dict[int, str] | None = None,
    ) -> int:
        """Step 6: Write triplets + outcome markers back into the source dicts.

        Every empty result is disambiguated on disk — [] alone is never
        left open to interpretation:
        - failed items:  kg=[] plus ``{model}_extraction_error`` (cause)
        - empty, no error: kg=[] plus ``is_abstention: true`` (the model
          asserted nothing — per project semantics that IS an abstention,
          justified or not)

        When ``dedup`` is enabled (default), exact (s, p, o) duplicates are
        dropped before writing — a loss-free cleanup of the extractor's output.
        Returns the total number of duplicate triplets removed.
        """
        error_causes = error_causes or {}
        dups_removed = 0
        for i, (item, triplets) in enumerate(zip(items, results)):
            self._clear_markers(item)

            if i in error_causes:
                item[self._kg_key] = []
                item[self._error_key] = error_causes[i]
                continue

            claims = [t.model_dump() for t in triplets]
            if self._dedup:
                unique = deduplicate_triplets(claims)
                dups_removed += len(claims) - len(unique)
                claims = unique
            item[self._kg_key] = claims

            if not claims and self._mark_abstention:
                item["is_abstention"] = True
                item["abstention_source"] = "llm"
        return dups_removed

    # ── Logging ─────────────────────────────────────────────────

    def _log_validation(self, total: int, valid: int, abstained: int) -> None:
        """Print 📂 Validation section (full only — pipelines own the preamble)."""
        if self.verbosity != "full":
            return
        invalid = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if invalid > 0:
            logger.info("    ├─ dropped:  %d  (no '%s' key)", invalid, self.source_key)
        if abstained > 0:
            logger.info("    ├─ abstain:  %d  (heuristic: empty/refusal response)", abstained)
        logger.info("    └─ valid:  %d items", valid - abstained)
        logger.info("")

    def _log_skip(self, valid: int, skipped: int, pending: int) -> None:
        """Print 🔄 Skip section. Hidden entirely when nothing was skipped."""
        if self.verbosity != "full":
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
        if self.verbosity != "full":
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
        phase_stats: PhaseStats,
        dups_removed: int = 0,
    ) -> None:
        """Print the full results block: API summary, BL results, tokens, done line.

        compact: API + BL only (pipeline owns tokens + done);
        silent: nothing."""
        if self.verbosity == "silent":
            return
        logger.info("")
        if self.verbosity == "full":
            logger.info(settings.section_rule("EXTRACTOR RESULTS", char="═"))
            logger.info("")
        log_api_parsing(pending_count, phase_stats)
        worker_empty = phase_stats.empty if phase_stats else 0
        total_output = successful + worker_empty + abstained_count
        # Claims actually stored = raw extracted minus duplicates dropped.
        unique_claims = total_claims - dups_removed
        self._log_bl_results(
            total_output, unique_claims, successful,
            worker_empty + abstained_count, dups_removed,
        )
        if self.verbosity == "full":
            log_token_stats()
        self._log_done(total, unique_claims, abstained_count, skipped, dups_removed)

    def _log_bl_results(
        self, valid_items: int, claims: int, with_claims: int, empty_count: int,
        dups_removed: int = 0,
    ) -> None:
        """Print 📝 Extraction summary.

        Shows items with output: how many generated claims vs empty, and any
        exact duplicates removed by the dedup pass.
        """
        logger.info(" 📝 Extraction:")
        logger.info("    %d items with output", valid_items)
        # Sub-lines after the 💎 line; the last one gets the └─ connector.
        tail = []
        if dups_removed > 0:
            tail.append(("🔁", f"{dups_removed} exact duplicates removed"))
        if empty_count > 0:
            tail.append(("⚪", f"{empty_count} abstentions → 0 claims"))
        prefix = "├─" if tail else "└─"
        logger.info("     %s 💎 %d items → %d claims", prefix, with_claims, claims)
        for i, (icon, text) in enumerate(tail):
            prefix = "└─" if i == len(tail) - 1 else "├─"
            logger.info("     %s %s %s", prefix, icon, text)
        logger.info("")

    def _log_done(
        self, total: int, claims: int, abstentions: int, skipped: int,
        dups_removed: int = 0,
    ) -> None:
        """Print ✅ Done summary line (full only — pipelines own the done line)."""
        if self.verbosity != "full":
            return
        parts = [f"{claims} claims"]
        if dups_removed > 0:
            parts.append(f"{dups_removed} dups removed")
        if abstentions > 0:
            parts.append(f"{abstentions} abstentions")
        if skipped > 0:
            parts.append(f"{skipped} skipped")
        logger.info(
            " ✅ Done: %d items extracted → %s", total, ", ".join(parts)
        )

    # TODO: _run_validation pass (optional, percentage-based)
    # TODO: _run_add_missing pass (optional, percentage-based)
