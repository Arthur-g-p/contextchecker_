"""
Atomization service — orchestrates the atomize pipeline.

Pipeline steps:
0. Canonicalize: Normalize key aliases (context→reference, query→question).
1. Validation: Ensures API key is configured and input data has the source kg key.
2. Filtering:  Skips items that already contain an atomized kg key.
3. Payloading: Builds structured AtomizationPayload per triplet for the worker.
4. Execution:  Delegates to the Atomizer worker (async).
5. Serialization: Overwrites source key with atomized triplets, archives
   originals to _not_atomized for items that had splits.
6. Reporting:  Logs validation, skip, config, results, and done line.
"""

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError, FilterError
from contextchecker.services.base import BaseService
from contextchecker.models import AtomizationPayload
from contextchecker.utils import canonicalize_triplets, deduplicate_triplets
from contextchecker.workers.atomizer import Atomizer, AtomicTriplet, AtomizationDecision
from contextchecker.stats import PhaseStats, log_api_parsing, log_token_stats

logger = settings.get_logger(__name__)


class AtomizationService(BaseService):
    """Orchestrates atomization: validate → filter → atomize → serialize.

    The service owns all validation and filtering logic. The Atomizer
    worker is a dumb execution unit that the service delegates to.

    Operates on items that already have extracted triplets under a source
    kg key (e.g. from ExtractionService). Overwrites that key with
    atomized output and archives originals to {source}_not_atomized
    for items that had compound triplets.
    """

    def __init__(
        self,
        model: str,
        source_kg_key: str,
        base_url: str | None = None,
        concurrency: int = 10,
        max_retries: int | None = None,
        quiet: bool = False,
        dedup: bool = True,
    ):
        # Fail-fast: validate config before creating any workers.
        api_key = self._require_api_key(
            settings.ATOMIZER_API_KEY, "ATOMIZER_API_KEY"
        )

        self.model = model
        self.base_url = base_url
        self._source_kg_key = source_kg_key
        self._archive_kg_key = f"{source_kg_key}_not_atomized"
        self._atomizer = Atomizer(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
            max_retries=max_retries,
        )

        self.quiet = quiet
        self._dedup = dedup
        self.last_trace: list[dict] = []  # per-item decision trace, built in _serialize

    # ── Public API ───────────────────────────────────────────────

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the full atomization pipeline.

        Mutates and returns *data*:
        - Source key overwritten with atomized triplets (+ atomized flag)
        - _not_atomized key added for items that had compounds

        Raises:
            InvalidInputError: No items contain the source kg key.
        """
        # Step 0: Normalize key aliases
        self._canonicalize_keys(data)

        valid = self._validate(data)
        pending, skipped = self._filter(valid)

        self._log_validation(len(data), len(valid))
        self._log_skip(len(valid), skipped, len(pending))

        if not pending:
            if not self.quiet:
                logger.info(" ✨ All items already atomized. Nothing to do.")
            return data

        self._log_config()

        # Count total triplets to process
        total_triplets = sum(
            len(item[self._source_kg_key]) for item in pending
        )

        # Build payloads — one per triplet, flattened across all items
        payloads = self._build_payloads(pending)

        # Execute — fatal errors (auth, connection) propagate to CLI
        if not self.quiet:
            logger.info("── Atomization ───────────────────────────────────────")
        batch_results = await self._atomizer.atomize_batch(payloads)
        phase_stats = self._atomizer.last_stats

        # Serialize results back into dicts (dedups output when enabled)
        atom_stats = self._serialize(pending, payloads, batch_results, phase_stats)

        # Log results
        self._log_results(
            pending_count=len(pending),
            total_triplets=total_triplets,
            total=len(data),
            skipped=skipped,
            phase_stats=phase_stats,
            stats=atom_stats,
        )

        return data

    # run_sync() inherited from BaseService

    # ── Pipeline steps (private) ─────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing required fields.

        Required:
        - source kg key with non-empty triplets
        - response text (needed as LLM context for atomization)
        """
        valid = []
        for i, item in enumerate(data):
            if self._source_kg_key not in item:
                logger.debug(
                    "Item %d has no '%s' key — skipping.", i, self._source_kg_key,
                )
                continue
            if not item[self._source_kg_key]:
                logger.debug(
                    "Item %d has empty '%s' — skipping.", i, self._source_kg_key,
                )
                continue
            if "response" not in item or not item["response"]:
                logger.debug(
                    "Item %d has no 'response' key — skipping.", i,
                )
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                f"No items contain a non-empty '{self._source_kg_key}' key with a 'response'."
            )

        for item in valid:
            canonicalize_triplets(item[self._source_kg_key])
        return valid

    def _filter(self, valid: list[dict]) -> tuple[list[dict], int]:
        """Step 2: Skip items whose triplets are already atomized.

        An item is already-atomized when ALL its triplets have an
        'atomized' key (either True or 'failed'). This works regardless
        of whether the item had splits (_not_atomized key) or was
        all-atomic (no archive key, just the flag on each triplet).

        Returns (pending, skipped_count).
        """
        pending: list[dict] = []
        skipped = 0

        for item in valid:
            triplets = item[self._source_kg_key]
            if all("atomized" in t for t in triplets):
                skipped += 1
                continue
            pending.append(item)

        return pending, skipped

    def _build_payloads(
        self, pending: list[dict],
    ) -> list[AtomizationPayload]:
        """Step 3: Build structured payloads — one per triplet."""
        payloads = []
        for item_idx, item in enumerate(pending):
            triplets = item[self._source_kg_key]
            response = item.get("response", "")
            for tri_idx, triplet in enumerate(triplets):
                s, p, o = self._extract_spo(triplet)
                payloads.append(AtomizationPayload(
                    subject=s,
                    predicate=p,
                    object=o,
                    response=response,
                    item_index=item_idx,
                    triplet_index=tri_idx,
                ))
        return payloads

    @staticmethod
    def _extract_spo(triplet: dict) -> tuple[str, str, str]:
        """Extract (subject, predicate, object) from a canonical triplet dict."""
        return triplet["subject"], triplet["predicate"], triplet["object"]

    def _serialize(
        self,
        items: list[dict],
        payloads: list[AtomizationPayload],
        results: list[AtomizationDecision],
        phase_stats: PhaseStats,
    ) -> dict:
        """Step 5: Write decisions back (overwrite source key) AND build the trace.

        Per source triplet, given its AtomizationDecision:
        - failed                  → keep ORIGINAL untouched, flag "atomized": "failed"
        - is_atomic / no split    → keep ORIGINAL BYTES untouched, flag "atomized": True
                                     (we do NOT trust the LLM's echo on a keep — do-no-harm)
        - genuine split (>=2)     → emit one triplet per child, carrying original metadata
        When dedup is enabled, exact (s,p,o) duplicates in an item's output are
        dropped (loss-free). If an item had any split → archive originals to
        {source}_not_atomized.

        Populates self.last_trace (per item: input triplets + every decision +
        reasoning + children) and returns aggregate stats for the results log.
        """
        self.last_trace = []
        stats = {"keep": 0, "split": 0, "failed": 0, "children": 0, "dups": 0, "output": 0}

        # Group decisions by item, carrying the flat index for failure lookup.
        per_item: dict[int, list[tuple[int, int, AtomizationDecision]]] = {}
        for flat_idx, (payload, decision) in enumerate(zip(payloads, results)):
            per_item.setdefault(payload.item_index, []).append(
                (payload.triplet_index, flat_idx, decision)
            )

        failed_indices = set(phase_stats.failed_indices) if phase_stats else set()

        for item_idx, item in enumerate(items):
            if item_idx not in per_item:
                continue

            original_triplets = item[self._source_kg_key]
            groups = sorted(per_item[item_idx], key=lambda x: x[0])

            new_triplets = []
            had_split = False
            decisions_trace = []

            for tri_idx, flat_idx, decision in groups:
                original = original_triplets[tri_idx]
                s, p, o = self._extract_spo(original)

                if flat_idx in failed_indices:
                    merged = dict(original)
                    merged["atomized"] = "failed"
                    new_triplets.append(merged)
                    label, children = "failed", []
                    stats["failed"] += 1
                elif (not decision.is_atomic) and len(decision.split) >= 2:
                    had_split = True
                    label = "split"
                    children = [
                        {"subject": c.subject, "predicate": c.predicate, "object": c.object}
                        for c in decision.split
                    ]
                    for child in decision.split:
                        merged = dict(original)
                        self._apply_spo(merged, child)
                        merged["atomized"] = True
                        new_triplets.append(merged)
                    stats["split"] += 1
                    stats["children"] += len(children)
                else:
                    # KEEP — original bytes untouched (ignore any LLM echo).
                    merged = dict(original)
                    merged["atomized"] = True
                    new_triplets.append(merged)
                    label, children = "keep", []
                    stats["keep"] += 1

                decisions_trace.append({
                    "triplet_index": tri_idx,
                    "original": {"subject": s, "predicate": p, "object": o},
                    "decision": label,
                    "reasoning": decision.reasoning,
                    "children": children,
                })

            # Exact (s,p,o) duplicates in this item's output — counted always,
            # removed only when dedup is enabled (loss-free cleanup).
            deduped = deduplicate_triplets(new_triplets)
            item_dups = len(new_triplets) - len(deduped)
            stats["dups"] += item_dups
            if self._dedup:
                new_triplets = deduped

            self.last_trace.append({
                "id": item.get("id"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                "input_triplets": list(original_triplets),
                "decisions": decisions_trace,
                "duplicates_removed": item_dups,
            })

            if had_split:
                item[self._archive_kg_key] = list(original_triplets)
            item[self._source_kg_key] = new_triplets
            stats["output"] += len(new_triplets)

        stats["input"] = stats["keep"] + stats["split"] + stats["failed"]
        return stats

    @staticmethod
    def _apply_spo(target: dict, child: AtomicTriplet) -> None:
        """Write S/P/O from an AtomicTriplet into a canonical target dict."""
        target["subject"] = child.subject
        target["predicate"] = child.predicate
        target["object"] = child.object

    # ── Logging ─────────────────────────────────────────────────

    def _log_validation(self, total: int, valid: int) -> None:
        """Print 📂 Validation section."""
        if self.quiet:
            return
        invalid = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if invalid > 0:
            logger.info("    ├─ dropped:  %d  (no '%s' key or empty)", invalid, self._source_kg_key)
        logger.info("    └─ valid:  %d items", valid)
        logger.info("")

    def _log_skip(self, valid: int, skipped: int, pending: int) -> None:
        """Print 🔄 Skip section. Hidden entirely when nothing was skipped."""
        if self.quiet:
            return
        if skipped == 0:
            return
        logger.info(" 🔄 Skip: items with existing atomization")
        logger.info("    Total:      %d valid items", valid)
        logger.info("    ├─ skipped: %d items (already atomized)", skipped)
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
        logger.info("    Source key:   %s", self._source_kg_key)
        logger.info("    Archive key:  %s", self._archive_kg_key)
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(
        self,
        pending_count: int,
        total_triplets: int,
        total: int,
        skipped: int,
        phase_stats: PhaseStats,
        stats: dict,
    ) -> None:
        """Print the full results block."""
        logger.info("")
        logger.info("── ATOMIZER RESULTS ─────────────────────────────────────────")
        logger.info("")
        log_api_parsing(total_triplets, phase_stats)
        self._log_bl_results(stats)
        log_token_stats()
        self._log_done(total, stats, skipped)

    def _log_bl_results(self, stats: dict) -> None:
        """Print 📝 Atomization summary — input→output with the keep/split/failed
        breakdown, the claims splitting added, and any exact duplicates removed
        (removed when dedup is on, only reported when off). Duplicates get their
        own top-level line because they can arise from the input OR from a split.
        """
        keep, split, failed = stats["keep"], stats["split"], stats["failed"]
        dups = stats["dups"]
        gross_new = stats["children"] - split  # claims splitting added (pre-dedup)

        logger.info(" 📝 Atomization:")
        logger.info(
            "    %d input triplets → %d output triplets", stats["input"], stats["output"]
        )

        # Only present sections show; the last one gets the └─ connector.
        rows = (
            (["atomic"] if keep > 0 else [])
            + (["split"] if split > 0 else [])
            + (["dups"] if dups > 0 else [])
            + (["failed"] if failed > 0 else [])
        )
        for i, kind in enumerate(rows):
            last = i == len(rows) - 1
            conn = "└─" if last else "├─"
            if kind == "atomic":
                logger.info("    %s ✅ %d triplets already atomic", conn, keep)
            elif kind == "split":
                logger.info("    %s 🔀 %d compound triplets split", conn, split)
                bar = "    " if last else "│   "
                word = "claim" if gross_new == 1 else "claims"
                logger.info("    %s└─ ➕ %d new %s from splitting", bar, gross_new, word)
            elif kind == "dups":
                word = "duplicate" if dups == 1 else "duplicates"
                verb = "removed" if self._dedup else "found — kept (dedup off)"
                logger.info("    %s 🔁 %d %s %s", conn, dups, word, verb)
            elif kind == "failed":
                logger.info("    %s ⚠️  %d failed to check for atomization", conn, failed)
        logger.info("")

    def _log_done(
        self, total: int, stats: dict, skipped: int,
    ) -> None:
        """Print ✅ Done summary line — items processed + what actually happened."""
        if self.quiet:
            return
        parts = [f"{stats['input']} → {stats['output']} triplets"]
        if stats["split"] > 0:
            parts.append(f"{stats['split']} split")
        if stats["dups"] > 0:
            parts.append(f"{stats['dups']} dups {'removed' if self._dedup else 'found'}")
        if stats["failed"] > 0:
            parts.append(f"{stats['failed']} failed")
        if skipped > 0:
            parts.append(f"{skipped} skipped")
        logger.info(" ✅ Done: %d items · %s", total, " · ".join(parts))
