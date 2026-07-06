"""
Direction runner — the shared execution block for comparison directions.

A "direction" is one comparison in a RAGChecker-style pipeline
(answer2response, response2answer, retrieved2response, retrieved2answer):
claims from one triplet list checked against one reference source, verdicts
written under one namespace.

This is neither a service nor a worker — it is a composition helper in the
pipelines layer. It owns no validation/logging lifecycle (the calling
pipeline does) and makes no LLM calls itself (the CheckingService it drives
does). It cannot live in utils.py because it calls a service, and utils sit
below services in the import graph.

How verdicts land on the source items:
- Flat mode: shadow items share the triplet list objects with the source
  items, so the CheckingService writes verdict keys onto the real triplets
  in place. No fold-back needed.
- Matrix mode (per_chunk): one shadow item per (item, chunk) over DEEP
  COPIES of the triplets — the same verdict key would collide across chunks
  otherwise. Afterwards the per-doc results are folded back onto the
  original triplets as {doc_id: verdict} dicts under "{namespace}_verdicts"
  (and "{namespace}_errors" for null-verdict causes).
"""

import copy

from contextchecker import settings
from contextchecker.exceptions import FilterError, InvalidInputError
from contextchecker.models import Direction
from contextchecker.services.checking import CheckingService

logger = settings.get_logger(__name__)


# ── Input normalization ──────────────────────────────────────────────────────

def _normalize_reference(value) -> list[str]:
    """The checker contract wants a list of passages; accept bare strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _chunk_id_and_text(chunk, index: int) -> tuple[str, str]:
    """Accept {'doc_id','text'} dicts or bare strings (ids synthesized)."""
    if isinstance(chunk, dict):
        return str(chunk.get("doc_id", f"{index:03d}")), str(chunk.get("text", ""))
    return f"{index:03d}", str(chunk)


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_direction(
    checking: CheckingService,
    items: list[dict],
    direction: Direction,
) -> None:
    """Run one comparison direction over *items*, writing verdicts in place.

    The CheckingService must be constructed with kg_key=direction.kg_key and
    a direction-specific verdict_namespace — the runner reads the output key
    names from the service's public contract properties.

    "Nothing to check" (all items empty/already done) is a normal outcome
    for a single direction, not a pipeline failure — FilterError and
    InvalidInputError from the service are logged and swallowed.
    """
    # Fail-fast: a flat direction without a reference source is a
    # programming error in the calling pipeline, not a data problem.
    if not direction.per_chunk and not direction.reference_key:
        raise ValueError(
            f"Direction '{direction.name}' is flat but has no reference_key."
        )

    if direction.per_chunk:
        await _run_matrix(checking, items, direction)
    else:
        await _run_flat(checking, items, direction)


async def _run_flat(
    checking: CheckingService, items: list[dict], direction: Direction
) -> None:
    """Claims vs item[reference_key]: verdicts written in place via shared
    triplet lists."""
    shadow: list[dict] = []
    for item in items:
        if direction.reference_key not in item:
            continue
        shadow_item = {
            "reference": _normalize_reference(item[direction.reference_key]),
            "response": item.get("response", ""),
            # Shared object, not a copy — the service mutates the real triplets.
            direction.kg_key: item.get(direction.kg_key, []),
        }
        # Carry the extraction-error marker so skip stats classify correctly.
        if checking.extraction_error_key in item:
            shadow_item[checking.extraction_error_key] = item[checking.extraction_error_key]
        shadow.append(shadow_item)

    await _run_service(checking, shadow, direction)


async def _run_matrix(
    checking: CheckingService, items: list[dict], direction: Direction
) -> None:
    """Claims vs each chunk: deep-copied triplets per (item, chunk), folded
    back as {doc_id: verdict} dicts on the original triplets."""
    shadow: list[dict] = []
    bookkeeping: list[tuple[dict, str, list[dict]]] = []  # (item, doc_id, copies)

    for item in items:
        chunks = item.get(direction.chunks_key) or []
        triplets = item.get(direction.kg_key) or []
        for idx, chunk in enumerate(chunks):
            doc_id, text = _chunk_id_and_text(chunk, idx)
            copies = copy.deepcopy(triplets)
            shadow_item = {
                "reference": [text],
                "response": item.get("response", ""),
                direction.kg_key: copies,
            }
            if checking.extraction_error_key in item:
                shadow_item[checking.extraction_error_key] = item[checking.extraction_error_key]
            shadow.append(shadow_item)
            bookkeeping.append((item, doc_id, copies))

    if not await _run_service(checking, shadow, direction):
        return

    # Fold: per-doc verdicts from the copies → matrix dicts on the originals.
    matrix_verdict_key = checking.verdict_key + "s"
    matrix_error_key = checking.checker_error_key + "s"
    for item, doc_id, copies in bookkeeping:
        originals = item.get(direction.kg_key) or []
        for original, copied in zip(originals, copies):
            original.setdefault(matrix_verdict_key, {})[doc_id] = copied.get(
                checking.verdict_key
            )
            error = copied.get(checking.checker_error_key)
            if error:
                original.setdefault(matrix_error_key, {})[doc_id] = error


async def _run_service(
    checking: CheckingService, shadow: list[dict], direction: Direction
) -> bool:
    """Run the service over the shadow items. Returns True if it executed.

    Empty shadow lists and service-level "nothing to check" are normal
    per-direction outcomes: log and continue, never crash the pipeline.
    """
    if not shadow:
        logger.debug("Direction %s: no eligible items.", direction.name)
        return False
    try:
        await checking.run(shadow)
        return True
    except (FilterError, InvalidInputError) as exc:
        logger.warning("Direction %s: nothing to check (%s)", direction.name, exc)
        return False
