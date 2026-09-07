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
  otherwise. Afterwards the per-chunk results are folded back onto the
  original triplets as {chunk_index: ...} dicts under "{namespace}_verdicts"
  and "{namespace}_explanations" (plus "{namespace}_errors" for
  null-verdict causes). Keyed by position, not doc_id: a corpus chunked from
  one document repeats the same doc_id across its chunks, which would collapse
  every cell in the row onto the last one checked.
"""

import copy

from claimlens import settings
from claimlens.exceptions import FilterError, InvalidInputError
from claimlens.models import Direction
from claimlens.services.checking import CheckingService
from claimlens.stats import PhaseStats, log_mece_tree

logger = settings.get_logger(__name__)


# ── Input normalization (shared by RAGChecker-style pipelines) ───────────────

def unwrap_items(data) -> list[dict]:
    """Accept the original RAGChecker input envelope {"results": [...]} as
    well as a bare item list — legacy input accepted, never emitted."""
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise InvalidInputError(
        "Input must be a list of items or a {'results': [...]} document, "
        f"got {type(data).__name__}."
    )


def normalize_chunks(chunks: list) -> list[dict]:
    """Accept [{'doc_id','text'}] or bare strings; always emit dicts."""
    normalized = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            normalized.append({
                "doc_id": str(chunk.get("doc_id", f"{i:03d}")),
                "text": str(chunk.get("text", "")),
            })
        else:
            normalized.append({"doc_id": f"{i:03d}", "text": str(chunk)})
    return normalized


def abstention_counts(entries: list[dict]) -> dict:
    """Item-level behavior counts over report entries.

    Single source for both the metric rates and the ⚪ Abstention Behavior
    tree, so footer fractions can never drift from the JSON rates."""
    evaluated = len(entries)
    errored = sum(1 for e in entries if e.get("extraction_errors"))
    abstained = sum(1 for e in entries
                    if e.get("is_abstention") and not e.get("extraction_errors"))
    return {
        "evaluated": evaluated,
        "errored": errored,
        "abstained": abstained,
        "answered": evaluated - errored - abstained,
    }


def _location(service) -> str:
    """``model @ base_url`` for Config blocks — the same string the bare
    services print for themselves."""
    return f"{service.model} @ {service.base_url}" if service.base_url else service.model


def log_pipeline_tree(phases: list[tuple[str, str, object, str]]) -> None:
    """🔀 Pipeline: one branch per phase. HTTP requests partition the total;
    claims, verdict tallies and failure notes are foreign numbers and ride
    in the square bracket (output_conventions rule 1.4).

    *phases* are (icon, name, PhaseStats-or-None, summary)."""
    total = sum(p[2].http_requests for p in phases if p[2])
    branches = []
    for icon, name, stats, summary in (p[:4] for p in phases):
        foreign = " · ".join([summary] + phase_failure_lines(stats))
        branches.append((icon, stats.http_requests if stats else 0, name,
                         f"[{foreign}]"))
    log_mece_tree("🔀 Pipeline", total, "LLM requests", branches)
    logger.info("")


def phase_record(stats, **counts) -> dict:
    """One ``counts.pipeline`` entry: the phase's requests plus its own
    tallies (claims, verdicts), and the failure numbers behind the ♻️ / 💥
    notes — only when something failed. *stats* is the worker's PhaseStats
    (None before the phase ran)."""
    live = isinstance(stats, PhaseStats)
    record = {"requests": stats.http_requests if live else 0, **counts}
    if live:
        failures: dict = {}
        if stats.parse_error:
            failures["retryable"] = stats.parse_error
            failures["recovered"] = sum(r.recovered for r in stats.rounds)
        for cause in ("context_too_long", "content_policy",
                      "finish_reason_length", "timeout"):
            if getattr(stats, cause):
                failures[cause] = getattr(stats, cause)
        if stats.permanently_failed:
            failures["exhausted_retries"] = stats.permanently_failed
        if failures:
            record["failures"] = failures
    return record


def pipeline_counts(phases: list[tuple]) -> dict:
    """``counts.pipeline`` for the record from the same *phases* the 🔀 tree
    prints: (icon, name, PhaseStats, summary, tallies)."""
    return {name: phase_record(stats, **(tallies or {}))
            for _, name, stats, _, tallies in phases}


def verdict_summary(counts: dict) -> str:
    """``72 verdicts · 🟢 17 · 🔴 1 · ⚪ 54`` (+ ``· ❓ n`` for null cells)."""
    text = (f"{counts['total']} verdicts · 🟢 {counts['Entailment']}"
            f" · 🔴 {counts['Contradiction']} · ⚪ {counts['Neutral']}")
    if counts.get("unknown"):
        text += f" · ❓ {counts['unknown']}"
    return text


def phase_failure_lines(stats) -> list[str]:
    """Failure sub-lines for one pipeline phase, only when something went
    wrong. Derived from the worker's PhaseStats."""
    if stats is None:
        return []
    lines = []
    if stats.parse_error:
        recovered = sum(r.recovered for r in stats.rounds)
        lines.append(f"♻️  {stats.parse_error} retryable → {recovered} recovered")
    permanent = []
    if stats.context_too_long:
        permanent.append(f"{stats.context_too_long} context too long")
    if stats.content_policy:
        permanent.append(f"{stats.content_policy} content policy")
    if stats.finish_reason_length:
        permanent.append(f"{stats.finish_reason_length} finish reason length")
    if stats.timeout:
        permanent.append(f"{stats.timeout} timeout")
    if stats.permanently_failed:
        permanent.append(f"{stats.permanently_failed} exhausted retries")
    if permanent:
        lines.append("💥 " + ", ".join(permanent))
    return lines


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
    bookkeeping: list[tuple[dict, int, list[dict]]] = []  # (item, chunk_idx, copies)

    for item in items:
        chunks = item.get(direction.chunks_key) or []
        triplets = item.get(direction.kg_key) or []
        for idx, chunk in enumerate(chunks):
            _, text = _chunk_id_and_text(chunk, idx)
            copies = copy.deepcopy(triplets)
            shadow_item = {
                "reference": [text],
                "response": item.get("response", ""),
                direction.kg_key: copies,
            }
            if checking.extraction_error_key in item:
                shadow_item[checking.extraction_error_key] = item[checking.extraction_error_key]
            shadow.append(shadow_item)
            bookkeeping.append((item, idx, copies))

    if not await _run_service(checking, shadow, direction):
        return

    # Fold: per-doc verdicts + explanations from the copies → matrix dicts
    # on the originals.
    matrix_verdict_key = checking.verdict_key + "s"
    matrix_explanation_key = checking.explanation_key + "s"
    matrix_error_key = checking.checker_error_key + "s"
    for item, chunk_idx, copies in bookkeeping:
        originals = item.get(direction.kg_key) or []
        for original, copied in zip(originals, copies):
            original.setdefault(matrix_verdict_key, {})[chunk_idx] = copied.get(
                checking.verdict_key
            )
            original.setdefault(matrix_explanation_key, {})[chunk_idx] = copied.get(
                checking.explanation_key
            )
            error = copied.get(checking.checker_error_key)
            if error:
                original.setdefault(matrix_error_key, {})[chunk_idx] = error


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
