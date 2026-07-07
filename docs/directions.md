# Directions — the shared execution core of RAGChecker-style pipelines

`models.Direction` + `pipelines/directions.py`

## The concept

A **direction** is one comparison: *claims from one triplet list, checked
against one reference source, verdicts written under one namespace*. The
RAGChecker methodology is four of them (`answer2response`,
`response2answer`, `retrieved2response`, `retrieved2answer`); the
faithfulness pipeline is exactly one (`retrieved2response`). Refcheck's
checking step and the extractor eval's two matching passes are the same
shape and are candidates for opportunistic migration (deliberately not
big-banged — the eval is a measurement instrument and gets refactored only
with before/after regression runs).

## Where it sits in the architecture

`run_direction(checking_service, items, direction)` is a **composition
helper in the pipelines layer** — deliberately none of the existing
categories:

- not a worker: it makes no LLM calls itself;
- not a service: it has no validate/filter/log/serialize lifecycle — the
  calling pipeline owns those steps;
- not a util: it calls a service, and `utils.py` sits below services in the
  import graph.

The pipeline constructs one `CheckingService` per direction with
`kg_key=direction.kg_key` and a direction-specific `verdict_namespace`
(e.g. `{checker}_answer2response`), then hands both to the runner. The
runner reads the service's output-contract properties (`verdict_key`,
`checker_error_key`, `extraction_error_key`) — never its privates.

## The Direction contract

```python
@dataclass
class Direction:
    name: str                              # "answer2response" — identity/logging
    kg_key: str                            # triplet list supplying the claims
    reference_key: str | None = None       # item field checked against (flat mode)
    per_chunk: bool = False                # matrix mode toggle
    chunks_key: str = "retrieved_context"  # chunk list field (matrix mode)
```

## Flat mode (per_chunk=False)

Claims vs `item[reference_key]` — one verdict per claim.

The runner builds shadow items `{reference, response, kg_key}` where the
triplet list is the **same object** as in the source item. The
CheckingService mutates the triplets in place, so verdicts land on the real
data with no fold-back step. String references are normalized to
single-passage lists. Items missing the reference field are skipped.
A flat direction without `reference_key` raises immediately (programming
error in the calling pipeline, not a data problem).

## Matrix mode (per_chunk=True)

Claims vs each chunk — a verdict per (claim, chunk) cell.

One shadow item per (item, chunk) pair, over **deep copies** of the
triplets: the same verdict key would collide across chunks otherwise. All
pairs go through the service as a single batch (one progress bar, full
concurrency, joint chunking per pair). Afterwards the per-doc verdicts are
folded back onto the original triplets:

```
triplet["{namespace}_verdicts"] = {"000": "Entailment", "001": "Neutral", ...}
triplet["{namespace}_errors"]   = {"001": "parse_failure"}   # only non-null causes
```

Chunk identity: `{doc_id, text}` dicts are used as-is; bare strings get
synthesized ids (`000`, `001`, ...).

## Failure semantics

"Nothing to check" in a single direction (all items empty, all already
done — `FilterError` / `InvalidInputError` from the service) is a normal
per-direction outcome: logged at warning level and swallowed, never a
pipeline crash. Null verdicts carry their cause through
`{namespace}_error(s)` per the outcome-marker rules (docs/outcome_markers.md).

## Shared input helpers (same module)

- `unwrap_items(data)` — accepts the original RAGChecker
  `{"results": [...]}` envelope or a bare list; anything else raises
  `InvalidInputError`. Legacy input accepted, never emitted.
- `normalize_chunks(chunks)` — `[{doc_id, text}]` out, whatever came in.
- `phase_failure_lines(stats)` — log-tree failure sub-lines from a worker's
  `PhaseStats` (retryable/recovered, context too long, content policy,
  exhausted retries).
