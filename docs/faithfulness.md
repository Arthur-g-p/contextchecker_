# Faithfulness Pipeline

`claimlens faithcheck` (batch) and `check_faithfulness(...)` (library,
real-time) — faithfulness checking **without ground truth**.

## Why this exists

Faithfulness is the only RAGChecker metric that needs no reference answer:
it is derived purely from `retrieved2response` (response claims vs retrieved
chunks). That makes it computable on live production traffic, where no
gt_answer exists — the monitoring use case ragcheck structurally cannot
serve. One extraction + one matrix direction; the same building blocks as
ragcheck, composed smaller.

Without ground truth there is no hallucination / self_knowledge split (both
need correctness): an unfaithful claim here means "not grounded in the
retrieved context", whether it happens to be true in the world or not.

## Input

Required per item (hard drop): `response`, `retrieved_context`. An empty
`response` string is data — a full abstention — not a missing field;
absent or `null` is missing. An empty chunk list is a drop.
(`[{doc_id, text}]` or bare strings). No `gt_answer` anywhere. The
`{"results": [...]}` envelope is accepted.

## Library facade (the real-time entry point)

```python
from claimlens import check_faithfulness

entry = check_faithfulness(
    response="The Nile is the longest river. It has 5 million inhabitants.",
    retrieved_context=[
        {"doc_id": "000", "text": "The Nile is traditionally considered the longest river."},
        {"doc_id": "001", "text": "The Nile flows through Egypt."},
    ],
    extractor_model="...",
    checker_model="...",
    # optional: extractor_base_url=, checker_base_url=, joint_num=, ...
)

entry["metrics"]["faithfulness"]   # 0.5  — 1 of 2 claims grounded
entry["claim_support"]             # [["000"], []] — which chunks ground each claim
entry["is_abstention"]             # explicit refusal detection
```

No CLI, no files, runs in process (`quiet=True` internally; library imports
produce no log output unless `enable_logging()` is called). Requires
`EXTRACTOR_API_KEY` / `CHECKER_API_KEY` in the environment like everything
else. Note: the internal progress bars write to stderr — cosmetic in
notebooks, known minor issue for embedded use.

## Report

Two files, the same skeleton as `ragcheck` and the evals (see
docs/ragchecker.md, "Output"): the **record**
`{_args, _meta, metrics, variance, runs: [{_meta, metrics, counts, items}]}`
and the **findings** `{_args, _meta, runs: [{_meta, findings}]}`. `runs` is a
list at `--runs 1` too; `--runs N` adds entries and reshapes nothing.

- `metrics` — `faithfulness`, `abstention_rate`, `extraction_error_rate`,
  `checker_failure_rate`: the variance roster, means over runs at the top,
  the run's own values inside each run entry.
- `counts` — `support` (items behind the macro average), `pipeline` (requests
  and tallies per phase), `abstention` (evaluated / errored / abstained /
  answered), `reliability` (the two 💥 rows).
- `items` — one entry per evaluated item: dict claims, the
  `retrieved2response` verdict-object matrix, explicit `is_abstention`,
  sparse `extraction_errors`, per-item `metrics`, and `claim_support` —
  parallel to `response_claims`, for each claim the list of doc_ids whose
  chunk entails it: the per-claim attribution answer to "which retrieved
  document grounds this statement?".
- `findings` — the review queue, one list per branch: `ungrounded` (no chunk
  entails the claim), `contradicted` (a chunk contradicts it, named with
  its explanation), `undecidable` (no chunk entails it and a verdict is
  missing, so the score excludes it), `abstained`, `extraction_failed`.
  Empty branches stay present.

## Metrics

- Per item: `faithfulness` = grounded claims / decidable claims. Same
  None-propagation and gating rules as ragcheck (errors and abstentions are
  `null`, unknown matrix rows leave both sides of the ratio).
- Overall: macro `faithfulness` with `support`, plus `abstention_rate`
  (denominator: evaluated − extraction-errored — a tooling failure is
  charged once, in `extraction_error_rate`, never by diluting a behavior
  rate), `extraction_error_rate`, `checker_failure_rate`. There is no
  justified/unjustified abstention split here — that requires ground truth.
