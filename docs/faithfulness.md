# Faithfulness Pipeline

`contextchecker faithcheck` (batch) and `check_faithfulness(...)` (library,
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

Required per item (hard drop): `response`, `retrieved_context`
(`[{doc_id, text}]` or bare strings). No `gt_answer` anywhere. The
`{"results": [...]}` envelope is accepted.

## Library facade (the real-time entry point)

```python
from contextchecker import check_faithfulness

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

Same conventions as the ragcheck report (`schema_version: 2`,
`report_type: "faithfulness"`): dict claims, verdict-object matrix,
explicit `is_abstention`, sparse `extraction_errors`, per-item `metrics`
plus `overall_metrics`. Additional field:

- `claim_support` — parallel to `response_claims`: for each claim, the list
  of doc_ids whose chunk entails it. The per-claim attribution answer to
  "which retrieved document grounds this statement?".

## Metrics

- Per item: `faithfulness` = grounded claims / decidable claims. Same
  None-propagation and gating rules as ragcheck (errors and abstentions are
  `null`, unknown matrix rows leave both sides of the ratio).
- Overall: macro `faithfulness` with `support`, plus `abstention_rate`,
  `extraction_error_rate`, `judge_failure_rate`. There is no
  justified/unjustified abstention split here — that requires ground truth.
