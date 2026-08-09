# Outcome markers — why `[]` is never ambiguous

Before these markers existed, an empty extraction result (`kg: []`) had five
possible causes that were indistinguishable on disk: heuristic abstention,
model-detected abstention, genuinely-no-facts, context-too-long /
content-policy rejection, and parse failure after all retries. Downstream
code (the checker's skip logic, the extractor eval's abstention buckets,
every metric) was forced to guess — and guessed "abstention", which meant
tooling failures were silently scored as model behavior: an eval could count
our own parse failure as a "correct abstention", and a metrics pass would
inflate abstention rates with crashes.

The rule set that fixes it:

## Extraction outcomes (item level)

| On disk | Meaning |
| --- | --- |
| `kg: [...claims...]` | Normal extraction |
| `kg: []` + `{model}_extraction_error: <cause>` | **Tooling failure. Never an abstention.** Causes: `parse_failure` (no valid response after all retry rounds — the most common), `context_too_long`, `content_policy`. |
| `kg: []` + `is_abstention: true` + `abstention_source: "heuristic"\|"llm"` | Abstention. `heuristic` = the English refusal-phrase pre-filter caught it before any LLM call (a cheap fast-path; ~impossible false positives at its 85% coverage threshold). `llm` = the extractor returned an empty claims array. |
| `kg: []` bare (older files) | Treated as abstention (empty-without-error). By project semantics, a response that asserts nothing IS an abstention — whether it was *justified* is a separate question answered by ragcheck's retrieval evidence. |

Conventions:

- Markers live on the item dicts (in memory and in files) because composite
  consumers (evals, pipelines) work on the data structures, not on files.
- The error key is model-prefixed (`{model}_extraction_error`) like the kg
  key — multi-extractor files must not mix up whose attempt failed. Pipelines
  extracting from other fields use their own error keys
  (`{model}_gt_extraction_error`).
- `is_abstention` is item-level and sparse (absent = false): it describes the
  *response*, so only the response extraction may write it — and only it may
  *delete* it. A service with `mark_abstention=False` (gt_answer extraction)
  neither writes nor clears the flags (regression-tested; the gt run once
  erased a correctly-set flag).
- Markers are cleared and rewritten on re-extraction, and an error marker is
  *not a result*: errored items re-enter the pending pool on re-runs instead
  of being skipped forever.
- The flag is deliberately human-editable: correct a misclassified item in
  the file and every metric recomputes from disk without an LLM call.
- Planned upgrade (A/B via the extractor eval suite before switching): an
  explicit `is_abstention` boolean in the extractor's response schema —
  language-agnostic classification instead of the empty-array convention,
  which currently both under-triggers (the model extracts SPO from refusal
  text) and is English-only in the heuristic path.

## Checking outcomes (triplet level)

| On triplet | Meaning |
| --- | --- |
| `{namespace}_verdict: "Entailment"\|"Contradiction"\|"Neutral"` | Normal verdict (+ `{namespace}_explanation`) |
| `{namespace}_verdict: null` + `{namespace}_error: <cause>` | The check itself failed — same three causes. |
| Matrix directions | `{namespace}_verdicts: {doc_id: verdict}` + `{namespace}_errors: {doc_id: cause}` (only non-null causes) |

## The consumption rule (metrics, evals, skip logic)

**Unknown is not "no".** A null verdict / errored extraction is excluded
from numerator AND denominator — never counted as "not entailed" or "not
extracted". Concretely:

- Metrics: error items are fully excluded and counted in
  `extraction_error_rate`; null verdicts feed `checker_failure_rate`; in
  matrix rows, known cells decide when they can (one Entailment grounds a
  claim regardless of unknown cells; no Entailment + an unknown cell makes
  the claim undecidable and excluded).
- Extractor eval: errored items go to their own bucket (excluded from
  P/R/F1 and from the abstention buckets), reported as an error rate with a
  per-cause breakdown, and listed in the disagreements file for
  identification.
- Checker skip stats: empty-kg items are reported as `abstained` vs
  `extraction_failed`, not lumped as "empty claims".

The information source: workers classify per-item failure causes during
batch execution (`PhaseStats.error_causes`) and services persist them at
serialization time — "we can catch ourselves, and the file remembers it."
