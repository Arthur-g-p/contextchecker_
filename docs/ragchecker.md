# RagChecker Pipeline

`contextchecker ragcheck` — full RAGChecker-style RAG evaluation:
2 extractions + 4 checking directions + 11 metrics, one self-contained
report file. Implements the methodology of RAGChecker (Ru et al., 2024)
with modernized data contracts, explicit error handling, and additional
reliability metrics.

## Input

A JSON list of items, or the original RAGChecker envelope
`{"results": [...]}` (accepted at the boundary, never emitted).

Required per item (hard drop — falsy counts as missing, because an empty
`gt_answer` or chunk list produces meaningless metrics):

| Key | Meaning |
| --- | --- |
| `response` | The RAG system's answer under evaluation |
| `gt_answer` | The ground-truth answer |
| `retrieved_context` | Chunks: `[{doc_id, text}]` or bare strings (ids synthesized as `000`, `001`, ...) |

Optional: `query_id` (falls back to `id`), `query` (canonicalized to
`question` internally, emitted back as `query`).

Items without ground truth belong in the faithfulness pipeline
(docs/faithfulness.md), not in a degraded ragcheck.

## What runs

```
response  --extract-->  {ext}_response_kg
gt_answer --extract-->  {ext}_gt_answer_kg     (mark_abstention=False:
                                                an empty GT extraction is a
                                                data-quality signal, not a
                                                response abstention)

answer2response      response claims vs gt_answer          (flat)
response2answer      gt claims       vs response           (flat)
retrieved2response   response claims vs each chunk         (matrix)
retrieved2answer     gt claims       vs each chunk         (matrix)
```

Each direction gets its own CheckingService with a direction-specific
verdict namespace (`{checker}_answer2response_verdict`, ...) so verdicts
over the same triplets never collide. Matrix directions run one joint
check per (item, chunk) and fold back `{chunk_index: verdict}` dicts —
keyed by position, because a corpus chunked from one document repeats the
same doc_id across its chunks.

There is no item-level skipping in v1 (planned for 2.0 together with
report re-ingestion for manually corrected ground truth). A run that dies
part-way is re-run from the start.

## Output: the report (single artifact)

The CLI writes exactly one file (default
`results/{input_stem}_ragcheck[_{runs}].json`, the suffix only for
`--runs > 1`) from `pipeline.last_report`. The
report preserves the original RAGChecker *structure* (results list, four
directional arrays parallel to the claims) with modernized leaves:

- claims are `{subject, predicate, object}` dicts (never `[s, p, o]` arrays),
- verdict entries are objects: `{"verdict": ..., "explanation": ...}` in the
  flat directions, `{"verdict": ...}` in the matrices (explanations in the
  matrices are claims x chunks output tokens and are deliberately omitted),
- `is_abstention` is explicit per item (sparse in the working data, explicit
  in the report — a report is a view for humans and frontends),
- `relevant_chunks` lists the doc_ids that entail at least one gt claim
  (the load-bearing intermediate for context_utilization and noise
  sensitivity, exposed so consumers never re-derive it from the matrix),
- `extraction_errors` appears only on items where tooling failed,
- `metrics` per item plus `overall_metrics` at the top, `_meta` with
  `schema_version: 3`.

Consumers reading both old paper outputs and these reports need one
canonicalization rule: string verdict entry = old format, object = new;
missing field = "information not provided", never a crash.

## The 11 paper metrics

Formulas follow the original RAGChecker; they are anchored by a unit test
reproducing the original implementation's reference output value-for-value
(`tests/unit/test_ragchecker.py::TestMetricsReference`).

Notation: R = response claims, G = gt claims, C = chunks. "Correct" =
entailed by gt_answer (`answer2response`). "Grounded" = entailed by at
least one chunk.

| Metric | Definition |
| --- | --- |
| precision | correct response claims / R |
| recall | gt claims entailed by response / G |
| f1 | harmonic mean of the two |
| claim_recall | gt claims grounded in any chunk / G (retriever) |
| context_precision | chunks entailing >=1 gt claim / C (retriever) |
| faithfulness | grounded response claims / R |
| hallucination | ungrounded AND incorrect response claims / R |
| self_knowledge | ungrounded AND correct response claims / R |
| context_utilization | gt claims grounded in chunks AND entailed by response / gt claims grounded in chunks (paper definition, GT-side) |
| noise_sensitivity_in_relevant | incorrect response claims entailed by a relevant chunk / R |
| noise_sensitivity_in_irrelevant | incorrect response claims entailed only by irrelevant chunks / R |

Identity (enforced by a shared denominator and asserted in tests):
`faithfulness = 1 - hallucination - self_knowledge`.

### Semantics that make the numbers trustworthy

- **None-verdict propagation.** A failed check is *unknown*, never "not
  entailed". Unknown claims leave numerator AND denominator, so checker
  failures cannot inflate hallucination. In matrix rows, known cells decide
  when they can: one Entailment makes a claim grounded regardless of unknown
  cells; no Entailment plus an unknown cell makes the claim undecidable and
  excluded.
- **Zero denominators are `null`, never `0.0`.** "Not computable" is not a
  score.
- **Gating.** Extraction error (either side) → the item is fully excluded:
  every metric `null`, counted in the error rate instead. Abstention → the
  generator family is `null`, but the retrieval metrics (claim_recall,
  context_precision) are still computed — `retrieved2answer` does not
  involve the response, and an abstention says nothing about the retriever.
- **Aggregation is macro** (per paper): per-item metrics averaged over
  contributing items, nulls skipped. `overall_metrics.support` reports how
  many items actually contributed to each average — exclusions shrink N
  invisibly otherwise. Micro aggregation is recomputable from the report
  (all verdict arrays are preserved) without any LLM calls.

## New metrics (beyond the paper)

| Metric | Definition | Why it exists |
| --- | --- | --- |
| `abstention_rate` | abstained items / items the model got to answer (evaluated − extraction-errored). Distribution information, never a quality score. | The paper punishes "I don't know" as a wrong answer, tanking generator metrics. Here abstentions are excluded from the generator family — which would be gameable (abstain on everything, look perfect) unless the rate itself is a headline number. |
| `justified_abstention_rate` | abstained / **unanswerable** items (those annotated `"gt_answer": ""` — no answer exists) | Correct silence, judged by the annotation (SQuAD 2.0's NoAns). |
| `unjustified_abstention_rate` | abstained / **answerable** items (a GT answer is present) | An answer was expected and the system refused — charged in recall. The *cause* is apportioned by retrieval evidence (`abstention_counts`: `all_chunks_irrelevant` / `relevant_chunk_present` / `relevance_unknown`), computed for free from `retrieved2answer`, which runs for abstained items regardless. The cause never softens the verdict. |
| `unwarranted_answer_rate` | answered / **unanswerable** items | Answered where no answer exists — the failure mode of systems that never shut up. Precision is 0 by necessity (there is no reference to check the claims against; no request is spent). |
| `answers_with_relevant_context` | per item, defined only when `claim_recall > 0` (a chunk entails a GT claim): 1 if the response has claims, 0 if it abstained; macro-averaged, printed as *answers when context is relevant* with its `(x of y items)` support | Did the generator talk when it had evidence. Higher is better; the complement is the tree's *refused with relevant chunks* count. |
| `abstains_without_relevant_context` | per item, defined only when `claim_recall = 0` (no chunk entails any GT claim): 1 if it abstained, 0 if it answered; macro-averaged, printed as *abstains when context is irrelevant* | Did the generator shut up on trash context. Higher is better under this project's reading; a system meant to answer from its own knowledge reads it the other way (see self-knowledge). Undefined, like its sibling, when `claim_recall` is null — no GT claims (the blank-GT items among them) or unjudged retrieval. |
| `extraction_error_rate` (+ per-side counts) | items with tooling failures / evaluated | The report is honest about its own tooling. These items are excluded from every quality metric — our parse failure must never masquerade as the evaluated system's abstention or hallucination. |
| `checker_failure_rate` | None verdicts / all issued checks | Checker reliability per run: a run with 4% failed judgments deserves less trust than one with 0.1%. Catches loud failures (parse/context); silent checker degradation (all-Entailment bias) is a known open problem — see the drawbacks backlog. |

Behavior-rate denominators exclude extraction-errored items (no-results
leave the denominator): a tooling failure is charged exactly once, in
`extraction_error_rate` — never by diluting a behavior rate.

**What an abstention costs** (docs/abstention.md §4): recall is judged
from the response text as the paper does — a refusal entails nothing, so
it reads 0 and F1 follows; precision and the faithfulness family have no
claims → `null`. Abstentions are never excluded from recall.

**The blank-GT convention**: mark an unanswerable question with an
explicit `"gt_answer": ""`. Field absent or `null` = missing GT (item
dropped); present and empty = *no answer exists*. An empty `response`
is likewise data (a full abstention), never a missing field.

## Repeated runs: `--runs N` (variance measurement)

Single-run numbers are point samples of a noisy process — a 25% vs 30% F1
across two runs decides nothing. `--runs N` (also on `faithcheck`,
`eval extractor`, `eval checker`) repeats the whole experiment N times and
reports `mean ± std [min, max]` per metric; the [min, max] band shows where
the results "stuck". N=3 is the floor, N=5 when a decision rides on it, and
the cost is N x the LLM bill. A `± 0.000` result against a deterministic
endpoint is a real finding, not a bug.

The feature belongs to the pipeline/evaluator (`runs` constructor
parameter) — the CLI only passes the number through. Mechanics:

- Every run starts from a pristine deep copy of the input, snapshotted
  BEFORE run 1 (run 1 mutates in place per the run() contract; a later
  copy of the mutated data would carry run 1's claims and verdicts, the
  skip logic would no-op runs 2..N, and the variance would read a fake
  zero).
- All N runs print symmetrically: a Run n/N header, progress bars, one
  summary line with duration. The full phase narrative belongs to
  `--runs 1` only; validation + config print once (they are run-invariant).
  One VARIANCE block and one cumulative token table land at the end.

The multi-run file (default `--runs 1` keeps today's document, byte-identical):

- `_meta` gains `runs` and total `duration_seconds`;
- `overall_metrics` stays **flat and means-valued** — variance-unaware
  readers parse a multi-run report like a single-run one;
- `variance` (new sibling): per metric `{std, min, max, values}`;
- `runs` (new sibling, the content): N complete, byte-normal N=1 reports —
  "a report is a report is a report", one parser reads everything, per-item
  run-vs-run comparison is the consumer's business and fully possible.
  There is no top-level `results` in a multi-run file.

## Known methodology limitations (documented, not hidden)

Beyond the metric definitions themselves: claim granularity acts as an
invisible denominator (duplicate facts double-weight recall; extraction
variance masquerades as metric variance), decontextualized triplets can be
unanswerable in isolation, context_precision is partly a chunking artifact,
and precision cannot distinguish hallucinated from true-but-not-in-GT
claims. These are properties of the RAGChecker methodology; the toolkit's
orthogonal axes (atomicity, duplicates, error rates) exist to expose rather
than blend them.
