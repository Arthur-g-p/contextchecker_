# Extractor Evaluator (`ExtractorEvaluator`)

The Extractor Evaluator measures **extraction quality**: it runs extraction live
on a labelled dataset, then reconciles the predicted triplets against the
ground-truth (GT) triplets in both directions using an LLM checker.

There is no gold extraction to compare against — the GT triplets in
`eval_data/msmarco/` were human-*verdicted*, never proven complete (see
[eval_data/msmarco/README.md](../eval_data/msmarco/README.md)). So this is
reconciliation, not scoring against a key: each side has to find its
counterpart in the other.

> **Qualify your checker first.** Every number below is produced by an LLM
> checker making entailment judgements. A checker that fails to parse, or that
> collapses to answering "Entailment" for everything, produces extractor scores
> that are partial or meaningless. Run [`eval checker`](eval_checker.md) on the
> same model before trusting `eval extractor` output, and read the
> `checker_failures` block in every report.

---

## Pipeline Overview

```
Input (list[dict])
  │
  ├─ _canonicalize_keys() →  Normalizes key aliases in-place ('context' → 'reference')
  ├─ _validate()   →  Drops items without a response. Canonicalizes GT triplets.
  ├─ extract       →  Runs `ExtractionService` (quiet) → predicted triplets
  ├─ 2b atomicity  →  OPTIONAL orthogonal axis, measured on a deep copy
  ├─ 2c duplicates →  Orthogonal axis, read-only
  ├─ _classify()   →  Five buckets: to_compare, abstention_recognized,
  │                   answer_missed, abstention_misread, extraction_error
  ├─ Match (LLM)   →  2-pass entailment check over the `to_compare` items
  └─ Metrics       →  Precision / Recall / F1 + orthogonal axes + tooling rates
```

Three families of number come out, and they are never mixed:

| family | what it measures | examples |
| --- | --- | --- |
| **Coverage** | how well the extractor found the facts | precision, recall, f1 |
| **Orthogonal axes** | independent properties of the extraction | atomicity, duplicates, abstention behavior |
| **Reliability** | how well *this evaluator* worked | extraction errors, checker failures, atomization failures |

Tooling failures are excluded from every quality metric. Our parse failure must
never be reported as the extractor's mistake.

## Step 1: Validation (`_validate`)

- **Keeps** items with a `response` and a GT key. Absent or null GT key
  = missing data (dropped, reported); an explicit empty list is data —
  nothing to extract: the response abstained, and that is what makes a
  misread abstention detectable.
- **Drops** items without a `response`; there is nothing to extract from.
- **Canonicalizes** key aliases (`context` → `reference`) and GT triplets
  (`{"triplet": [s, p, o]}` → `{"subject", "predicate", "object"}`).
- **Fails fast** (`InvalidInputError`) if no item survives, or if *no* item
  carries the `--gt-key` at all — that means the wrong file or the wrong key,
  and it is caught before a single LLM call is paid for.

GT presence is judged on the **key**, not its contents: an item carrying
`"claude2_response_kg": []` is a deliberate abstention trap, not missing data.

## Step 2: Extraction

Delegates to `ExtractionService` with the configured `--extractor-model`,
running quiet (the evaluator owns the logging). Predictions are always written
under `{extractor_model}_response_kg` — see the collision guard below.

Deduplication is **off**: the eval measures duplicates as an independent axis,
so predictions must stay raw. The eval never runs on deduplicated data.

## Step 2b: Atomicity axis (optional, orthogonal)

Runs the atomizer over the predictions to ask how *atomic* they are: does one
triplet carry exactly one fact? Measured on a **deep copy**, so coverage
matching and the output file always see the raw predictions.

Requires `--atomizer-model` **and** `ATOMIZER_API_KEY`. If either is missing the
axis is skipped and the console says why (`⏭️ Atomicity skipped (...)`); the
`atomicity` field is then `null`.

- `atomicity_rate` — kept claims / evaluated claims (1.0 = every triplet atomic)
- `information_density` — atomic units / evaluated claims (facts per triplet)
- Full split detail rides on its item in the record (`atomicity_splits`).

## Step 2c: Duplicate axis (orthogonal, read-only)

Counts exact (string-equal) duplicate claims in the predictions. Never mutates
anything. Reported as `duplicate_rate` plus the offending triplets per item.

## Step 3: Classification (`_classify`)

After extraction, every item lands in exactly one of **five** buckets. The
abstention is always the *response's* (GT has no claims because the response
asserted nothing); the extractor never abstains — it either *recognizes* an
abstaining response or *misreads* it into invented claims. This differs from
`ragcheck`, whose abstention judges are about the question.

| bucket | GT | predictions | effect |
| --- | --- | --- | --- |
| `to_compare` | yes | yes | sent to LLM matching |
| `abstention_recognized` | no | no | abstaining response, nothing extracted — correct, no penalty |
| `answer_missed` | yes | no | answering response, nothing extracted — every GT claim charged as a recall miss |
| `abstention_misread` | no | yes | abstaining response, claims invented — every predicted claim charged as a precision miss |
| `extraction_error` | any | — | tooling failure → **excluded from all metrics**, counted in `extraction_errors` |

The `extraction_error` bucket exists so a parse failure can never masquerade as
an abstention. An empty prediction list after a crashed extraction says nothing
about the extractor's willingness to answer.

## Step 4: Matching (`_match_all_llm`)

For `to_compare` items, two independent passes through `CheckingService` in
`joint` mode with the `checker_prompt_eval_joint` prompt:

1. **Pass 1 — GT → Pred (recall).** GT triplets are the claims, predictions are
   the reference. Entailment = the extractor found this fact.
2. **Pass 2 — Pred → GT (precision).** Roles reversed. Entailment = this
   prediction is supported by ground truth.

Two passes, two independent counts — following RAGChecker, there is **no shared
TP and no `min()`**.

### Three verdict outcomes, not two

Each claim comes back as exactly one of:

| verdict | meaning | effect on metrics |
| --- | --- | --- |
| `Entailment` | counterpart found | numerator **and** denominator |
| `Contradiction` / `Neutral` | judged, no counterpart | denominator only (a real miss) |
| `None` | the checker never returned a verdict (parse failure after all retries) | **neither** — the claim leaves the ratio entirely |

A claim nobody judged is not evidence about the extractor. Unjudged claims are
excluded from both sides of the fraction, counted in `checker_failures`, and
listed per item in the findings file as `unjudged`. This mirrors the
None-propagation rule `ragcheck` follows.

Note the sampling bias this leaves behind: bundles do not fail at random, so a
high `checker_failure_rate` means the numbers are not merely less precise, they
may be biased. That is why the console prints a "run `eval checker`" pointer
whenever the rate is above zero.

## Step 5: Metrics

Both ratios are computed from **exhaustive partitions** of the issued claims —
every claim of a side lands in exactly one bucket:

```
recall_counts:
  total_gt_claims = covered + missed + answer_missed_penalty + unjudged
  denominator     = total_gt_claims - unjudged
  recall          = covered / denominator

precision_counts:
  total_pred_claims = supported + unsupported + abstention_misread_penalty + unjudged
  denominator       = total_pred_claims - unjudged
  precision         = supported / denominator

f1 = harmonic mean of precision and recall
```

The penalties are *synthetic*: no checker ever saw those claims; the item was
excluded wholesale and its claims charged. Keeping them separate from judged
misses is the point of the partition — "the extractor missed 48 claims" reads
very differently when it is 12 real misses plus 36 claims from 5 skipped items.

**The null rule.** A metric whose denominator is zero is `null`, never `0.0`.
Nothing was judged, and "not computable" is not a score. `f1` is `null` when
either side is. A `0.0` therefore always means a real, measured zero.

**Tooling rates** are reported alongside but never mixed in:

- `extraction_errors.rate` — items whose extraction failed / all extracted items
- `checker_failures.rate` — verdicts the checker never returned / verdicts issued
  (penalty claims were never sent, so they do not count as issued)

---

## Running the Evaluator (CLI)

```bash
claimlens eval extractor <input_file> [options]
```

> Requires **two** models: one performs the live extraction
> (`--extractor-model`), one acts as the checker in the 2-pass matching
> (`--checker-model`). They may be the same model — but then the thing being
> measured and the ruler measuring it are correlated. Both flags are required.

### Arguments and Options

| Parameter | Flag | Default | Description |
|---|---|---|---|
| **Input File** | `input_file` *(argument)* | *Required* | Eval JSON: items with a `response` and GT triplets. Positional — no flag in front of it. |
| **Extractor Model** | `--extractor-model`, `-e` | *Required* | Model used to extract triplets live. Also the prefix of the predicted-triplet key. |
| **Checker Model** | `--checker-model` | *Required* | LLM checker for the 2-pass matching. |
| **Extractor API Base** | `--extractor-base-api` | `None` | Base URL for the extractor (OpenAI-SDK route). Unset → LiteLLM provider routing. |
| **Checker API Base** | `--checker-base-api` | `None` | Base URL for the checker. |
| **GT Key** | `--gt-key` | `claude2_response_kg` | Item key holding the ground-truth triplets. |
| **Atomizer Model** | `--atomizer-model` | `None` | Enables the atomicity axis. Needs `ATOMIZER_API_KEY`; silently skipped if unset. |
| **Atomizer API Base** | `--atomizer-base-api` | `None` | Base URL for the atomizer. |
| **Output File** | `--output`, `-o` | `results/{input_stem}_extractor_eval[_{runs}].json` | Summary JSON. The `_{runs}` suffix appears only for `--runs > 1`. A `*_disagreements.json` is written alongside. |
| **Joint Bundle Size** | `--joint-num` | `10` | Max claims bundled per joint matching call. |
| **Word Budget** | `--max-words` | `None` | Word budget per matching call. |
| **Runs** | `--runs` | `1` | Repeat the whole eval N times and report variance (N × LLM cost). |
| **Concurrency** | `--concurrency` | `10` | Max simultaneous LLM requests per client. |
| **Debug Mode** | `--debug` | `False` | Timestamps + module names in logs. |

> **The predicted-triplet key is derived, not configurable.** Live extraction
> always writes under `{extractor_model}_response_kg`, and that is where the
> evaluator reads predictions from. If that key would equal `--gt-key` (e.g.
> `--extractor-model claude2` with the default GT key) the evaluator raises
> `InvalidInputError` immediately — otherwise extraction would overwrite the GT
> slot and the eval would match ground truth against itself, reporting a
> flawless and completely fictional score.

### Required environment (`.env`)

| Variable | Used for |
|---|---|
| `EXTRACTOR_API_KEY` | Auth for the extractor LLM endpoint. |
| `CHECKER_API_KEY` | Auth for the checker/matching LLM endpoint. |
| `ATOMIZER_API_KEY` | Auth for the atomizer. Without it the atomicity axis is skipped. |
| `LLM_TIMEOUT` *(optional)* | Per-call timeout in seconds (default `120.0`). |

### Worked example

```bash
claimlens eval extractor eval_data/msmarco/msmarco_gpt4_5.json \
  --extractor-model gemini-3.1 \
  --checker-model gemini-3.1 \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
```

Reads GT from `claude2_response_kg`, extracts live into
`gemini-3.1_response_kg`, runs the 2-pass match, then writes
`eval_data/msmarco/results/msmarco_gpt4_5_extractor_eval.json` plus its
`_findings.json` sibling. Output always lands in a `results/` directory
beside the input, so an input already inside `results/` nests one level deeper —
pass `-o` to control this.

---

## Output Format

Two files are written. The **record** holds everything: metrics, every count
the console prints, and the complete per-item claim lists. The **findings**
file is the review queue — only what went wrong, each entry tagged with a
`kind` — derived from the record's items and never a source of its own. Both
open with `_args` (what you asked for) and `_meta` (what the run turned out
to be; see docs/output_conventions.md).

The skeleton is the same at `--runs 1` and `--runs N`: `runs` is always a
list, `metrics` is the mean over runs (the run's own values at N = 1), and
`variance` the spread. `--runs` adds entries, it never reshapes.

### Record — `<filename>_extractor_eval[_N].json`

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "...", "pred_key": "gemini-3.1_response_kg", "matching": "llm-2-pass", "runs": 1 },
  "metrics": {
    "recall": 0.871, "precision": 0.991, "f1": 0.927,
    "atomicity_rate": 1.0, "claim_density": 1.0, "duplicate_rate": 0.0,
    "abstention_recognized_rate": 0.0, "abstention_misread_rate": 1.0, "answer_missed_rate": 0.0,
    "extraction_error_rate": 0.0, "checker_failure_rate": 0.0, "atomization_failure_rate": 0.0
  },
  "variance": { "recall": { "n": 1, "std": 0.0, "min": 0.871, "max": 0.871, "values": [0.871] }, "...": "..." },
  "runs": [
    {
      "_meta": { "...": "...", "run": 1, "duration_seconds": 90.6 },
      "metrics": { "...same keys, this run's values..." },
      "counts": {
        "data":      { "dropped_no_response": 0, "dropped_no_gt_key": 0, "gt_empty": 1 },
        "recall":    { "total_gt_claims": 116, "covered": 101, "missed": 15, "answer_missed_penalty": 0, "unjudged": 0, "denominator": 116 },
        "precision": { "total_pred_claims": 105, "supported": 104, "unsupported": 0, "abstention_misread_penalty": 1, "unjudged": 0, "denominator": 105 },
        "extraction_stats": { "gt": { "claims": 116, "avg_per_item": 4.83 }, "pred": { "claims": 105, "avg_per_item": 4.38 } },
        "abstention_handling": { "answers": 24, "answers_extracted": 24, "answers_missed": 0,
                                 "abstentions": 1, "abstentions_recognized": 0, "abstentions_misread": 1 },
        "reliability": {
          "extraction":       { "failed": 0, "items": 25, "by_cause": {} },
          "matching_checker": { "unjudged": 0, "issued": 221, "items_affected": 0, "unjudged_gt": 0, "unjudged_pred": 0 },
          "atomization":      { "measured": false, "reason": "no --atomizer-model" }
        },
        "atomicity":  null,
        "duplicates": { "predicted_claims": 105, "unique_claims": 105, "duplicate_claims": 0 }
      },
      "items": [
        { "id": "12345", "question": "...", "response": "...", "bucket": "compared",
          "gt_claims":   [ { "claim": "...", "verdict": "Entailment", "explanation": "..." },
                           { "claim": "...", "verdict": "Neutral",    "explanation": "..." } ],
          "pred_claims": [ { "claim": "...", "verdict": "Entailment", "explanation": "..." } ] },
        { "id": "1088355", "question": "...", "response": "The passages do not provide ...",
          "bucket": "abstention_misread", "gt_claims": [], "pred_claims": [ { "claim": "..." } ] }
      ]
    }
  ]
}
```

- `metrics` is the variance roster: the numbers the console's Metrics
  rows and the run line show. Rates live here and nowhere else.
- `counts` mirrors the console blocks one to one: 📂 Data, the two
  🔎 Matching Quality funnels, 📊 Extraction Stats, ⚪ Abstention Handling,
  the three 💥 Reliability rows, 🧬 Atomicity and 🔁 Duplicates (numbers
  only). An axis that did not run says so: `"measured": false` with the
  reason, and `atomicity: null`.
- `items` lists every attempted item with its `bucket` (`compared`,
  `answer_missed`, `abstention_recognized`, `abstention_misread`,
  `extraction_error`) and both claim lists. Compared items carry a verdict
  and explanation per claim (pass 1 on GT claims, pass 2 on predictions);
  the other buckets never reached the matcher, so their claims are bare.
  Split decisions (`atomicity_splits`) and `duplicate_claims` ride on the
  item they belong to; an errored item carries its `cause`.

### Findings — `<filename>_extractor_eval[_N]_findings.json`

The console branches opened up: one list per branch, every entry names its
item. Empty branches stay present — hidden is not zero.

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "..." },
  "runs": [
    { "_meta": { "...": "...", "run": 1 },
      "findings": {
        "missed":             [ { "id": "12345", "question": "...", "claim": "...", "verdict": "Neutral", "explanation": "..." } ],
        "unsupported":        [ { "id": "12345", "question": "...", "claim": "...", "verdict": "Neutral", "explanation": "..." } ],
        "answer_missed":      [ { "id": "77", "question": "...", "response": "...", "claims": ["..."] } ],
        "abstention_misread": [ { "id": "1088355", "question": "...", "response": "...", "claims": ["..."] } ],
        "unjudged":           [ { "id": "...", "question": "...", "claim": "...", "side": "gt", "cause": "checker_failure" } ],
        "extraction_failed":  [ { "id": "857956", "question": "...", "cause": "parse_failure" } ]
      } }
  ]
}
```

- `missed` / `unsupported` — the Matching Quality misses, each with the
  **judged** verdict and the checker's explanation.
- `unjudged` — a claim the checker never returned a verdict for (`side`
  names the pass). Not a disagreement, but the item stays traceable.
- `answer_missed` / `abstention_misread` — the ⚪ Abstention Handling
  branches: the GT claims lost to an empty extraction, the claims invented
  from a refusal.
- `extraction_failed` — the 💥 row's items, with their cause.
- `_meta` is a copy of the record's, so the two files identify each other.

With `--runs N`, the variance block aggregates the twelve `metrics` keys as
`mean ± std [min, max]` across runs (see docs/output_conventions.md rule
set 4); `variance[key].n` says how many runs contributed, and an axis that
never ran keeps its key with `n: 0`.

---

## Reading the console output

```
 🔎 Matching Quality  (LLM 2-pass)
    Recall — 27 total GT claims
     ├─ ✅ 15 covered by predictions  (judged)
     ├─ ❌ 0 missed  (judged)
     ├─ ⚪ 12 answer-missed penalty  (1 items, 0 predictions for 12 claims)
     ├─ 💥 0 unjudged by checker
     └─ → Recall 0.556  (15 / 27)
```

Each funnel accounts for every issued claim: the first three branches sum to the
denominator, and the 💥 branch sits explicitly outside it. The same partition is
in the JSON, so print and file can never disagree.

Below the funnels, the orthogonal sections always print (a hidden row is
indistinguishable from an unmeasured one): `⚪ Abstention Handling` (two
response kinds with their outcomes as sub-branches — answers: extracted /
nothing extracted; abstentions: recognized / misread — footer rates
`abstention recognized` and `abstention misread` over the abstaining
responses, `answer missed` over the answering ones), `💥 Reliability` (extraction + matching checker
+ atomization, each row `count of denominator → rate_key rate`; an
unconfigured axis prints `not measured`), `🧬 Atomicity` and
`🔁 Duplicates`. See docs/output_conventions.md for the display rules.
