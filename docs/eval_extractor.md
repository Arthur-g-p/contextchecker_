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
  ├─ _classify()   →  Five buckets: to_compare, justified_abstention,
  │                   unjustified_abstention, wrongful_answer, extraction_error
  ├─ Match (LLM)   →  2-pass entailment check over the `to_compare` items
  └─ Metrics       →  Precision / Recall / F1 + orthogonal axes + tooling rates
```

Three families of number come out, and they are never mixed:

| family | what it measures | examples |
| --- | --- | --- |
| **Coverage** | how well the extractor found the facts | precision, recall, f1 |
| **Orthogonal axes** | independent properties of the extraction | atomicity, duplicates, abstention behavior |
| **Eval tooling failures** | how well *this evaluator* worked | extraction errors, checker failures |

Tooling failures are excluded from every quality metric. Our parse failure must
never be reported as the extractor's mistake.

## Step 1: Validation (`_validate`)

- **Keeps** items with a `response`. Missing GT is *not* an error — it is the
  trap that catches hallucinated extractions (`wrongful_answer`).
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
- Full split detail goes to the disagreements file, not the summary.

## Step 2c: Duplicate axis (orthogonal, read-only)

Counts exact (string-equal) duplicate claims in the predictions. Never mutates
anything. Reported as `duplicate_rate` plus the offending triplets per item.

## Step 3: Classification (`_classify`)

After extraction, every item lands in exactly one of **five** buckets. The
abstention vocabulary matches the `ragcheck` pipeline: an abstention is
*justified* when there was nothing to find, *unjustified* when there was.

| bucket | GT | predictions | effect |
| --- | --- | --- | --- |
| `to_compare` | yes | yes | sent to LLM matching |
| `justified_abstention` | no | no | correct silence — no penalty |
| `unjustified_abstention` | yes | no | every GT claim charged as a recall miss |
| `wrongful_answer` | no | yes | every predicted claim charged as a precision miss |
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
listed per item in the disagreements file under `unjudged`. This mirrors the
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
  total_gt_claims = covered + missed + unjustified_abstention_penalty + unjudged
  denominator     = total_gt_claims - unjudged
  recall          = covered / denominator

precision_counts:
  total_pred_claims = supported + unsupported + wrongful_answer_penalty + unjudged
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
contextchecker eval extractor <input_file> [options]
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
contextchecker eval extractor eval_data/msmarco/msmarco_gpt4_5.json \
  --extractor-model gemini-3.1 \
  --checker-model gemini-3.1 \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
```

Reads GT from `claude2_response_kg`, extracts live into
`gemini-3.1_response_kg`, runs the 2-pass match, then writes
`eval_data/msmarco/results/msmarco_gpt4_5_extractor_eval.json` plus its
`_disagreements.json` sibling. Output always lands in a `results/` directory
beside the input, so an input already inside `results/` nests one level deeper —
pass `-o` to control this.

---

## Output Format

Two files are written. Both open with `_args` (what you asked for — every CLI
parameter, plus `_explicit` naming the ones actually typed) and `_meta` (what
the run turned out to be — derived keys, counts, timings).

### Summary file (single run)

```json
{
  "_args": {
    "command": "eval extractor",
    "input_file": "eval_data/msmarco/msmarco_gpt4_5.json",
    "extractor_model": "meta/muse-glimmer-30b",
    "checker_model": "meta/muse-glimmer-30b",
    "atomizer_model": "openai/gpt-5.6-luna",
    "gt_key": "claude2_response_kg",
    "joint_num": 10,
    "runs": 1,
    "concurrency": 10,
    "_explicit": ["checker_model", "extractor_model", "input_file"]
  },
  "_meta": {
    "schema_version": 4,
    "report_type": "extractor_eval",
    "contextchecker_version": "0.5.0",
    "timestamp": "2026-08-19T17:15:05",
    "duration_seconds": 89.8,
    "total_items": 5,
    "evaluated_items": 2,
    "dropped_items": 3,
    "pred_key": "meta/muse-glimmer-30b_response_kg",
    "matching": "llm-2-pass"
  },

  "precision": 0.9474,
  "recall": 0.5556,
  "f1": 0.7,

  "recall_counts": {
    "total_gt_claims": 27,
    "covered": 15,
    "missed": 0,
    "unjustified_abstention_penalty": 12,
    "unjudged": 0,
    "denominator": 27
  },
  "precision_counts": {
    "total_pred_claims": 19,
    "supported": 18,
    "unsupported": 0,
    "wrongful_answer_penalty": 1,
    "unjudged": 0,
    "denominator": 19
  },

  "total_items": 5,
  "to_compare_items": 2,
  "gt_stats":   { "total_triplets": 27, "avg_per_item": 13.5 },
  "pred_stats": { "total_triplets": 19, "avg_per_item": 9.5 },

  "abstentions": { "justified": 0, "unjustified": 1, "wrongful_answer": 1 },

  "checker_failures": {
    "count": 0,
    "issued_verdicts": 33,
    "rate": 0.0,
    "items_affected": 0,
    "unjudged_gt": 0,
    "unjudged_pred": 0
  },
  "extraction_errors": { "count": 1, "rate": 0.2, "by_cause": { "parse_failure": 1 } },

  "atomicity": {
    "extracted_claims": 19, "evaluated_claims": 19, "atomic_units": 19,
    "new_claims_from_splits": 0, "non_atomic": 0, "failed": 0,
    "atomicity_rate": 1.0, "information_density": 1.0
  },
  "duplicates": {
    "predicted_claims": 19, "unique_claims": 19, "duplicate_claims": 0,
    "duplicate_rate": 0.0, "items": []
  }
}
```

`precision`, `recall` and `f1` are `float | null` (the null rule).
`atomicity` is `null` when the axis was skipped. `duplicates` is `null` when
there were no predictions at all.

`_meta.dropped_items` counts items that never reached matching — note that this
mixes two very different things (a tooling failure and a meaningful abstention);
read `extraction_errors` and `abstentions` for the split.

### Summary file (`--runs > 1`)

```json
{
  "_args": { "...": "...", "runs": 3 },
  "_meta": { "...": "...", "runs": 3, "duration_seconds": 215.2 },

  "precision": 0.9425,
  "recall": 0.358,
  "f1": 0.6841,

  "variance": {
    "precision": { "n": 3, "std": 0.0498, "min": 0.8947, "max": 1.0,
                   "values": [0.9474, 0.8947, 1.0] }
  },
  "runs": [ { "...": "one complete, byte-normal single-run document" } ]
}
```

- Top-level metrics become **means**, so a variance-unaware reader parses a
  multi-run file like a single-run one.
- `variance` carries `{n, std, min, max, values}` per metric. `n` is how many
  runs actually contributed — a metric that was `null` in some runs is averaged
  over fewer, and `n` is where you see that. A metric that was `null` in *every*
  run keeps its key with `mean: null` and `n: 0`; it never silently disappears.
- `runs` holds N complete, normal single-run documents.

> **Known limitation.** Only top-level scalars are aggregated, so the multi-run
> top level currently carries the three coverage metrics but not the orthogonal
> rates (atomicity, duplicates, abstentions) or the tooling rates — those live
> inside each entry of `runs`. `ragcheck` does not have this limitation because
> it aggregates a flat `overall_metrics` dict. Unifying the two is planned.

### Disagreements file

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "..." },
  "total_disagreements": 3,
  "items": [
    {
      "id": "857956",
      "question": "...",
      "response": "...",
      "error_type": "extraction_error",
      "cause": "parse_failure"
    },
    {
      "id": "12345",
      "question": "...", "response": "...",
      "tp_recall": 6, "tp_precision": 4, "fp": 1, "fn": 2,
      "gt_triplets": ["..."], "pred_triplets": ["..."],
      "false_negatives": [
        { "gt_triplet": "...", "verdict": "Neutral", "reason": "..." }
      ],
      "false_positives": [
        { "pred_triplet": "...", "verdict": "Neutral", "reason": "..." }
      ],
      "unjudged": [
        { "gt_triplet": "...", "cause": "checker_failure" }
      ]
    }
  ],
  "atomicity_splits": [
    { "id": "...", "original": "...", "children": ["...", "..."], "reasoning": "..." }
  ]
}
```

- `error_type` is present only on whole-item entries and is one of
  `extraction_error`, `wrongful_answer`, `unjustified_abstention`.
- `false_positives` / `false_negatives` contain **judged** misses only — every
  entry carries a real verdict from the checker.
- `unjudged` holds claims the checker never returned a verdict for. An item
  whose *only* problem was a checker failure still appears here, so the failure
  stays traceable.
- `atomicity_splits` appears only when the atomicity axis ran and found
  something to split.
- With `--runs > 1` the document becomes `{_args, _meta, runs: [...]}`, mirroring
  the summary.

---

## Reading the console output

```
 🔎 Matching Quality  (LLM 2-pass)
    Recall — 27 total GT claims
     ├─ ✅ 15 covered by predictions  (judged)
     ├─ ❌ 0 missed  (judged)
     ├─ ⚪ 12 unjustified-abstention penalty  (1 items, 0 predictions for 12 claims)
     ├─ 💥 0 unjudged by checker
     └─ → Recall 0.556  (15 / 27)
```

Each funnel accounts for every issued claim: the first three branches sum to the
denominator, and the 💥 branch sits explicitly outside it. The same partition is
in the JSON, so print and file can never disagree.

Below the funnels, three orthogonal sections appear when they have something to
report — `💥 Eval Tooling Failures` (extraction + checker, the eval's own
reliability), `⚪ Abstention Behavior` (justified / unjustified / wrongful
answer), `🧬 Atomicity` and `🔁 Duplicates`.
