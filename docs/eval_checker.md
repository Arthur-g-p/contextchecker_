# Checker Evaluator (`CheckerEvaluator`)

The Checker Evaluator measures the entailment classification accuracy of the `CheckingService`. It takes ground-truth (GT) triplets annotated with a `human_label` (e.g., Entailment, Contradiction, Neutral) and runs them through the `CheckingService` to predict their verdicts against a reference passage.

---

## Pipeline Overview

```
Input (list[dict])
  │
  ├─ _validate()    →  Filters for items that have both a reference and GT triplets with `human_label`.
  ├─ _strip()       →  Removes any existing verdicts from the GT triplets to force a full recompute.
  ├─ Check          →  Delegates to `CheckingService` to predict verdicts (Joint or Single mode).
  ├─ _compare()     →  Compares predicted verdicts 1:1 against the `human_label`s.
  ├─ Metrics        →  Computes Accuracy, per-class F1/Precision/Recall, and a confusion matrix.
  └─ Disagreements  →  Collects every wrong or unjudged claim per item into a second document.
```

## Step 1: Validation (`_validate`)

Ensures the data has the required ground-truth fields for evaluation.

- Drops: Items missing `reference`.
- Drops: Items missing the specified `gt_key`.
- Drops: Any individual triplet that lacks a `human_label`. (Items with zero labeled triplets are dropped entirely).

## Step 2: Stripping (`_strip_existing_verdicts`)

The `CheckingService` skips claims that already have a verdict to save API costs. Because the evaluation relies on measuring the model's performance, the evaluator strips all existing verdicts and reasons from the GT triplets before checking them, ensuring a clean slate.

## Step 3: Checking

Delegates execution to the `CheckingService` using the configured `checker_model` and `joint_num` settings. 
The service calls the LLM, parses the entailment verdict out of the response, and populates the `verdict` and `reason` fields on the triplets in-place.

## Step 4: Comparison (`_compare_verdicts`)

Iterates over every validated triplet and directly compares the newly predicted `verdict` against its original `human_label`. 

- Collects matched triplets into lists for `y_true` (ground truth) and `y_pred` (predictions).
- Claims with no verdict (checker failure) are excluded from both lists and counted as unjudged.

## Step 5: Metrics and Disagreements

Computes standard multi-class classification metrics, reported as:

- **🔎 Verdicts tree** — correct / wrong / unjudged over the labeled
  claims, with accuracy derived in the footer (`→ accuracy 0.803
  (102 / 127)`; the denominator shrinks to the judged claims when the
  checker returned no verdict for some — marked `judged`). An ℹ️ line
  states the majority-class baseline: on an imbalanced slice a constant
  answer scores high, and accuracy below that baseline means the checker
  underperforms a constant classifier.
- **📊 Per-Label report** — precision / recall / F1 / total per class,
  plus macro and weighted averages. `macro_f1` (classes weighted
  equally) is the honest headline on imbalanced slices and is surfaced
  top-level in the JSON alongside per-label F1s (`entailment_f1`,
  `contradiction_f1`, `neutral_f1`; a zero-support label reports `null`,
  never 0.0). Zero-support labels are flagged — their zeros dilute the
  macro average.
- **📉 Confusion Matrix** — with marginal totals on both axes; the
  corner total reconciles with the Verdicts tree (correct + wrong).
  The off-diagonals are the checker's character: `Entailment → Neutral`
  is false skepticism, `Neutral → Entailment` false credulity.
- **💥 Reliability** — the checker under test failing to produce a
  verdict is a finding about the checker: excluded from accuracy,
  charged exactly once here (`N of M claims unjudged →
  checker_failure_rate`).

Two files are written. The **record** holds everything: metrics, every
count the console prints, and the complete per-item claim list. The
**findings** file is the review queue — only the claims worth a look, each
tagged with a `kind` — derived from the record's items and never a source
of its own. On a heavily skewed slice a handful of wrong human labels moves
the accuracy more than the checker does, so the findings are also where the
labels themselves get reviewed.

## Output Format

Both files open with `_args` and `_meta` (see docs/output_conventions.md).
The skeleton is the same at `--runs 1` and `--runs N`: `runs` is always a
list, `metrics` is the mean over runs (the run's own values at N = 1), and
`variance` the spread. `--runs` adds entries, it never reshapes.

### Record — `<filename>_checker_eval[_N].json`

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "...", "runs": 1 },
  "metrics":  { "accuracy": 0.845, "macro_f1": 0.388, "entailment_f1": 0.913,
                "contradiction_f1": null, "neutral_f1": 0.25, "checker_failure_rate": 0.0 },
  "variance": { "accuracy": { "n": 1, "std": 0.0, "min": 0.845, "max": 0.845, "values": [0.845] }, "...": "..." },
  "runs": [
    {
      "_meta": { "...": "...", "run": 1, "duration_seconds": 13.5 },
      "metrics": { "...same keys, this run's values..." },
      "counts": {
        "data":     { "dropped_no_gt_claims": 1, "dropped_no_reference": 0, "dropped_no_labels": 0, "unlabeled_claims": 0 },
        "verdicts": { "labeled": 116, "correct": 98, "wrong": 18, "unjudged": 0 },
        "labels":   { "Entailment": 113, "Contradiction": 0, "Neutral": 3 },
        "per_label": { "Entailment": { "precision": 1.0, "recall": 0.841, "f1": 0.913, "total": 113 },
                       "...": "...", "macro avg": { "..." }, "weighted avg": { "..." } },
        "confusion_matrix": { "labels": ["Entailment", "Contradiction", "Neutral"], "matrix": [[95, 0, 18], [0, 0, 0], [0, 0, 3]] }
      },
      "items": [
        { "id": "1006506", "question": "...", "response": "...",
          "claims": [ { "claim": "Mount Nyiragongo last erupted on January 17, 2002",
                        "human_label": "Entailment", "verdict": "Neutral",
                        "explanation": "The reference states that an eruption occurred in 2002 but ..." } ] }
      ]
    }
  ]
}
```

- `metrics` is the variance roster: the numbers the console's Metrics
  rows and the run line show. Nothing else is varianced.
- `counts` mirrors the console blocks one to one: 📂 Data, 🔎 Verdicts,
  the label distribution (its largest share is the majority baseline), the
  📊 Per-Label Report cell for cell, the 📉 Confusion Matrix.
- `items` lists every evaluated item with every labeled claim. A claim
  without a verdict carries `"error": <cause>`.
- The reference is not repeated (it can be a full retrieved context per
  item); join back to the input on `id`.

### Findings — `<filename>_checker_eval[_N]_findings.json`

The 🔎 Verdicts branches opened up: one list per branch, every entry names
its item. Empty branches stay present — hidden is not zero.

```json
{
  "_args": { "...": "..." },
  "_meta": { "...": "..." },
  "runs": [
    { "_meta": { "...": "...", "run": 1 },
      "findings": {
        "wrong":    [ { "id": "1006506", "question": "when did mount nyiragongo last erupted",
                        "claim": "Mount Nyiragongo last erupted on January 17, 2002",
                        "human_label": "Entailment", "verdict": "Neutral",
                        "explanation": "The reference states that an eruption occurred in 2002 but ..." } ],
        "unjudged": [ { "id": "...", "question": "...", "claim": "...", "human_label": "...", "cause": "timeout" } ]
      } }
  ]
}
```

- `wrong` — the checker's explanation is the text to read when deciding
  whether the checker or the annotator erred.
- `unjudged` — no verdict; `cause` from the checker error marker. Not a
  disagreement, but the item stays traceable.
- `_meta` is a copy of the record's, so the two files identify each other.

With `--runs N`, the variance block aggregates accuracy, macro F1, the
per-label F1s, and `checker_failure_rate` as `mean ± std [min, max]`
across runs (see docs/output_conventions.md rule set 4).
