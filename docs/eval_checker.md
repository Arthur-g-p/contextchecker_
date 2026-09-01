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
  └─ Metrics        →  Computes Accuracy, per-class F1/Precision/Recall, and a confusion matrix.
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
- Collects false positives and false negatives for disagreement logging.

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

Metrics and configuration are returned as a `CheckerEvalResult`. Disagreements (instances where the predicted verdict did not match the human label) are saved to a `<filename>_disagreements.json` file for further error analysis.

With `--runs N`, the variance block aggregates accuracy, macro F1, the
per-label F1s, and `checker_failure_rate` as `mean ± std [min, max]`
across runs (see docs/output_conventions.md rule set 4).
