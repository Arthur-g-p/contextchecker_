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

Computes standard multi-class classification metrics:
- Overall Accuracy
- Precision, Recall, and F1 for each class (`Entailment`, `Contradiction`, `Neutral`)
- Confusion Matrix

Metrics and configuration are returned as a `CheckerEvalResult`. Disagreements (instances where the predicted verdict did not match the human label) are saved to a `<filename>_disagreements.json` file for further error analysis.
