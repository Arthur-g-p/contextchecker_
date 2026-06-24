# Extractor Evaluator (`ExtractorEvaluator`)

The Extractor Evaluator assesses the quality of knowledge graph extraction by running the extraction live on a test dataset and then comparing the predicted triplets against ground-truth (GT) triplets. 

---

## Pipeline Overview

```
Input (list[dict])
  │
  ├─ _validate()  →  Drops invalid items (missing 'response'). Keeps items missing GT (for wrongful answer checks).
  ├─ extract      →  Runs `ExtractionService` (in quiet mode) to generate predicted triplets.
  ├─ _classify()  →  Sorts items into buckets: to_compare, correct_abstention, wrongful_abstention, wrongful_answer.
  ├─ Match (LLM)  →  2-pass batched semantic equivalence check for items in `to_compare`.
  └─ Metrics      →  Computes precision, recall, F1, and tracks abstention penalties.
```

## Step 1: Validation (`_validate`)

Ensures the data has the minimum required structure before attempting extraction.

- Keeps: Items with a `response`. (Missing GT is allowed and acts as a trap for hallucinated answers).
- Drops: Items missing `response`.

## Step 2: Extraction

Delegates to `ExtractionService` to extract triplets live using the configured `extractor_model`. Runs silently without printing progress bars to avoid cluttering the evaluator's output.

## Step 3: Classification (`_classify`)

After extraction, sorts the items into four buckets to handle cases where either the GT or predictions are missing.

1. `to_compare`: Both GT and predicted triplets exist. Sent to LLM matching.
2. `correct_abstention`: Neither GT nor predictions exist. (True Negatives).
3. `wrongful_abstention`: GT exists, but the model extracted nothing. Penalizes Recall (False Negatives).
4. `wrongful_answer`: No GT exists, but the model hallucinated predictions. Penalizes Precision (False Positives).

## Step 4: Matching (`_match_all_llm`)

For items in the `to_compare` bucket, evaluates extraction quality using a 2-pass LLM matching strategy:

1. **Pass 1 (Recall)**: Checks if every GT triplet is entailed by the predicted triplets. 
   - Uses `CheckingService` in `joint` mode to evaluate multiple GT claims against the predicted text chunk.
   - Missing GT triplets contribute to False Negatives (FN).

2. **Pass 2 (Precision)**: Checks if every predicted triplet is entailed by the GT triplets.
   - Reverses the roles: GT triplets become the reference, and predicted triplets are the claims.
   - Unsupported predictions contribute to False Positives (FP).

## Step 5: Metrics and Disagreements

Combines exact matches, LLM matching results, and abstention penalties into a final `ExtractorEvalResult` containing Precision, Recall, F1, and raw TP/FP/FN counts. 

Disagreements (False Positives and False Negatives) are saved to a separate `<filename>_disagreements.json` file for debugging.
