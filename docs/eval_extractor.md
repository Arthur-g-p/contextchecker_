# Extractor Evaluator (`ExtractorEvaluator`)

The Extractor Evaluator assesses the quality of knowledge graph extraction by running the extraction live on a test dataset and then comparing the predicted triplets against ground-truth (GT) triplets. 

---

## Pipeline Overview

```
Input (list[dict])
  │
  ├─ _canonicalize_keys() →  Normalizes key aliases in-place ('context' → 'reference')
  ├─ _validate()  →  Drops invalid items. Canonicalizes existing Ground Truth triplets.
  ├─ extract      →  Runs `ExtractionService` (in quiet mode) to generate predicted triplets.
  ├─ _canonicalize_preds() → Canonicalizes the newly generated predicted triplets.
  ├─ _classify()  →  Sorts items into buckets: to_compare, correct_abstention, wrongful_abstention, wrongful_answer.
  ├─ Match (LLM)  →  2-pass batched semantic equivalence check for items in `to_compare`.
  └─ Metrics      →  Computes precision, recall, F1, and tracks abstention penalties.
```

## Step 1: Validation (`_validate`)

Ensures the data has the minimum required structure before attempting extraction, and normalizes inputs to canonical format.

- Keeps: Items with a `response`. (Missing GT is allowed and acts as a trap for hallucinated answers).
- Drops: Items missing `response`.
- Canonicalizes: Key aliases (`context` -> `reference`) and any existing Ground Truth triplets (`{"triplet": [s,p,o]}` -> `{"subject", "predicate", "object"}`).

## Step 2: Extraction

Delegates to `ExtractionService` to extract triplets live using the configured `extractor_model`. Runs silently without printing progress bars to avoid cluttering the evaluator's output. After extraction, any newly predicted triplets are also forced through `canonicalize_triplets` to guarantee a uniform format for the downstream LLM judge.

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

---

## Running the Evaluator (CLI)

The evaluator is exposed as the **`extractor`** command inside the **`eval`** subcommand group:

```bash
contextchecker eval extractor <input_file> [options]
```

> **Note:** This requires **two** models — one to perform the live extraction (`--extractor-model`) and one to act as the LLM judge in the 2-pass matching step (`--checker-model`). They may be the same model. Both flags are required.

### Arguments and Options

| Parameter | Flag | Default | Description |
|---|---|---|---|
| **Input File** | `input_file` *(Argument)* | *Required* | Path to the eval JSON: items with a `response` and GT triplets. |
| **Extractor Model** | `--extractor-model`, `-e` | *Required* | Model used to extract triplets live. Also the prefix for the predicted-triplet key (`{model}_response_kg`). |
| **Checker Model** | `--checker-model` | *Required* | LLM judge used for the 2-pass semantic matching (Recall + Precision passes). |
| **Extractor API Base** | `--extractor-base-api` | `None` | Base URL for the extractor LLM (OpenAI-SDK route). If unset, LiteLLM provider routing is used. |
| **Checker API Base** | `--checker-base-api` | `None` | Base URL for the checker/matching LLM. |
| **GT Key** | `--gt-key` | `claude2_response_kg` | Item key holding the ground-truth triplets. |
| **Output File** | `--output`, `-o` | `results/{input_stem}_extractor_eval[_{runs}].json` | Summary metrics JSON, written next to the input. The `_{runs}` suffix appears only for `--runs > 1`. A `*_disagreements.json` is written alongside. |
| **Joint Bundle Size** | `--joint-num` | `10` | Max claims bundled per joint matching call. |
| **Word Budget** | `--max-words` | `None` | Word budget per matching call. |
| **Max Retries** | `--max-retries` | `None` | Retry rounds for API/parse failures. |
| **Debug Mode** | `--debug` | `False` | Timestamps + module names in logs. |

> **Predicted-triplet key is derived, not configurable.** Live extraction always writes its output under `{extractor_model}_response_kg`, so that is the key the evaluator reads predictions from — there is no override flag. The evaluator guards against a collision: if `{extractor_model}_response_kg` equals `--gt-key` (e.g. `--extractor-model claude2` with the default GT key), it raises `InvalidInputError` immediately, because otherwise extraction would target the GT slot and the evaluator would match ground truth against itself and report a misleading perfect score.

### Required environment (`.env`)

| Variable | Used for |
|---|---|
| `EXTRACTOR_API_KEY` | Auth for the extractor LLM endpoint. |
| `CHECKER_API_KEY` | Auth for the checker/matching LLM endpoint. |
| `LLM_TIMEOUT` *(optional)* | Per-call timeout in seconds (default `120.0`). |

Settings reads these eagerly at import; the service validates the relevant key before any work and raises `InvalidInputError` if missing.

### Worked example

Against a local LiteLLM proxy, using one model for both roles:

```bash
contextchecker eval extractor eval_data/msmarco/msmarco_5.json \
  --extractor-model gemini-3.1 \
  --checker-model gemini-3.1 \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
```

This reads GT from `claude2_response_kg`, extracts live into `gemini-3.1_response_kg`, runs the 2-pass match, then writes `eval_data/msmarco/results/msmarco_5_extractor_eval.json` plus its `_disagreements.json` sibling. Output always lands in a `results/` directory beside the input, so an input already inside `results/` nests one level deeper — pass `-o` to control this.

## Output Format

Two files are written. The summary file wraps the `ExtractorEvalResult` dataclass with a `_meta` block:

```json
{
  "_meta": {
    "eval_type": "extractor",
    "extractor_model": "gemini-3.1",
    "gt_key": "claude2_response_kg",
    "pred_key": "gemini-3.1_response_kg",
    "method": "llm",
    "checker_model": "gemini-3.1",
    "timestamp": "..."
  },
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "tp": 0,
  "fp": 0,
  "fn": 0,
  "total_items": 0,
  "to_compare_items": 0,
  "gt_stats": { "total_triplets": 0, "avg_per_item": 0.0 },
  "pred_stats": { "total_triplets": 0, "avg_per_item": 0.0 },
  "abstention_errors": {
    "wrongful_answer": 0,
    "wrongful_abstention": 0,
    "wrongful_abstention_fn_penalty": 0,
    "wrongful_answer_fp_penalty": 0
  },
  "correct_abstention": 0,
  "method": "llm"
}
```

The disagreements file (`{output_stem}_disagreements.json`) contains `_meta`, a `total_disagreements` count, and an `items` list of the individual False Positives / False Negatives for error analysis.
