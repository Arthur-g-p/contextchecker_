# Extraction Pipeline — Validation & Filtering

How `ExtractionService` decides what to extract, what to skip, and what to reject.

## Pipeline overview

```
Input (list[dict])
  │
  ├─ _validate()  →  drops invalid items, raises if none survive
  ├─ _filter()    →  splits into: pending / abstained / already-processed
  ├─ extract      →  only "pending" items hit the LLM
  └─ serialize    →  writes results back into the original dicts
```

---

## Step 1: Validation (`_validate`)

**Goal**: Ensure every item has the data we need before doing any work.

| Scenario | Input | Outcome |
|----------|-------|---------|
| Item has `"response"` key | `{"response": "Paris is in France"}` | ✅ Passes validation |
| Item missing `"response"` | `{"question": "Where is Paris?"}` | ⚠️ Skipped with warning log |
| ALL items missing `"response"` | `[{"q": "..."}, {"a": "..."}]` | ❌ `InvalidInputError` — hard stop |
| Empty list `[]` | `[]` | ❌ `InvalidInputError` — hard stop |

**Behavior**: Partial failures are tolerated. If 8 out of 10 items have a `"response"` key, the 8 valid ones proceed and the 2 invalid ones are silently skipped (with a log warning per item).

Only when **zero** items survive does it raise `InvalidInputError`.

---

## Step 2: Filtering (`_filter`)

**Goal**: Avoid redundant work. Don't re-extract items that already have results, and don't waste LLM calls on obvious abstentions.

The filter splits valid items into three buckets:

### Bucket 1: Already processed → **silently skipped**

| Scenario | Input | Outcome |
|----------|-------|---------|
| Item has `{model}_response_kg` key | `{"response": "...", "gemini-2.0-flash_response_kg": [...]}` | Skipped — left untouched |

The key name is dynamic: `f"{model}_response_kg"`. So `model="gemini-2.0-flash"` checks for `"gemini-2.0-flash_response_kg"`.

This enables **resumable runs** — you can re-run extraction on the same file and it won't redo work.

### Bucket 2: Full abstention → **empty triplets, no LLM call**

| Scenario | Input | Outcome |
|----------|-------|---------|
| Empty response | `{"response": ""}` | Abstention → `[]` |
| Whitespace only | `{"response": "   \n  "}` | Abstention → `[]` |
| `None` response | `{"response": null}` | Abstention → `[]` |
| Pure refusal | `{"response": "I don't know"}` | Abstention → `[]` |
| Refusal with punctuation | `{"response": "I don't know."}` | Abstention → `[]` |
| Refusal, case-insensitive | `{"response": "I DON'T KNOW"}` | Abstention → `[]` |

**How abstention detection works** (`_is_full_abstention`):
1. Strip, lowercase, remove all punctuation, collapse whitespace.
2. Check if any known refusal phrase is present in the cleaned text.
3. If the phrase covers ≥85% of the cleaned text's length → it's a full abstention.

This means a **long** response that *contains* "I don't know" is NOT flagged — the phrase has to dominate the response.

Known refusal phrases:
- `"i dont know"`
- `"i cannot answer"`
- `"not provided in the context"`
- `"i dont have enough information"`
- `"information not provided"`

### Bucket 3: Pending → **sent to LLM**

Everything that isn't already processed and isn't an abstention.

### Edge case: nothing to do

| Scenario | Outcome |
|----------|---------|
| All items already have `{model}_response_kg` and none are abstentions | ❌ `FilterError` — "Nothing to extract" |
| Mix of already-processed + abstentions (but zero pending) | ✅ Succeeds — abstentions get `[]`, no LLM call needed |
| All items are abstentions | ✅ Succeeds — all get `[]`, no LLM call needed |

---

## What gets sent to the LLM

Only **pending** items. Each one becomes an `ExtractionPayload(text=item["response"])` and goes through the `Extractor` worker.

## What gets written back

| Category | What's written to `{model}_response_kg` |
|----------|----------------------------------------|
| Pending (success) | `[{"subject": ..., "predicate": ..., "object": ...}, ...]` |
| Pending (LLM failure) | `[]` (worker returns empty list for failed items) |
| Abstention | `[]` |
| Already processed | Nothing — left untouched |
| Invalid (no `response` key) | Nothing — skipped entirely |
