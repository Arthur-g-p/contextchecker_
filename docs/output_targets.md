# Extraction Output — Target Designs

Three scenarios showing exactly how the CLI output should look.
These are the **Zielvorstellung** — what we implement against.

## Output modes

| Mode | Activated by | What it shows |
|------|-------------|---------------|
| **Normal** | Default | Pretty output as shown below |
| **Debug** | `--debug` or `CONTEXTCHECKER_LOG_LEVEL=DEBUG` | Same pretty output + `HH:MM:SS | module |` prefix on every line |

There is no separate "verbose" mode. Just normal and debug.

---

## Scenario 1: Happy Path (structured output works first try)

Model supports reasoning + schema. All items valid. No errors.

```
╭──────────────────────────────────────────────────╮
│  ContextChecker · extract                        │
╰──────────────────────────────────────────────────╯

 📂 Validation: customer_responses.json
    Total:      10 items
    ├─ abstain:  0
    └─ valid:   10 items

 🔄 Skip: items with existing extraction (≥1 claim)
    Total:      10 valid items
    └─ pending: 10 items

 ⚙️  Config
    Model:       gemini-2.0-flash @ http://localhost:4000/v1
    Prompts:     C:\Users\Arthur\contextchecker\src\contextchecker\prompt_map.json
    LLMClient:   gemini-2.0-flash via OpenAI SDK @ http://localhost:4000/v1

── Extraction ────────────────────────────────────────
    📡 Testing connection to http://localhost:4000/v1/models...
       ✅ Connection confirmed. Server reachable
    🔬 Discovering best strategy for gemini-2.0-flash...
       🔒 Strategy locked: 'Reasoning + Schema'
    Extracting: 100%|██████████████████████████████| 10/10 [00:08<00:00, 1.24it/s]


── EXTRACTOR RESULTS ────────────────────────────────────────

 🌐 API-Request summary:
    10 (pending) input items
     └─ ✅ 10 successful calls

 📝 Extraction Result summary (out of 10 responses):
    10 results
     └─ 34 claims

── Execution Stats ────────────────────────────────

 📊 Tokens
     └─ extract:  10 reqs · 1,280 in · 1,940 out

 ✅ Done: 10 items extracted → 34 claims, 0 abstentions, 0 skipped
 📁 results/customer_responses_extracted.json
```

---

## Scenario 2: Mixed — Abstentions, Retries, Recovery

Model doesn't support structured output. Falls back through strategy matrix.
Some items are abstentions. First pass has parse errors. Retry recovers them.

```
╭──────────────────────────────────────────────────╮
│  ContextChecker · extract                        │
╰──────────────────────────────────────────────────╯

 📂 Validation: example_in_ref.json
    Total:      7 items
    ├─ abstain:  2  (empty response)
    └─ valid:   7 items

 🔄 Skip: items with existing extraction (≥1 claim)
    Total:      7 valid items
    └─ pending: 7 items

 ⚙️  Config
    Model:       llama3 @ http://localhost:4000/v1
    Prompts:     C:\Users\Arthur\contextchecker\src\contextchecker\prompt_map.json
    LLMClient:   llama3 via OpenAI SDK @ http://localhost:4000/v1

── Extraction ────────────────────────────────────────
    📡 Testing connection to http://localhost:4000/v1/models...
       ✅ Connection confirmed. Server reachable
    🔬 Discovering best strategy for llama3...
       ⚠️  UNSUPPORTED PARAMS IN 500 ERROR (llama3).
       ⬇️  Next strategy: 'Schema Only'
       ⚠️  UNSUPPORTED PARAMS IN 500 ERROR (llama3).
       ⬇️  Next strategy: 'Reasoning + JSON'
       ⚠️  UNSUPPORTED PARAMS IN 500 ERROR (llama3).
       ⬇️  Next strategy: 'JSON Only'
       ⚠️  UNSUPPORTED PARAMS IN 500 ERROR (llama3).
       ⬇️  Next strategy: 'Vanilla'
       🔒 Strategy locked: 'Vanilla'
    Extracting: 100%|██████████████████████████████| 5/5 [01:05<00:00, 13.08s/it]

    ♻️  Retry: 5 failed items...
    Retrying extraction: 100%|█████████████████████| 5/5 [00:00<00:00, 9.36it/s]


── EXTRACTOR RESULTS ────────────────────────────────────────

 🌐 API-Request summary:
    7 (pending) input items
     ├─ 🔇 2 prefiltered
     ├─ ✅ 0 successful calls
     └─ ❌ 5 invalid results
        ├─ ♻️ 5 recovered on first retry
        └─ ❌ 0 permanent failures

 📝 Extraction Result summary (out of 5 responses):
    7 results
     ├─ 17 claims
     └─ 2 abstentions
        └─ 2 prefiltered (empty response)

── Execution Stats ────────────────────────────────

 📊 Tokens
     └─ extract:  10 reqs · 1,463 in · 1,501 out

 ✅ Done: 7 items extracted → 17 claims, 2 abstentions, 0 skipped
 📁 results/ext_example_in_ref_ext-llama3.json
```

---

## Scenario 3: Partial Failure — Resumable Run + Permanent Errors

Second run on the same file. Some items already extracted (skipped).
Some items hit content policy. Some permanently fail.

```
╭──────────────────────────────────────────────────╮
│  ContextChecker · extract                        │
╰──────────────────────────────────────────────────╯

 📂 Validation: large_dataset.json
    Total:      50 items
    ├─ abstain:  3  (empty response)
    └─ valid:   48 items

 🔄 Skip: items with existing extraction (≥1 claim)
    Total:      48 valid items
    ├─ skipped: 30 items (already extracted)
    └─ pending: 18 items

 ⚙️  Config
    Model:       gemini-2.0-flash @ https://api.openai.com/v1
    Prompts:     C:\Users\Arthur\contextchecker\src\contextchecker\prompt_map.json
    LLMClient:   gemini-2.0-flash via OpenAI SDK @ https://api.openai.com/v1

── Extraction ────────────────────────────────────────
    📡 Testing connection to https://api.openai.com/v1/models...
       ✅ Connection confirmed. Server reachable
    🔬 Discovering best strategy for gemini-2.0-flash...
       🔒 Strategy locked: 'Reasoning + Schema'
       ⚠️  CONTENT POLICY VIOLATION (gemini-2.0-flash): Safety filter triggered.
       ⚠️  CONTEXT WINDOW EXCEEDED (gemini-2.0-flash): Input too long.
    Extracting: 100%|██████████████████████████████| 18/18 [00:12<00:00, 1.50it/s]

    ♻️  Retry: 3 failed items...
    Retrying extraction: 100%|█████████████████████| 3/3 [00:02<00:00, 1.12it/s]


── EXTRACTOR RESULTS ────────────────────────────────────────

 🌐 API-Request summary:
    18 (pending) input items
     ├─ ✅ 13 successful calls
     ├─ 📏 1 context too long
     ├─ 🛡️  1 content blocked
     └─ ❌ 3 invalid results
        ├─ ♻️ 2 recovered on first retry
        └─ ❌ 1 permanent failures

 📝 Extraction Result summary (out of 15 responses):
    50 results
     ├─ 42 claims
     └─ 5 abstentions
        ├─ 3 prefiltered (empty response)
        └─ 2 model returned empty

── Execution Stats ────────────────────────────────

 📊 Tokens
     └─ extract:  21 reqs · 4,812 in · 3,290 out

 ✅ Done: 50 items extracted → 42 claims, 5 abstentions, 30 skipped
 📁 results/large_dataset_extracted.json
```

---

## Debug Mode

Activated with `--debug` flag or `CONTEXTCHECKER_LOG_LEVEL=DEBUG`.
Same pretty output, every line gets a timestamp + source module prefix.

```
17:42:48 | cli          | ╭──────────────────────────────────────────────────╮
17:42:48 | cli          | │  ContextChecker · extract                        │
17:42:48 | cli          | ╰──────────────────────────────────────────────────╯
17:42:48 | extraction   |
17:42:48 | extraction   |  📂 Validation: large_dataset.json
17:42:48 | extraction   |     Total:      50 items
17:42:48 | extraction   |     ├─ abstain:  3  (empty response)
17:42:48 | extraction   |     └─ valid:   48 items
17:42:48 | extraction   |
17:42:48 | extraction   |  🔄 Skip: items with existing extraction (≥1 claim)
17:42:48 | extraction   |     Total:      48 valid items
17:42:48 | extraction   |     ├─ skipped: 30 items (already extracted)
17:42:48 | extraction   |     └─ pending: 18 items
17:42:48 | extraction   |
17:42:48 | extraction   |  ⚙️  Config
17:42:48 | extraction   |     Model:       gemini-2.0-flash @ https://api.openai.com/v1
17:42:48 | extraction   |     Prompts:     C:\Users\Arthur\contextchecker\src\contextchecker\prompt_map.json
17:42:49 | llmclient    |     LLMClient:   gemini-2.0-flash via OpenAI SDK @ https://api.openai.com/v1
17:42:49 | llmclient    |
17:42:49 | llmclient    |     📡 Testing connection to https://api.openai.com/v1/models...
17:42:49 | llmclient    |        ✅ Connection confirmed. Server reachable
17:42:49 | llmclient    |     🔬 Discovering best strategy for gemini-2.0-flash...
17:42:50 | llmclient    |        🔒 Strategy locked: 'Reasoning + Schema'
17:42:51 | llmclient    |        ⚠️  CONTENT POLICY VIOLATION: Safety filter triggered.
17:42:55 | extractor    |     Extracting: 100%|████████████████| 18/18 [00:12]
17:42:58 | extractor    |     ♻️  Retry: 3 failed items...
17:43:00 | extractor    |     Retrying: 100%|██████████████████| 3/3 [00:02]
17:43:00 | extraction   |
17:43:00 | extraction   |  🌐 API-Request summary:
17:43:00 | extraction   |     18 (pending) input items
17:43:00 | extraction   |      ├─ ✅ 13 successful calls
17:43:00 | extraction   |      └─ ❌ 3 invalid results
...
```

The debug prefix reveals **when** each step happened and **which module** produced it.
This makes it possible to trace timing issues, identify which component is slow,
and see the handoff between service → worker → llmclient.
