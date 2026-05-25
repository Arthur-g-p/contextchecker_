# LLMClient — Architecture & Design Rationale

## What it does

`LLMClient` is a unified async wrapper around LLM API calls. It handles three things that no individual API SDK handles well on its own:

1. **Automatic capability discovery** — detects what a model/endpoint actually supports
2. **Structured error taxonomy** — classifies every possible API failure into an actionable category
3. **Crash recovery** — persists partial results so users don't lose work on failures

## Why two routes?

### OpenAI SDK Route (direct endpoint)

Used when `base_url` is set. The client talks directly to an API endpoint using the OpenAI SDK protocol (which is the de facto standard — vLLM, Ollama, LiteLLM proxy, Azure, and dozens of providers all speak it).

**The problem this solves**: Not every endpoint supports every feature. A self-hosted vLLM instance might not support `reasoning_effort`. An older model might not support structured output (`response_format` with a JSON schema). A provider might support `json_object` mode but not strict schema enforcement.

If you just send the "best" request and it fails, you get a `BadRequestError` — and you don't know if the _model_ is broken or if you just asked for a feature it doesn't support.

**The retry matrix** solves this by trying strategies from best to worst:

```
1. Reasoning + Schema     ← ideal: thinking + constrained decoding
2. Schema Only            ← no reasoning, but still structured output
3. Reasoning + JSON       ← thinking + loose JSON mode
4. JSON Only              ← loose JSON, no reasoning
5. Vanilla                ← raw text, prompt-based format instruction
```

The **first request** walks the matrix alone (serialized via `asyncio.Lock`). Once a strategy succeeds, it's locked and all subsequent requests run concurrently with that strategy. This means:

- **One-time cost**: The first request might take 2-3 extra round trips (usually <5 seconds total)
- **All subsequent requests**: Zero overhead, locked at the best working level
- **No static compatibility tables**: Works with any endpoint, any model, any provider — discovered at runtime

> **"Why not just check the model name and pick the right strategy?"**
>
> Because model names lie. A `gpt-4o` behind a LiteLLM proxy might have different capabilities than `gpt-4o` direct from OpenAI. A `gemini-2.0-flash` on Vertex AI behaves differently than the same model on AI Studio. Self-hosted models via vLLM or Ollama have their own quirks. The only reliable way to know what works is to try it.

### LiteLLM Route (provider routing)

Used when `base_url` is **not** set. LiteLLM handles provider detection, API key routing, and capability mapping internally via its own model registry.

- No capability probing needed — LiteLLM's `drop_params=True` handles unsupported features
- No strategy matrix — LiteLLM does its own fallback logic
- Trade-off: Less control, but works with 100+ providers out of the box

### When to use which

| Scenario                       | Route      | Why                                    |
| ------------------------------ | ---------- | -------------------------------------- |
| Self-hosted vLLM/Ollama        | OpenAI SDK | Need capability discovery              |
| LiteLLM proxy (corporate)      | OpenAI SDK | Direct endpoint, proxy handles routing |
| OpenAI/Anthropic/Google direct | Either     | Both work, LiteLLM is simpler          |
| Multi-provider evaluation      | LiteLLM    | Switch models with just the model name |

## Error Taxonomy

Every API error is classified into exactly one of five actions:

```
FATAL         → raise immediately, kill the batch, save cache
SKIP          → per-item failure, return error, batch continues
RETRY         → transient error, backoff and retry (max 3 attempts)
RATE_LIMIT    → long backoff (60s), retry, no attempt increment
SERVER_ERROR  → progressive backoff (5s × count), max 3 consecutive
```

### Why this matters

Most LLM wrappers treat errors as binary: retry or crash. In production batch processing (hundreds of items), you need granularity:

- **Auth error on item 47**: Don't process items 48-500. The key is dead. Save what you have. → `FATAL`
- **Context too long on item 47**: Skip it, extract from item 48. The input was too big, not the system. → `SKIP`
- **Rate limit on item 47**: Wait 60 seconds, retry the same item. The system is healthy, just busy. → `RATE_LIMIT`
- **Server 500 on item 47**: Maybe transient. Retry 3 times with backoff. If it persists, the server is down. → `SERVER_ERROR`

### Error classification order

Subclasses **must** be checked before parents. The handler checks in this order:

```
AuthenticationError          → FATAL
PermissionDeniedError        → FATAL
NotFoundError                → FATAL
BudgetExceededError          → FATAL
ContextWindowExceededError   → SKIP
ContentPolicyViolationError  → SKIP
UnsupportedParamsError       → SKIP (or strategy advance during discovery)
BadRequestError (base)       → SKIP or FATAL (model name check)
RateLimitError               → RATE_LIMIT
APITimeoutError              → SERVER_ERROR
APIConnectionError           → SERVER_ERROR
InternalServerError          → SERVER_ERROR
APIError (base)              → SERVER_ERROR (or FATAL for 402)
ValidationError              → RETRY
Unknown                      → RETRY
```

## Crash Cache

On fatal errors or rate limit exhaustion, the client saves all successful responses to a local SQLite file (`.rag_crash_cache.db`). On the next run, cached responses are loaded and skipped — the user picks up where they left off without re-processing (and re-paying for) completed items.

**Lifecycle:**

1. Run starts → check for existing cache file → load if found
2. Each successful API response → stored in memory dict
3. Fatal error → dump memory dict to SQLite → raise exception
4. Next run → cache loaded → cached items return instantly → new items processed normally
5. Successful completion → caller deletes cache file (nothing to recover)

## Error Propagation

```
LLMClient.generate()
  │
  ├─ per-item error (ContextTooLong, ContentPolicy, ParseError)
  │   └─ _generate_safe() catches → returned as value in results list
  │       └─ Worker classifies: permanent failure vs retryable
  │
  └─ fatal error (Auth, Connection, Budget)
      └─ _generate_safe() does NOT catch → propagates
          └─ generate_batch() saves cache → re-raises
              └─ Worker lets it propagate
                  └─ Service lets it propagate
                      └─ CLI catches ContextCheckerError → logs → exit(1)
```

## Concurrency Model

- `asyncio.Semaphore(concurrency)` limits parallel API calls (default: 10)
- `asyncio.Lock` serializes strategy discovery (first request only)
- `tqdm_asyncio.gather` runs all batch items concurrently within the semaphore limit
- `GLOBAL_STATS` uses `threading.Lock` for token counting (safe across async tasks)
