# Request Strategies

How a request reaches the model, what happens when it fails, and which knobs
you can turn. Implemented in `llmclient.py`.

Three things stack here, and they are easy to confuse because all three are
called "retry" in casual conversation:

| Layer | Question it answers | Scope | Lifetime |
| --- | --- | --- | --- |
| **1. Strategy discovery** | What will this endpoint accept? | one `(base_url, model)` | discovered once, cached for the process |
| **2. Request handling** | Did *this* request succeed? | one request | one `generate()` call |
| **3. Retry rounds** | Did the model return something parseable for *this item*? | one item | one batch, then discarded |

Layer 3 lives in the workers, not here — see `architecture.md`. Layers 1 and 2
are this document.

---

## Two paths to the model

**Set a base URL** — `--extractor-base-api` or `--checker-base-api` — and the
request goes out over the [OpenAI SDK][openai-sdk], directly to that endpoint.
**Leave it unset** and the provider is read from a prefix on the model name, and
LiteLLM resolves the endpoint, credentials and request transformation.

```bash
# base URL set → OpenAI SDK, straight to the endpoint
claimlens extract data.json \
  --extractor-base-api https://openrouter.ai/api/v1 \
  --extractor-model openai/gpt-5.6-luna

# no base URL → LiteLLM routes it
claimlens extract data.json \
  --extractor-model openrouter/openai/gpt-5.6-luna
```

The model names differ on purpose. `openai/gpt-5.6-luna` is OpenRouter's own
identifier and is sent as-is; the extra `openrouter/` in the second is
[LiteLLM's provider prefix][prefix], not part of any model ID.

No capability / Strategy discovery (Layer 1) probing happens on the second path. It is a deliberate
passthrough: the request is handed to LiteLLM as written and LiteLLM's judgement
about the model is trusted. Probing there would not mean much anyway — a
rejection from LiteLLM is LiteLLM's opinion about the backend, not the backend's
answer.

That opinion is formed before the network. LiteLLM resolves the provider, builds
a [supported-parameter list][supported-params], compares the request against it,
and maps what survives into a provider-specific shape. What does not survive is
[stripped rather than reported][filtering] — this path sends `drop_params=True`,
so an unrecognised field is removed and the call goes out without it.

Anything LiteLLM believes the model does not support is removed — including
`response_format`, whenever a schema is asked for. The verdict comes from
provider allowlists, model-name rules, transformation code — the OpenRouter
adapter, for instance, [gates reasoning parameters on its own capability
check][reasoning-gate] — and a [per-model capability record][capability-record].
All of it is only as current as the installed version.

When that metadata is stale, `response_format` is removed from a model that
would have honoured it: the request succeeds, the output comes back
unconstrained, and nothing in the response says an intermediary changed the ask.
You see it later, as parse failures, attributed to the model.

Updating LiteLLM narrows the window; it cannot close the race between a
provider's new capability and client-side metadata. Setting a base URL is not the
escape either — LiteLLM accepts a base URL and
[still validates][still-validates]. Leaving LiteLLM is, and that is what the
first path does.

**LiteLLM Proxy can exhibit the same problem**, because it runs the LiteLLM SDK
under the hood. A base URL pointing at a Proxy escapes none of the above; it only
moves the checks from your machine to the server.

| Scenario | Path | Why |
| --- | --- | --- |
| Self-hosted vLLM / Ollama / SGlang | OpenAI SDK | discovery needs the real backend |
| Provider endpoint directly | OpenAI SDK | rejections come from the service itself |
| A model newer than your LiteLLM release | OpenAI SDK | capability metadata lags the provider |
| Behind LiteLLM Proxy | either | the same checks run server-side regardless |
| Portability across many providers | LiteLLM | switch models by name alone |

> **About the source links:** pinned to LiteLLM `v1.98.0`, the most recent release
> at the time of writing, so the line numbers stay valid.

[openai-sdk]: https://github.com/openai/openai-python
[prefix]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/litellm_core_utils/get_llm_provider_logic.py#L184-L220
[supported-params]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/litellm_core_utils/get_supported_openai_params.py#L8-L59
[filtering]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/utils.py#L4005-L4053
[capability-record]: https://github.com/BerriAI/litellm/blob/v1.98.0/model_prices_and_context_window.json#L25089-L25149
[reasoning-gate]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/llms/openrouter/chat/transformation.py#L36-L50
[still-validates]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/main.py#L5327-L5373

---

## Layer 1 — Strategy discovery

**This is the OpenAI SDK path only.** With no base URL there are no strategies
and nothing is probed — the first request is still serialized, but only to
validate the connection, and LiteLLM's judgement about the model stands as
described above.

Not every endpoint supports every feature. A self-hosted vLLM may reject
`reasoning_effort`. An older model may not support a JSON schema. A provider
may accept `json_object` but not strict schema enforcement.

Sending the best request and reading the `BadRequestError` tells you nothing:
you cannot see whether the *model* is broken or the *feature* is unsupported.
So the client tries strategies from best to worst and keeps the first that
works.

| # | Strategy | `reasoning_effort` | `response_format` | Shape enforced? |
| --- | --- | --- | --- | --- |
| 1 | Reasoning + Schema | `low` | strict JSON schema | yes |
| 2 | Schema Only | — | strict JSON schema | yes |
| 3 | Reasoning + JSON | `low` | `{"type": "json_object"}` | no — valid JSON, any keys |
| 4 | JSON Only | — | `{"type": "json_object"}` | no — valid JSON, any keys |
| 5 | Unguided Decoding | — | *none sent* | no — the prompt carries the format |

**The first request walks the matrix alone**, serialized behind a lock while
every other request waits. On the first success the level is locked and stored
in a process-wide cache keyed by `(base_url, model)`, and recorded per model as
`request_strategies` in the `_meta` of every envelope report (refcheck, ragcheck,
faithcheck, both evals — not extract/check/atomize), next to `usage`, the run's
request and token counts. `requests` there counts every HTTP call, tokens only
exist for calls that answered — a timeout is a request with no tokens. Every later request —
and every other worker pointed at the same endpoint — starts there with zero
overhead. Typical cost: two or three extra round trips, once.

Walking down does **not** consume a retry attempt. A capability error is not a
failure of the item.

Two behaviours worth knowing:

- **`drop_params` probe.** `drop_params` is a LiteLLM-proxy field that a direct
  endpoint rejects. On the first rejection the client retries the *same* level
  without it — keeping reasoning — and only moves down if that also fails. The
  answer is cached per endpoint.
- **Unguided Decoding needs a plain prompt.** Nothing constrains the output at
  level 5, so the caller must supply a prompt that states the format in prose
  (`plain_messages`). Reaching level 5 without one is fatal: sending a
  schema-shaped prompt into an unconstrained decode produces unparseable text
  for every item. The workers warn at startup if their plain prompt is missing.
---

## Layer 2 — Request handling

### The seven actions

Every API error is classified into exactly one. **Blast radius** is the column
that matters at 3am: only two of the seven can end a run.

| Action | Retries | Blast radius | Behaviour |
| --- | --- | --- | --- |
| `FATAL` | none | **kills the batch** | auth, permissions, unknown model, budget, 402, 405 — nothing downstream can succeed |
| `SKIP` | none | one item | the input was the problem, not the system |
| `RETRY` | 3 tries | one item | back off `0.5s × attempt` |
| `RESAMPLE` | 3 tries, plus 3 per level during discovery | one item | retried immediately — the response arrived and was billed, it just wasn't JSON |
| `RATE_LIMIT` | **unbounded** | none — never drops an item | waits for `Retry-After`, else ~60s jittered; **fatal** only if the server asks for longer than `RATE_LIMIT_MAX_WAIT` |
| `SERVER_ERROR` | 3 consecutive | **kills the batch** on the 4th | back off `5s × count`; a transient 500 is worth retrying, a dead server is not |
| `TIMEOUT` | 1 | one item | the payload is identical, so more attempts only re-buy the same timeout |

An item that exhausts its retries is skipped and its cause recorded — never
silently dropped. A capability error is neither: walking the strategy matrix
costs no attempts at all.

### Worked example

A 500-item batch against a busy provider:

```
item  12   429, no Retry-After         → waits ~60s, succeeds       nothing lost
item  47   response is not valid JSON  → resampled 3x, still bad    recorded as parse_failure
item  88   reference exceeds context   → skipped immediately        recorded as context_too_long
item 133   request exceeds LLM_TIMEOUT → retried once, times out    recorded as timeout
item 201   three consecutive 500s      → batch aborts               items 202-500 never sent
```

Four of those cost one claim each and the run still produces a report you can
read. Only the fifth ends it.

### Why the granularity matters

In batch processing you need more than "retry or crash":

- **Auth error on item 47** — don't process 48–500, the key is dead → `FATAL`
- **Context too long on item 47** — the input was too big, not the system → `SKIP`
- **Rate limit on item 47** — the system is healthy, just busy → `RATE_LIMIT`
- **Server 500 on item 47** — maybe transient, maybe not → `SERVER_ERROR`

### Classification order

Subclasses **must** be checked before parents:

```
AuthenticationError            → FATAL
PermissionDeniedError          → FATAL
NotFoundError                  → FATAL
BudgetExceededError            → FATAL
ContextWindowExceededError     → SKIP
ContentPolicyViolationError    → SKIP
ContentFilterFinishReasonError → SKIP
UnsupportedParamsError         → SKIP (or strategy advance during discovery)
JSONSchemaValidationError      → RESAMPLE
UnprocessableEntityError       → SKIP
BadRequestError (base)         → SKIP, or FATAL when the model name is rejected
RateLimitError                 → RATE_LIMIT
APITimeoutError                → TIMEOUT
asyncio.TimeoutError           → TIMEOUT
APIConnectionError             → SERVER_ERROR
InternalServerError            → SERVER_ERROR
LengthFinishReasonError        → SKIP
APIError (base)                → SERVER_ERROR; FATAL for 402 and 405
ConflictError                  → RETRY
ValidationError                → RESAMPLE
Unknown                        → RETRY
```

---

## What you can set

Environment variables, read once at import (`.env` is loaded automatically):

| Variable | Default | What it does |
| --- | --- | --- |
| `LLM_TIMEOUT` | `120.0` | Seconds a single request may take. See below. |
| `LLM_MAX_TOKENS` | unset | Output-token cap per request. Unset = the model's own limit. |
| `RATE_LIMIT_WAIT` | `60` | Back-off when a 429 carries no `Retry-After`. |
| `RATE_LIMIT_MAX_WAIT` | `300` | Abort if the server asks to wait longer than this. |
| `RATE_LIMIT_HEARTBEAT` | `30` | How often to log "still rate-limited" so a long wait never looks hung. |

CLI flags, per command:

| Flag | Default | What it does |
| --- | --- | --- |
| `--concurrency` | `10` | Simultaneous requests **per LLM client**. An extractor and a checker each get their own. |
| `--joint-num` | `10` | Max claims per joint checker call. |
| `--max-words` | `6000` joint, unset single | Word budget per call. If one reference fills the budget alone, the service falls back to one claim per call. |

### Set the timeout high

`LLM_TIMEOUT` is a wall-clock deadline per request, and a reasoning model on a
long joint prompt can legitimately take minutes. Set it generously — a low
timeout does not make anything faster, it just converts slow successes into
lost items.

A timed-out request gets exactly one retry (the payload is identical, so more
attempts only re-buy the same timeout), then the item is skipped.

The transport gets `LLM_TIMEOUT × 1.5` so one deadline wins deterministically
instead of two racing. The progress bar shows the age of the longest
outstanding request once it passes three seconds.

### Nothing is lost silently

A skipped item is **written into the output**, next to the claim it belongs to,
under `{model}_extraction_error` or `{namespace}_error`:

| Cause | Meaning |
| --- | --- |
| `parse_failure` | no valid response after every retry round — the most common |
| `timeout` | exceeded `LLM_TIMEOUT`, twice |
| `context_too_long` | input exceeded the context window |
| `finish_reason_length` | the answer was cut off by an output-token limit — raise `LLM_MAX_TOKENS` |
| `content_policy` | a safety filter rejected it |

So a run with failures does not quietly score lower. The claim keeps both sides
of the ratio and the cause is on disk. Check these before comparing runs — see
`docs/outcome_markers.md`.

---
