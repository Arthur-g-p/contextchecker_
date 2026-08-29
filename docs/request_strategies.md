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

The path a request takes determines whether Layer 1 can exist at all.

### OpenAI-compatible path — recommended

Selected when `--extractor-base-api` or `--checker-base-api` is set. ContextChecker
uses the [OpenAI Python SDK][openai-sdk] and sends the request directly to the
configured endpoint over the OpenAI-compatible protocol. vLLM, Ollama, OpenRouter,
Azure OpenAI, LiteLLM Proxy and many other services expose some form of it.

```bash
contextchecker extract data.json \
  --extractor-base-api https://openrouter.ai/api/v1 \
  --extractor-model openai/gpt-5.6-luna
```

`openai/gpt-5.6-luna` is OpenRouter's own model identifier, sent as-is. It must
**not** carry LiteLLM's outer provider prefix: `openrouter/` belongs to LiteLLM's
provider-selection convention, not to OpenRouter's model ID ([prefix
handling][prefix]).

When the base URL points at the provider or model server being measured, that
endpoint decides which parameters and values it accepts. A supported parameter
reaches the service; an unsupported one produces the service's real rejection.
Those responses are the evidence Layer 1 needs.

Passing `api_base` to LiteLLM does **not** bypass its validation: LiteLLM resolves
the provider and base URL first, then still calls `get_optional_params()` before
dispatch ([resolution][resolve] · [dispatch][optional-params]). The bypass comes
from ContextChecker switching away from LiteLLM to the OpenAI SDK.

### LiteLLM path — convenient and compatible

Selected when no base URL is set. The provider is encoded in the model name, and
LiteLLM resolves the endpoint, credentials and request transformation:

```bash
contextchecker extract data.json \
  --extractor-model openrouter/openai/gpt-5.6-luna
```

The first segment selects LiteLLM's OpenRouter adapter; the remainder is the
model identifier sent upstream ([prefix handling][prefix] ·
[adapter][openrouter-adapter]).

**This applies to LiteLLM Proxy too, and the Proxy cannot opt out.** Its
OpenAI-compatible chat endpoint routes ordinary requests into LiteLLM's router
and completion machinery, through the same adapters, capability checks and
parameter mappings ([endpoint][proxy-endpoint] · [dispatch][proxy-dispatch] ·
[router][router]). The Proxy *is* LiteLLM, so it cannot take the direct SDK route
the way ContextChecker can. Pointing `--extractor-base-api` at a Proxy therefore
still lands you in the capability layer below — it moves it from your machine to
the server, nothing more.

### Why the OpenAI-compatible path is recommended

LiteLLM does more than route. Before a completion reaches the network it resolves
the provider, obtains a supported-parameter list, compares it against the request,
and maps accepted fields into a provider-specific format.

Those checks are spread across provider allowlists, model-name rules,
transformation code and a per-model capability record — a **provider- and
model-specific capability layer**, not a single registry — and every one of them
is only as current as your installed version:

| What decides | Where |
| --- | --- |
| supported-parameter list | [`get_supported_openai_params()`][supported-params] |
| allowlist comparison, drop-or-raise | [`utils.py` L4005–4053][filtering] |
| per-model capability record | [`model_prices_and_context_window.json` L25089–25149][capability-record] |
| reasoning gate for OpenRouter | [transformation L36–50][reasoning-gate] · [`supports_reasoning()`][supports-reasoning] |

So the layer helps when its knowledge matches the provider, and lags when it does
not. LiteLLM's OpenRouter adapter adds reasoning parameters only when its checks
call the model reasoning-capable — a newly released model can already accept
`reasoning_effort` upstream while the installed version still considers it
unsupported.

What happens then depends on configuration ([source][filtering]):

| Setting | Behaviour |
| --- | --- |
| `drop_params=False` | raises `UnsupportedParamsError` before sending |
| `drop_params=True` | removes the parameter and continues |
| `allowed_openai_params` | re-adds fields the operator knows the backend accepts |

The second is the hard one to diagnose: the request succeeds, the feature is gone,
the provider never received the field, and nothing in the response explains that
an intermediary altered it. Keep LiteLLM updated — old adapters and old metadata
make this more likely — but updating cannot close the race between new provider
capabilities and client-side metadata.

**For Layer 1 this is decisive.** Probing through LiteLLM measures the installed
version's understanding of the backend rather than the backend itself, and "the
backend does not support this" becomes indistinguishable from "the intermediary
does not know that it does."

### When to keep the LiteLLM path

It reaches many providers by name, resolves credentials and endpoints, and
translates one request shape into provider-specific formats. For established
models covered by your installed version, that convenience is often the right
trade.

| Scenario | Path | Why |
| --- | --- | --- |
| Self-hosted vLLM / Ollama | OpenAI-compatible | discovery needs the real backend |
| Provider endpoint directly | OpenAI-compatible | rejections come from the service itself |
| A model newer than your LiteLLM release | OpenAI-compatible | capability metadata lags the provider |
| Behind LiteLLM Proxy | either | the capability layer applies server-side regardless |
| Portability across many providers | LiteLLM | switch models by name alone |

> **About the source links:** pinned to LiteLLM `v1.98.0`, the most recent release
> at the time of writing, so the line numbers stay valid. Later versions move
> them; the mechanisms are what to look for.

[openai-sdk]: https://github.com/openai/openai-python
[prefix]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/litellm_core_utils/get_llm_provider_logic.py#L184-L220
[resolve]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/main.py#L5228-L5237
[optional-params]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/main.py#L5327-L5373
[openrouter-adapter]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/llms/openrouter/chat/transformation.py#L36-L77
[proxy-endpoint]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/proxy/proxy_server.py#L9803-L9892
[proxy-dispatch]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/proxy/route_llm_request.py#L419-L490
[router]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/router.py#L2130-L2157
[supported-params]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/litellm_core_utils/get_supported_openai_params.py#L8-L59
[filtering]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/utils.py#L4005-L4053
[capability-record]: https://github.com/BerriAI/litellm/blob/v1.98.0/model_prices_and_context_window.json#L25089-L25149
[reasoning-gate]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/llms/openrouter/chat/transformation.py#L36-L50
[supports-reasoning]: https://github.com/BerriAI/litellm/blob/v1.98.0/litellm/utils.py#L2601-L2605

---

## Layer 1 — Strategy discovery

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
in a process-wide cache keyed by `(base_url, model)`, so every later request —
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

> **Why not just check the model name and pick the right level?**
>
> Because a capable model is not the same as a capable backend. The production
> case is a model behind a LiteLLM proxy, and a model released last week is not
> in the installed LiteLLM's registry yet — so the proxy rejects `reasoning_effort`
> for a model that supports it perfectly well, purely because it has never heard
> of it. Model names tell you nothing about what the path in front of them will
> accept. Probing does.

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
