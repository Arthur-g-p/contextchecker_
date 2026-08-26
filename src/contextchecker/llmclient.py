import asyncio
import contextlib
import random
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import (
    AsyncOpenAI,
    APIError, APIStatusError, APIConnectionError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, BadRequestError,
    NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError,
    LengthFinishReasonError, ContentFilterFinishReasonError,
)
from openai.lib._parsing import type_to_response_format_param
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pydantic import ValidationError

from contextchecker.stats import GLOBAL_STATS
from contextchecker import settings as default_config
from contextchecker.exceptions import (
    LLMClientError, WorkerError, ContextTooLongError, ContentPolicyError,
    LLMParseError, LLMTimeoutError,
)
from contextchecker.utils import build_compact_schema_example

logger = default_config.get_logger(__name__)


# ───────────────────────────────────────────────────────────────
#  RETRY MATRIX
#  Each strategy is a complete request configuration.
#  On capability errors, we advance to the next strategy.
#  First real request discovers the working strategy (serialized).
#  All subsequent requests use the locked strategy concurrently.
# ───────────────────────────────────────────────────────────────

@dataclass
class RetryStrategy:
    """One level in the retry matrix. Readable and explicit."""
    name: str
    reasoning_effort: str | None = None   # "low", "medium", "high" — OpenAI standard. None = don't send.
    use_schema: bool = True                  # Strict JSON Schema (structured output, constrained decoding)
    use_json_object: bool = False            # Loose JSON (valid JSON, shape not enforced)
    temperature: float = 0.0


# Best case at top, vanilla at bottom.
# On capability errors (BadRequest, UnsupportedParams), we walk down.
RETRY_MATRIX = [
    RetryStrategy("Reasoning + Schema",  reasoning_effort="low",  use_schema=True, use_json_object=False),
    RetryStrategy("Schema Only",                                  use_schema=True, use_json_object=False),
    RetryStrategy("Reasoning + JSON",    reasoning_effort="low",  use_json_object=True, use_schema=False),
    RetryStrategy("JSON Only",                                    use_json_object=True, use_schema=False),
    RetryStrategy("Vanilla", use_schema=False, reasoning_effort=None, use_json_object=False)
]


# Seconds a request must be outstanding before the bar shows its age.
HEARTBEAT_MIN_WAIT = 3.0

# Retries a timed-out request gets before the item is skipped. Retry rounds
# resend the same payload, so more attempts only re-buy the same timeout.
TIMEOUT_RETRIES = 1

# How much longer the transport may wait than the wall-clock deadline. Keeping
# them apart means one deadline wins deterministically instead of racing.
TIMEOUT_BACKSTOP_FACTOR = 1.5

# Wall-clock timer for the progress bar, bound at import. Tests patch
# asyncio.sleep to skip back-offs, which would turn the heartbeat into a
# spin loop; the bar's repaint interval is not a back-off.
_ui_sleep = asyncio.sleep


def _describe_error(e: Exception | None) -> str:
    """One-line, human-readable rendering of an error for the console.

    Pydantic's own repr is multi-line and blows past any truncation mid-word
    ('... [type=json'), so validation errors get flattened to their locations
    and messages instead.
    """
    if isinstance(e, ValidationError):
        parts = []
        for err in e.errors():
            # A whole-body parse failure has no loc — don't prefix it with ': '.
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}" if loc else msg)
        return " | ".join(parts)
    return str(e)[:200]


class ErrorAction(Enum):
    """What to do when an API error occurs."""
    FATAL = "fatal"   # Exit program — unrecoverable
    SKIP  = "skip"    # Return "", continue batch — per-item failure
    RETRY = "retry"   # Backoff and retry — transient
    RESAMPLE = "resample" # Immediate retry, no backoff — response arrived but didn't parse
    RATE_LIMIT = "rate_limit" # Long backoff, no attempt increment
    SERVER_ERROR = "server_error" # Independent attempts backoff for infrastructure
    TIMEOUT = "timeout" # TIMEOUT_RETRIES, then the item is skipped — never aborts the batch


class LLMClient:
    # ── Process-level capability memo (shared across all instances) ──────────
    # Endpoint reachability and the working request strategy are idempotent and
    # deterministic per (base_url, model). We discover them once and reuse, so
    # the 2nd/3rd worker hitting the same endpoint skips the /models probe and
    # the strategy-discovery round-trips. Workers still own their own client.
    _VERIFIED_ENDPOINTS: set[str] = set()
    _STRATEGY_CACHE: dict[tuple[str | None, str], int] = {}
    # Whether the endpoint accepts the LiteLLM-only `drop_params` field.
    _DROP_PARAMS_CACHE: dict[tuple[str | None, str], bool] = {}
    # Init lines already emitted at info level, so duplicate (model, mode, url)
    # constructions don't spam startup. Repeats still log at debug.
    _INIT_LOGGED: set[tuple[str, str, str | None]] = set()

    def __init__(self, api_key: str, model: str, base_url: str | None = None, concurrency: int = 10):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        
        self.timeout = getattr(default_config, 'LLM_TIMEOUT', 120.0)

        # OpenAI SDK client — only created if base_url is set (direct endpoint mode)
        if self.base_url:
            self.client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key,
                # Backstop only. httpx measures the gap between two chunks, never
                # total duration, so a trickling server never trips it — the
                # asyncio deadline around each call is the real limit.
                timeout=self.timeout * TIMEOUT_BACKSTOP_FACTOR,
                # Retrying is this class's job. The SDK's own retries are invisible
                # here: they multiply the request count and hide timeouts.
                max_retries=0,
            )
        else:
            self.client = None  # LiteLLM mode — no direct client needed
        
        self._connection_verified = False
        self.sem = asyncio.Semaphore(self.concurrency)

        # Retry matrix state (OpenAI SDK path only)
        self._strategy_index = 0
        self._strategy_discovered = False   # True after first successful request
        self._discovery_succeeded = False   # True only if discovery pioneer actually got a 200
        self._discovery_lock = asyncio.Lock()  # Serializes strategy discovery
        self._cache_hit_logged = False  # Only log cache hint once
        self._fatal_error_occurred = False
        self._discovering = False  # True while this client is walking the retry matrix

        # drop_params is a LiteLLM-proxy-only field. A direct endpoint rejects it.
        # We learn which we're talking to during discovery (A/B): None = unknown,
        # True = proxy (keep sending it), False = direct endpoint (never send it).
        # Seeded from the process-level cache if a sibling client already learned.
        self._drop_params_supported: bool | None = LLMClient._DROP_PARAMS_CACHE.get(
            (self.base_url, self.model)
        )

        # Rate-limit backoff state (429 handling). Never drops an item.
        self._rate_limit_wait = 0.0          # seconds to sleep on the pending 429
        self._rate_limited_since: float | None = None  # monotonic start of current episode
        self._rate_limit_last_log = 0.0      # monotonic time of last 429 console message

        # Start times of requests currently awaiting a response, keyed by call id.
        self._inflight: dict[int, float] = {}
        self._call_n = 0
        self._last_batch_requests = 0

        # Reuse a strategy already discovered this process for the same endpoint+model.
        self._try_adopt_cached_strategy()

        sdk_mode = "OpenAI SDK" if self.base_url else "LiteLLM"
        # Collapse duplicate init lines: the first unique (model, mode, url) logs
        # at info; repeats drop to debug so --debug still shows every construction.
        init_combo = (self.model, sdk_mode, self.base_url)
        first_time = init_combo not in LLMClient._INIT_LOGGED
        LLMClient._INIT_LOGGED.add(init_combo)
        log = logger.info if first_time else logger.debug
        log("   LLMClient initialized: %s via %s%s", self.model, sdk_mode, f" @ {base_url}" if base_url else "")


    @property
    def strategy(self) -> RetryStrategy:
        """Current retry strategy."""
        return RETRY_MATRIX[self._strategy_index]


    @contextlib.contextmanager
    def _inflight_request(self):
        """Mark a request as awaiting a response for the duration of the block."""
        self._call_n += 1
        key = self._call_n
        GLOBAL_STATS.log_request()
        self._inflight[key] = time.monotonic()
        try:
            yield
        finally:
            self._inflight.pop(key, None)


    @property
    def last_batch_requests(self) -> int:
        """Requests the most recent generate_batch() actually put on the wire.

        Exact only because the SDK's own retry layer is off — one attempt here
        is one HTTP request.
        """
        return self._last_batch_requests


    def oldest_inflight_age(self) -> float | None:
        """Seconds the longest-waiting request has been outstanding, if any."""
        if not self._inflight:
            return None
        return time.monotonic() - min(self._inflight.values())


    def _try_adopt_cached_strategy(self) -> bool:
        """Adopt a strategy already discovered this process for our (base_url, model).

        Returns True if a cached strategy was adopted, else False. Safe to call
        repeatedly: a sibling client may populate the cache *after* we were
        constructed (e.g. an atomizer built at boot whose sibling extractor
        discovers first), so we re-check at request time, not only in __init__.
        """
        if self._strategy_discovered:
            return False
        cached_idx = LLMClient._STRATEGY_CACHE.get((self.base_url, self.model))
        if cached_idx is None:
            return False
        self._strategy_index = cached_idx
        self._strategy_discovered = True
        self._discovery_succeeded = True
        logger.debug(
            "   🔒 Strategy cache hit — reusing '%s' for %s (skipping discovery)",
            self.strategy.name, self.model,
        )
        return True


    def _adopt_cached_drop_params(self) -> None:
        """Seed drop_params support from the process-level cache if still unknown.

        Mirrors _try_adopt_cached_strategy for the eager-construction case: a
        client built before a sibling learned the endpoint type must pick it up
        before firing its first (possibly concurrent) wave, or it re-learns it
        under a stampede.
        """
        if self._drop_params_supported is None:
            cached = LLMClient._DROP_PARAMS_CACHE.get((self.base_url, self.model))
            if cached is not None:
                self._drop_params_supported = cached


    def _next_strategy(self) -> bool:
        """Advance to next strategy. Returns True if advanced, False if at bottom."""
        if self._strategy_index < len(RETRY_MATRIX) - 1:
            self._strategy_index += 1
            logger.info("   ⬇️  Next strategy: '%s'", self.strategy.name)
            return True
        return False


    def _parse_retry_after(self, e: Exception) -> float | None:
        """Extract a Retry-After backoff (seconds) from a rate-limit response.

        Honors the standard `Retry-After` header (integer seconds) and the
        `retry-after-ms` variant some providers send. Returns None when neither
        is present or parseable (an HTTP-date form is treated as absent).
        """
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if not headers:
            return None
        seconds = headers.get("retry-after")
        if seconds is not None:
            try:
                return float(seconds)
            except (TypeError, ValueError):
                pass  # HTTP-date form — fall back to the configured wait
        ms = headers.get("retry-after-ms")
        if ms is not None:
            try:
                return float(ms) / 1000.0
            except (TypeError, ValueError):
                pass
        return None


    def _log_rate_limit(self, wait: float, from_header: bool) -> None:
        """Throttled 429 console output.

        One clear message when an episode begins, then a heartbeat every
        RATE_LIMIT_HEARTBEAT seconds so a long back-off never looks hung — even
        under concurrency, where many tasks hit the limit at once. The episode
        is reset on the next success (see the success path in generate()).
        """
        now = time.monotonic()
        if self._rate_limited_since is None:
            self._rate_limited_since = now
            self._rate_limit_last_log = now
            source = " (server Retry-After)" if from_header else ""
            logger.warning(
                "⏳ Rate limited by %s. Waiting %.0fs%s, then continuing — nothing will be dropped.",
                self.model, wait, source,
            )
            return
        heartbeat = getattr(default_config, 'RATE_LIMIT_HEARTBEAT', 300.0)
        if now - self._rate_limit_last_log >= heartbeat:
            self._rate_limit_last_log = now
            elapsed_min = (now - self._rate_limited_since) / 60.0
            logger.warning(
                "⏳ Still rate-limited by %s after %.0fm — still retrying…",
                self.model, elapsed_min,
            )


    # ───────────────────────────────────────────────────────────────
    #  CENTRAL ERROR HANDLER
    #  Order matters! Subclasses MUST be checked before parents.
    # ───────────────────────────────────────────────────────────────

    def _handle_api_error(self, e: Exception, attempt: int = 0, max_retries: int = 0) -> ErrorAction:
        if getattr(self, '_fatal_error_occurred', False):
            # A fatal error has already occurred. Silence subsequent concurrent errors.
            return ErrorAction.SKIP

        # ── FATAL: Auth / Permissions / Not Found ─────────────────

        if isinstance(e, AuthenticationError):
            logger.error("⛔ AUTH ERROR (%s)", self.model)
            logger.error("   API Key rejected or expired.")
            logger.error("   Key: %s...", self.api_key[:6])
            logger.error("   Error: %s", e)
            return ErrorAction.FATAL

        if isinstance(e, PermissionDeniedError):
            logger.error("⛔ PERMISSION DENIED (%s)", self.model)
            logger.error("   Your API key is valid but lacks access to this resource.")
            logger.error("   Check your plan/tier or model permissions.")
            logger.error("   Error: %s", e)
            return ErrorAction.FATAL

        if isinstance(e, NotFoundError):
            logger.error("⛔ NOT FOUND: Model '%s' does not exist on %s", self.model, self.base_url)
            logger.error("   Error: %s", e)
            return ErrorAction.FATAL

        if e.__class__.__name__ == 'BudgetExceededError':
            logger.error("⛔ BUDGET EXCEEDED — LiteLLM proxy budget limit reached.")
            logger.error("   Error: %s", e)
            return ErrorAction.FATAL

        # ── SKIP: Per-item failures ───────────────────────────────

        if e.__class__.__name__ == 'ContextWindowExceededError':
            logger.warning("⚠️  CONTEXT WINDOW EXCEEDED (%s): Input too long.", self.model)
            logger.warning("   Details: %s", str(e)[:300])
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'ContentPolicyViolationError':
            logger.warning("⚠️  CONTENT POLICY VIOLATION (%s): Safety filter triggered.", self.model)
            logger.warning("   Details: %s", str(e)[:300])
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'UnsupportedParamsError':
            if self.base_url == None:
                logger.warning("⚠️  UNSUPPORTED PARAMS (%s): %s", self.model, str(e)[:300])
                logger.warning("💡 HINT: If using a new model behind litellmproxy, litellm may not detect the provider/model combinations thus its capabilites. Update litellm with pip install --upgrade litellm. Or set the --checker/extractor-base-api to a provider that supports the model. Or set drop_params=True in the LLMClient.")

            return ErrorAction.SKIP

        if e.__class__.__name__ == 'JSONSchemaValidationError':
            return ErrorAction.RESAMPLE

        if isinstance(e, UnprocessableEntityError):
            logger.warning("⚠️  UNPROCESSABLE ENTITY (%s): %s", self.model, str(e)[:300])
            return ErrorAction.SKIP

        # ── CONFIG ERROR: BadRequest base (after subclass checks!) ─

        if isinstance(e, BadRequestError) or e.__class__.__name__ == 'BadRequestError':   
            error_text = ""
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_text = str(e.body).lower()
            else:
                error_text = str(e).lower()        
            if (
                "invalid model" in error_text 
                or "model name" in error_text 
                or "model id" in error_text
                or "provider not provided" in error_text
                or "pass in the llm provider" in error_text
            ):
                logger.error("⛔ CRITICAL: Model Error for '%s' - %s", self.model, e)
                
                if "/" in self.model:
                    prefix, actual_model = self.model.split("/", 1)
                    logger.error("💡 HINT: You are using the prefix '%s/'.", prefix)
                    logger.error("   When custom base_url: a possible error cause is that you MUST NOT use a provider prefix, since it is not using LiteLLM. The provider information is only for the LiteLLM SDK.")
                    logger.error("   -> If that is the case: Change model to '%s' instead of '%s'", actual_model, self.model)
                    logger.error("💡 When using litellm which means no custom base_url: You most likely used to wrong model id.")

                else:
                    logger.error("💡 HINT: The model name was rejected by your base_url. Call `/v1/models` to check available models.")
                
                return ErrorAction.FATAL
        
            # Fallback. During discovery a BadRequest is an expected capability
            # probe (the matrix walks past unsupported strategies), so log it
            # quietly instead of as a scary warning.
            if self._discovering:
                logger.info("   ⬇️  Strategy '%s' not supported here — trying next.", self.strategy.name)
            else:
                logger.warning("⚠️ BAD REQUEST (%s): %s", self.model, str(e)[:300])
            return ErrorAction.SKIP

        # ── RETRY: Transient errors ───────────────────────────────

        retry_label = f"Attempt {attempt + 1}/{max_retries + 1}"

        if isinstance(e, RateLimitError):
            retry_after = self._parse_retry_after(e)
            cap = getattr(default_config, 'RATE_LIMIT_MAX_WAIT', 300.0)
            if retry_after is not None:
                wait = retry_after
                from_header = True
            else:
                base = getattr(default_config, 'RATE_LIMIT_WAIT', 60.0)
                # Jitter the fallback only — it destaggers concurrent tasks so they
                # don't all wake at once and re-trip the limit. Never jitter a
                # server-provided Retry-After: it told us exactly when.
                wait = base * random.uniform(0.9, 1.1)
                from_header = False

            if wait > cap:
                logger.error(
                    "⛔ RATE LIMIT (%s): server asked to wait %.0fs, exceeding cap %.0fs "
                    "(RATE_LIMIT_MAX_WAIT). Aborting.", self.model, wait, cap,
                )
                return ErrorAction.FATAL

            self._rate_limit_wait = wait
            self._log_rate_limit(wait, from_header)
            return ErrorAction.RATE_LIMIT

        # asyncio.TimeoutError is the wall-clock deadline; APITimeoutError the
        # transport backstop. Same condition, same handling.
        if isinstance(e, (APITimeoutError, asyncio.TimeoutError)):
            return ErrorAction.TIMEOUT

        if isinstance(e, APIConnectionError):
            logger.warning("🔄 CONNECTION ERROR (%s) — %s", self.model, retry_label)
            return ErrorAction.SERVER_ERROR

        if isinstance(e, InternalServerError) or e.__class__.__name__ == 'ServiceUnavailableError':
            err_str = str(e).lower()
            if "unexpected keyword argument" in err_str: # catch litellm specific error
                logger.warning("⚠️  UNSUPPORTED PARAMS IN 500 ERROR (%s).", self.model)
                return ErrorAction.SKIP
                
            logger.warning("🔄 SERVER ERROR (%s) — %s", self.model, retry_label) 
            return ErrorAction.SERVER_ERROR# Usually this would be a fatal error, but sometimes it is just a transient error. We retry only very few times then gracefully crash.

        if isinstance(e, ConflictError):
            logger.warning("🔄 CONFLICT (%s) — %s", self.model, retry_label)
            return ErrorAction.RETRY

        if isinstance(e, APIError):
            # Generic APIError fallback — treat as retryable

            # 1. Specific check for 402 (Insufficient Credits / Payment Required)
            status_code = getattr(e, "status_code", None)
            
            if status_code == 402:
                logger.error("⛔ CRITICAL ERROR (402): Out of Credits or Context too large for %s.", self.model)
                logger.error("    Error: %s", e)
                return ErrorAction.FATAL

            # 2. 405 — the endpoint refuses the method outright.
            if status_code == 405:
                logger.error("⛔ ENDPOINT ERROR (405): %s does not accept this request.",
                             self.base_url or "the endpoint")
                logger.error("   Reachable, but not serving an OpenAI-compatible API here.")
                if self.base_url:
                    logger.error("   Check the base URL (a wrong host or a missing /v1 suffix does this).")
                logger.error("   Error: %s", e)
                return ErrorAction.FATAL

            # 3. Generic APIError fallback — treat as retryable (e.g. 500, 502)
            # gotta work this out better! litellm should crash most likely crash. openaisdk NOT
            logger.warning("🔄 API ERROR (%s) — %s: %s", self.model, retry_label, str(e)[:300])
            return ErrorAction.SERVER_ERROR

        if isinstance(e, ValidationError):
            return ErrorAction.RESAMPLE
        # ── UNKNOWN ───────────────────────────────────────────────

        logger.warning("💥 UNEXPECTED ERROR (%s): %s: %s", self.model, type(e).__name__, str(e)[:300])
        return ErrorAction.RETRY


    # ───────────────────────────────────────────────────────────────
    #  GENERATE
    # ───────────────────────────────────────────────────────────────

    async def generate(self, messages: list[dict], schema: Any = None, max_retries=2, task: str = None, **kwargs) -> str:
        """
        Runs one LLM request.
        - base_url set     → OpenAI SDK (direct endpoint, no provider prefix needed)
        - base_url not set → LiteLLM (provider routing via model prefix, e.g. 'openrouter/...')
        - First request discovers the best strategy (serialized via lock).
        - All subsequent requests use the locked strategy concurrently.
        """
        if not self._connection_verified:
            await self.check_connection()

        # ── Discovery: serialize the first request to walk the matrix alone ──
        # All other requests wait at the lock until discovery is done.
        discovering = False
        # A sibling client for the same (base_url, model) may have discovered a
        # strategy after we were constructed (e.g. an atomizer built at boot whose
        # sibling extractor discovers first). Adopt it before paying for our own.
        if not self._strategy_discovered:
            self._try_adopt_cached_strategy()

        # Same for drop_params support — pick up what a sibling already learned.
        self._adopt_cached_drop_params()

        if not self._strategy_discovered:
            await self._discovery_lock.acquire()
            if self._strategy_discovered:
                # Someone else validated while we waited — release and continue
                self._discovery_lock.release()
            else:
                discovering = True
                self._discovering = True
                if self.base_url:
                    logger.info("🔬 Discovering best strategy for %s starting with '%s'...", self.model, self.strategy.name)
                else:
                    logger.info("📡 LiteLLM mode (%s) — validating connection on first request...", self.model)

        try:
            last_error = None
            async with self.sem:
                # Fast-fail: a sibling task already hit a fatal error while we were
                # queued (behind the discovery lock or the semaphore). The whole
                # batch is doomed — abort now instead of firing a doomed request.
                # Raising (not returning a value) keeps this task off the progress
                # bar, so the bar only ever counts genuinely-completed work.
                if self._fatal_error_occurred:
                    raise LLMClientError("aborted: a fatal error already occurred in this batch")

                attempt = 0
                schema_retries = 0
                server_err_count = 0
                timeout_count = 0
                only_parse_failures = True

                while attempt <= max_retries:
                    sent_drop_params = False  # did THIS request carry drop_params?
                    try:
                        if self.base_url:
                            # ── OpenAI SDK Path ────────────────────────────
                            # Strategy sets the default temperature, but caller
                            # kwargs win: the workers' retry rounds vary it to
                            # shake a model out of a bad completion, and a
                            # strategy-owned 0.0 would silently discard that.
                            # Format and reasoning below stay strategy-owned.
                            strategy = self.strategy
                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                "temperature": strategy.temperature,
                                **kwargs,
                            }

                            # Strategy controls reasoning — always overwrites
                            if strategy.reasoning_effort:
                                call_kwargs["reasoning_effort"] = strategy.reasoning_effort
                                # drop_params is LiteLLM-proxy-only. Send it unless
                                # we've learned this is a direct endpoint that rejects
                                # it (learned via the A/B probe in the except handler).
                                if self._drop_params_supported is not False:
                                    existing_extra_body = call_kwargs.get("extra_body", {})
                                    existing_extra_body["drop_params"] = True
                                    call_kwargs["extra_body"] = existing_extra_body
                                    sent_drop_params = True
                                # -----------------------------------------------
                            else:
                                call_kwargs.pop("reasoning_effort", None)
                                # If extra_body was set, clean it up
                                if "extra_body" in call_kwargs and "allowed_openai_params" in call_kwargs["extra_body"]:
                                    call_kwargs["extra_body"]["allowed_openai_params"] = []

                            # Strategy controls output format — always overwrites
                            if schema:
                                if strategy.use_schema:
                                    call_kwargs["response_format"] = type_to_response_format_param(schema)
                                elif strategy.use_json_object:
                                    call_kwargs["response_format"] = {"type": "json_object"}
                                else:
                                    # Vanilla mode — no response_format
                                    call_kwargs.pop("response_format", None)
                                    patched_messages = list(messages)

                                    # Try task-specific vanilla prompt from prompt_map
                                    vanilla_key = f"{task}_vanilla" if task else None
                                    prompts = getattr(default_config, 'PROMPTS', {})
                                    if vanilla_key and vanilla_key in prompts:
                                        # Use human-written vanilla prompt (no schema dump)
                                        logger.debug(
                                            "   📝 Vanilla rung: using hand-written prompt '%s'",
                                            vanilla_key,
                                        )
                                        patched_messages[-1] = {
                                            **patched_messages[-1],
                                            "content": patched_messages[-1]["content"]
                                                + f"\n\n{prompts[vanilla_key]}"
                                        }
                                    else:
                                        # Fallback: compact schema example (no $defs vomit)
                                        logger.debug(
                                            "   📝 Vanilla rung: no '%s' in prompt_map — "
                                            "using generated schema example (fallback)",
                                            vanilla_key or "<no task>",
                                        )
                                        example = build_compact_schema_example(schema)
                                        patched_messages[-1] = {
                                            **patched_messages[-1],
                                            "content": patched_messages[-1]["content"]
                                                + f"\n\nRespond ONLY with valid JSON matching this structure:\n{example}"
                                        }
                                    call_kwargs["messages"] = patched_messages

                            with self._inflight_request():
                                response = await asyncio.wait_for(
                                    self.client.chat.completions.create(
                                        stream=False, **call_kwargs
                                    ),
                                    timeout=self.timeout,
                                )

                        else:
                            # ── LiteLLM Path (no matrix, passthrough) ─────
                            # Lazy import because litellm is HUGE and not always needed.
                            import litellm
                            litellm.suppress_debug_info = True
                            from litellm import acompletion

                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                "api_key": self.api_key,
                                "drop_params": True, # Does NOT try out capability matrix. Trust in Litellm. If you want to do capabilitytesting set base_url even when using a provider.
                                "timeout": self.timeout * TIMEOUT_BACKSTOP_FACTOR,
                                # See max_retries=0 above. num_retries is LiteLLM's own
                                # layer, max_retries the openai client it builds underneath.
                                "max_retries": 0,
                                "num_retries": 0,
                                **kwargs
                            }
                            if schema:
                                call_kwargs["response_format"] = schema

                            with self._inflight_request():
                                response = await asyncio.wait_for(
                                    acompletion(**call_kwargs), timeout=self.timeout
                                )

                        # ── Success ────────────────────────────────────
                        # A 429 episode (if any) has cleared — reset so a later one
                        # logs fresh instead of staying silent.
                        self._rate_limited_since = None
                        # If we sent drop_params and it worked, this is a LiteLLM
                        # proxy: keep sending it for all subsequent requests.
                        if sent_drop_params and self._drop_params_supported is None:
                            self._drop_params_supported = True
                            LLMClient._DROP_PARAMS_CACHE[(self.base_url, self.model)] = True

                        if hasattr(response, 'usage') and response.usage:
                            GLOBAL_STATS.update(response.usage.model_dump())

                        # Validated here, not by .parse(), so a structurally broken
                        # response is still counted above before it raises.
                        if self.base_url and schema and self.strategy.use_schema:
                            choice = response.choices[0]
                            if choice.finish_reason == "length":
                                raise LengthFinishReasonError(completion=response)
                            if choice.finish_reason == "content_filter":
                                raise ContentFilterFinishReasonError()
                            schema.model_validate_json(choice.message.content)

                        # Lock strategy on first success
                        if discovering and not self._strategy_discovered:
                            self._strategy_discovered = True
                            self._discovery_succeeded = True
                            LLMClient._STRATEGY_CACHE[(self.base_url, self.model)] = self._strategy_index
                            logger.info("   🔒 Strategy locked: '%s'", self.strategy.name)

                        # Cache hint (only log once to avoid spam) Catch it properly
                        if not self._cache_hit_logged:
                            cache_hit = getattr(response, '_hidden_params', {}).get('cache_hit', False)
                            if cache_hit:
                                logger.info("   💾 Cache hit detected — provider is caching responses.")
                                self._cache_hit_logged = True

                        return response.choices[0].message.content

                    except (TypeError, AttributeError, KeyError, NameError, SyntaxError, ImportError) as e:
                        # Local coding bugs
                        logger.error("CODE BUG (not an API error): %s: %s", type(e).__name__, e)
                        logger.error("%s", traceback.format_exc())
                        self._save_and_raise(f"Code bug: {type(e).__name__}: {e}")

                    except Exception as e:
                        last_error = e

                        # A sibling task already triggered a fatal abort for this
                        # client. Don't reclassify our error as a per-item SKIP —
                        # that returns a value and falsely advances the progress
                        # bar. Propagate quietly so the batch stops immediately and
                        # the bar reflects only genuinely-completed work.
                        if self._fatal_error_occurred:
                            raise LLMClientError("aborted: a fatal error already occurred in this batch") from e

                        action = self._handle_api_error(e, attempt, max_retries)

                        if action != ErrorAction.SERVER_ERROR:
                            server_err_count = 0

                        if action != ErrorAction.RESAMPLE:
                            only_parse_failures = False

                        if action == ErrorAction.FATAL:
                            self._save_and_raise(f"FATAL: {type(e).__name__} — {str(e)[:200]}")

                        # A/B probe for the LiteLLM-only `drop_params` field: if a
                        # request that carried drop_params just failed and we haven't
                        # learned the endpoint type yet, assume this is a direct
                        # endpoint that rejects drop_params. Turn it off and retry the
                        # SAME strategy (keeps reasoning). If it fails again it was a
                        # real capability gap and we advance the matrix normally.
                        _is_bad_request = (
                            isinstance(e, BadRequestError)
                            or e.__class__.__name__ == 'BadRequestError'
                        )
                        if sent_drop_params and self._drop_params_supported is None and _is_bad_request:
                            self._drop_params_supported = False
                            LLMClient._DROP_PARAMS_CACHE[(self.base_url, self.model)] = False
                            logger.info(
                                "   🔎 Endpoint rejected 'drop_params' — direct endpoint "
                                "detected, retrying without it (keeping reasoning)."
                            )
                            continue  # retry same strategy, drop_params now off

                        # During discovery: advance strategy on capability errors
                        # This does NOT count as a retry attempt except if it is a fatal error
                        is_capability_error = (
                            isinstance(e, BadRequestError) 
                            or e.__class__.__name__ == 'UnsupportedParamsError'
                            or (isinstance(e, InternalServerError) and "unexpected keyword argument" in str(e).lower()) # Litellm can throw this for ANY unsupported param
                        ) and not (
                            e.__class__.__name__ in ('ContextWindowExceededError', 'ContentPolicyViolationError')
                        )

                        if is_capability_error and discovering and self._next_strategy():
                            schema_retries = 0  # reset — give new strategy its full 3 attempts
                            continue  # same attempt counter, just different strategy

                        if isinstance(e, ValidationError) and discovering:
                            if schema_retries < 3:
                                logger.warning("   ⚠️ Schema Error. Retrying same strategy (%d/3)...", schema_retries + 1)
                                schema_retries += 1
                                continue
                            else:
                                logger.warning("   ❌ Model failed JSON schema 3 times. Downgrading strategy...")
                                schema_retries = 0  # reset before downgrade
                                if self._next_strategy():
                                    continue  # same attempt counter, just different strategy

                        elif action == ErrorAction.SKIP:
                            GLOBAL_STATS.log_error()
                            # Raise typed exception instead of returning ""
                            if e.__class__.__name__ == 'ContextWindowExceededError':
                                raise ContextTooLongError(str(e)) from e
                            elif e.__class__.__name__ == 'ContentPolicyViolationError':
                                raise ContentPolicyError(str(e)) from e
                            else:
                                raise LLMParseError(str(e)) from e

                        elif action == ErrorAction.RATE_LIMIT:
                            # Infinite retries: a 429 never counts against `attempt`,
                            # so the item is never dropped. Wait honors Retry-After
                            # (or the jittered fallback) computed in the handler.
                            await asyncio.sleep(self._rate_limit_wait)
                            continue

                        elif action == ErrorAction.TIMEOUT:
                            timeout_count += 1
                            if timeout_count > TIMEOUT_RETRIES:
                                GLOBAL_STATS.log_error()
                                raise LLMTimeoutError(str(e)) from e
                            continue

                        elif action == ErrorAction.SERVER_ERROR:
                            server_err_count += 1
                            if server_err_count > 3:
                                self._save_and_raise(f"FATAL: Infrastructure failure. Aborting after 3 consecutive Server Errors. Last error: {str(e)[:1000]}")
                            wait_time = 5.0 * server_err_count 
                            logger.info("   ⏳ Server Error: sleeping %ss...", wait_time)
                            await asyncio.sleep(wait_time)
                            continue

                        elif action == ErrorAction.RESAMPLE:
                            # No backoff, waiting cannot make output parseable.
                            if attempt < max_retries:
                                attempt += 1
                                continue
                            else:
                                break

                        elif action == ErrorAction.RETRY:
                            if attempt < max_retries:
                                wait_time = 0.5 * (attempt + 1)
                                logger.info("   ⏳ Waiting %ss before retry...", wait_time)
                                await asyncio.sleep(wait_time)
                                attempt += 1
                                continue
                            else:
                                break

                # All retries exhausted for this try. Breaking
                if not only_parse_failures:
                    logger.error("🔴 FAILED after %d attempts. Last error: %s", attempt + 1, _describe_error(last_error))
                GLOBAL_STATS.log_error()
                raise LLMParseError(f"Exhausted {attempt + 1} retries: {str(last_error)[:200]}") from last_error

        finally:
            self._discovering = False
            # Release the discovery lock if we hold it
            if discovering:
                self._strategy_discovered = True  # lock at current level (already walked past incapable strategies)
                if self._discovery_lock.locked():
                    self._discovery_lock.release()
                if not self._discovery_succeeded:
                    # Don't mask a fatal error or cancellation that's already propagating
                    current_exc = sys.exc_info()[1]
                    if current_exc is not None and (
                        isinstance(current_exc, LLMClientError)
                        or not isinstance(current_exc, Exception)
                    ):
                        pass  # let fatal errors and system/cancellation exceptions propagate
                    else:
                        raise LLMClientError(
                            f"No compatible strategy found for '{self.model}'. "
                            f"Exhausted all {len(RETRY_MATRIX)} strategies."
                        )



    # ── Fatal checkpoint support ──────────────────────────────

    def _save_and_raise(self, message: str):
        """Log the error and raise LLMClientError, aborting the batch."""
        self._fatal_error_occurred = True
        logger.error("💀 %s", message)
        raise LLMClientError(message)

    async def _generate_safe(self, **task_args) -> str | WorkerError:
        """Wrapper that catches per-item errors and returns them as values.

        Fatal errors (LLMClientError) propagate — they kill the batch.
        Per-item errors (ContextTooLong, ContentPolicy, Parsing) are returned
        as values so the worker can classify and retry them.
        """
        try:
            return await self.generate(**task_args)
        except (ContextTooLongError, ContentPolicyError, LLMParseError, LLMTimeoutError) as e:
            return e
        # LLMClientError and anything else propagates → kills batch


    async def generate_batch(self, tasks_data: list[dict], description="Processing", task: str = None) -> list[str | WorkerError]:
        """
        Batch helper. Returns List[Union[str, WorkerError]].
        Successful items are strings, failed items are WorkerError instances.

        Args:
            tasks_data: List of dicts with messages, schema, temperature, etc.
            description: tqdm progress bar label
            task: "extract" or "check" — sets GLOBAL_STATS phase and routes vanilla prompts
        """
        if not self._connection_verified:
            await self.check_connection()

        if task:
            GLOBAL_STATS.set_phase(task)

        async def _run_safe(task_args):
            return await self._generate_safe(task=task, **task_args)

        # Hermetic progress bar: use standard non-async tqdm and update it manually
        # inside standard asyncio.gather, ensuring perfect exception cancellation.
        pbar = tqdm(total=len(tasks_data), desc="  " + description)

        failed = 0
        timed_out = 0

        def _set_postfix():
            bits = []
            waiting = self.oldest_inflight_age()
            # Below a few seconds the number is noise that flickers every tick.
            if waiting is not None and waiting >= HEARTBEAT_MIN_WAIT:
                bits.append(f"waiting {waiting:.0f}s/{self.timeout:.0f}s")
            if failed:
                bits.append(f"{failed} failed")
            if timed_out:
                bits.append(f"{timed_out} timed out")
            pbar.set_postfix_str(" · ".join(bits), refresh=False)

        async def _run_and_update(args):
            nonlocal failed, timed_out
            completed = False
            try:
                res = await _run_safe(args)
                completed = True
                # A timeout is the line, not the model — counted apart.
                if isinstance(res, LLMTimeoutError):
                    timed_out += 1
                elif isinstance(res, Exception):
                    failed += 1
                return res
            finally:
                if completed:
                    _set_postfix()
                    pbar.update(1)

        async def _heartbeat():
            """Repaint on a timer so the elapsed clock moves while requests hang."""
            while True:
                await _ui_sleep(1.0)
                _set_postfix()
                pbar.refresh()

        calls_before = self._call_n
        tasks = [asyncio.create_task(_run_and_update(args)) for args in tasks_data]
        heartbeat = asyncio.create_task(_heartbeat())
        with logging_redirect_tqdm(loggers=[logger.parent]):
            try:
                results = await asyncio.gather(*tasks)
            except BaseException as e:
                # Cancel all other tasks immediately!
                for t in tasks:
                    if not t.done():
                        t.cancel()
                # Wait for all tasks to be cancelled/done to avoid orphaned tasks
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                heartbeat.cancel()
                self._last_batch_requests = self._call_n - calls_before
                pbar.close()

        # Scan for fatal LLMClientError that leaked through as a value (just in case)
        for r in results:
            if isinstance(r, LLMClientError) and not isinstance(r, (ContextTooLongError, ContentPolicyError, LLMTimeoutError)):
                raise r

        return results


    async def check_connection(self):
        """Pre-flight check: verifies API reachability and authentication."""
        if not self.base_url:
            # LiteLLM mode — no direct endpoint to check, skip pre-flight
            self._connection_verified = True
            return

        if self.base_url in LLMClient._VERIFIED_ENDPOINTS:
            # Already probed this endpoint earlier in the process — skip the round-trip.
            self._connection_verified = True
            logger.debug(
                "   📡 Connection cache hit — %s already verified, skipping pre-flight",
                self.base_url,
            )
            return

        logger.info("📡 Testing connection to %s/models...", self.base_url)
        try:
            models_response = await self.client.models.list()
            logger.info("   ✅ Connection confirmed. Server reachable")
            self._connection_verified = True
            LLMClient._VERIFIED_ENDPOINTS.add(self.base_url)

            # Show available models in debug mode — helps diagnose model name typos
            model_ids = sorted([m.id for m in models_response.data])
            if model_ids:
                logger.debug("   Available models (%d):", len(model_ids))
                for mid in model_ids:
                    logger.debug("     - %s", mid)

        except AuthenticationError as e:
            logger.error("❌ FATAL: Authentication Failed.")
            logger.error("   Key: %s...", self.api_key[:6])
            logger.error("   Error: %s", e)
            raise LLMClientError(f"Authentication failed for {self.base_url}. Check your API key.") from e

        except PermissionDeniedError as e:
            logger.error("❌ FATAL: Permission Denied.")
            logger.error("   Your key is valid but cannot access this endpoint.")
            logger.error("   Error: %s", e)
            raise LLMClientError(f"Permission denied for {self.base_url}. Check your API plan/tier.") from e

        except NotFoundError as e:
            # /v1/models may not exist on all custom endpoints
            logger.info("   ⚡ /models endpoint not available (HTTP %s) — skipping pre-flight check.",
                        getattr(e, "status_code", "?"))
            self._connection_verified = True
            LLMClient._VERIFIED_ENDPOINTS.add(self.base_url)

        except APITimeoutError as e:
            # APITimeoutError extends APIConnectionError — must be caught FIRST
            logger.error("❌ FATAL: Connection timed out during pre-flight check.")
            logger.error("   URL: %s", self.base_url)
            logger.error("   Error: %s", e)
            raise LLMClientError(f"Timeout connecting to {self.base_url}") from e

        except APIConnectionError as e:
            logger.error("❌ FATAL: Cannot connect to API endpoint.")
            logger.error("   URL: %s", self.base_url)
            logger.error("   Error: %s", e)
            logger.error("   Check: Is the URL correct? Is the server running? Firewall/proxy issues? Are you online?")
            # TODO: Add CLI flag to skip pre-flight model check
            raise LLMClientError(f"Cannot connect to {self.base_url}") from e

        except Exception as e:
            logger.error("❌ FATAL: Unexpected error during connection check.")
            logger.error("   Type: %s", type(e).__name__)
            logger.error("   Error: %s", e)
            raise LLMClientError(f"{type(e).__name__} in check_connection: {e}") from e
