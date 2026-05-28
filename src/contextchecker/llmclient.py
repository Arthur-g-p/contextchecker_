import asyncio
import os
import sqlite3
import sys
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
)
from tqdm import tqdm
from pydantic import ValidationError

from contextchecker.stats import GLOBAL_STATS
from contextchecker import settings as default_config
from contextchecker.exceptions import LLMClientError, LLMError, ContextTooLongError, ContentPolicyError, LLMParseError
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


class ErrorAction(Enum):
    """What to do when an API error occurs."""
    FATAL = "fatal"   # Exit program — unrecoverable
    SKIP  = "skip"    # Return "", continue batch — per-item failure
    RETRY = "retry"   # Backoff and retry — transient
    RATE_LIMIT = "rate_limit" # Long backoff, no attempt increment
    SERVER_ERROR = "server_error" # Independent attempts backoff for infrastructure


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None, concurrency: int = 10, cache_file: str | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        
        self.timeout = getattr(default_config, 'LLM_TIMEOUT', 120.0)

        # OpenAI SDK client — only created if base_url is set (direct endpoint mode)
        if self.base_url:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
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

        self._session_cache = {}
        self._new_cache_entries = 0
        self._cache_enabled = True
        self._cache_loaded_from_disk = False
        self._cache_file = cache_file or ".rag_crash_cache.db"
        if os.path.exists(self._cache_file):
            try:
                with sqlite3.connect(self._cache_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'")
                    if cursor.fetchone():
                        cursor.execute("SELECT hash, response FROM cache")
                        for row in cursor.fetchall():
                            self._session_cache[row[0]] = row[1]
                        self._cache_loaded_from_disk = True
                        logger.info("   ♻️  Found %d cached responses in %s", len(self._session_cache), self._cache_file)
            except Exception as e:
                logger.warning("   ⚠️  Failed to load crash cache. No optional caching will take place. (%s)", e)
                self._cache_enabled = False

        sdk_mode = "OpenAI SDK" if self.base_url else "LiteLLM"
        logger.info("   LLMClient initialized: %s via %s%s", self.model, sdk_mode, f" @ {base_url}" if base_url else "")


    @property
    def strategy(self) -> RetryStrategy:
        """Current retry strategy."""
        return RETRY_MATRIX[self._strategy_index]


    def _next_strategy(self) -> bool:
        """Advance to next strategy. Returns True if advanced, False if at bottom."""
        if self._strategy_index < len(RETRY_MATRIX) - 1:
            self._strategy_index += 1
            logger.info("   ⬇️  Next strategy: '%s'", self.strategy.name)
            return True
        return False


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
                logger.warning("💡 HINT: If using a new model, litellm may not detect the provider/model combinations thus its capabilites. Update litellm with pip install --upgrade litellm. Or set the --checker/extractor-base-api to a provider that supports the model. Or set drop_params=True in the LLMClient.")

            return ErrorAction.SKIP

        if e.__class__.__name__ == 'JSONSchemaValidationError':
            logger.warning("⚠️  SCHEMA VALIDATION FAILED (%s): %s", self.model, str(e)[:300])
            return ErrorAction.RETRY

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
        
            # Fallback 
            logger.warning("⚠️ BAD REQUEST (%s): %s", self.model, str(e)[:300])
            return ErrorAction.SKIP

        # ── RETRY: Transient errors ───────────────────────────────

        retry_label = f"Attempt {attempt + 1}/{max_retries + 1}"

        if isinstance(e, RateLimitError):
            logger.warning("🔄 RATE LIMITED (%s) — %s. Waiting 60s...", self.model, retry_label)
            return ErrorAction.RATE_LIMIT

        if isinstance(e, APITimeoutError):
            logger.warning("🔄 TIMEOUT (%s) — %s", self.model, retry_label)
            return ErrorAction.SERVER_ERROR

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

            # 2. Generic APIError fallback — treat as retryable (e.g. 500, 502)
            # gotta work this out better! litellm should crash most likely crash. openaisdk NOT
            logger.warning("🔄 API ERROR (%s) — %s: %s", self.model, retry_label, str(e)[:300])
            return ErrorAction.SERVER_ERROR

        if isinstance(e, ValidationError):
            # Format validation errors nicely
            error_details = []
            for err in e.errors():
                loc = " -> ".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "")
                error_details.append(f"{loc}: {msg}")
            details_str = " | ".join(error_details)
            logger.warning("⚠️ LOCAL SCHEMA VALIDATION FAILED (%s): %s", self.model, details_str)
            return ErrorAction.RETRY 
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
        prompt_text = messages[-1]["content"] if isinstance(messages, list) and messages and "content" in messages[-1] else None
        cache_key = f"{self.model}:{prompt_text}" if prompt_text else None
        if self._cache_loaded_from_disk and cache_key and cache_key in self._session_cache:
            return self._session_cache[cache_key]

        if not self._connection_verified:
            await self.check_connection()

        # ── Discovery: serialize the first request to walk the matrix alone ──
        # All other requests wait at the lock until discovery is done.
        discovering = False
        if not self._strategy_discovered:
            await self._discovery_lock.acquire()
            if self._strategy_discovered:
                # Someone else validated while we waited — release and continue
                self._discovery_lock.release()
            else:
                discovering = True
                if self.base_url:
                    logger.info("🔬 Discovering best strategy for %s starting with '%s'...", self.model, self.strategy.name)
                else:
                    logger.info("📡 LiteLLM mode (%s) — validating connection on first request...", self.model)

        try:
            last_error = None
            async with self.sem:
                attempt = 0
                schema_retries = 0
                server_err_count = 0

                while attempt <= max_retries:
                    try:
                        if self.base_url:
                            # ── OpenAI SDK Path ────────────────────────────
                            # kwargs go in first, strategy overwrites on top
                            strategy = self.strategy
                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                **kwargs,
                                "temperature": strategy.temperature
                            }

                            # Strategy controls reasoning — always overwrites
                            if strategy.reasoning_effort:
                                call_kwargs["reasoning_effort"] = strategy.reasoning_effort
                                existing_extra_body = call_kwargs.get("extra_body", {})
                                existing_extra_body["drop_params"] = True
                                call_kwargs["extra_body"] = existing_extra_body
                                # -----------------------------------------------
                            else:
                                call_kwargs.pop("reasoning_effort", None)
                                # If extra_body was set, clean it up
                                if "extra_body" in call_kwargs and "allowed_openai_params" in call_kwargs["extra_body"]:
                                    call_kwargs["extra_body"]["allowed_openai_params"] = []

                            # Strategy controls output format — always overwrites
                            if schema:
                                if strategy.use_schema:
                                    call_kwargs["response_format"] = schema
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
                                        patched_messages[-1] = {
                                            **patched_messages[-1],
                                            "content": patched_messages[-1]["content"]
                                                + f"\n\n{prompts[vanilla_key]}"
                                        }
                                    else:
                                        # Fallback: compact schema example (no $defs vomit)
                                        example = build_compact_schema_example(schema)
                                        patched_messages[-1] = {
                                            **patched_messages[-1],
                                            "content": patched_messages[-1]["content"]
                                                + f"\n\nRespond ONLY with valid JSON matching this structure:\n{example}"
                                        }
                                    call_kwargs["messages"] = patched_messages

                            response = await self.client.chat.completions.parse(**call_kwargs)

                        else:
                            # ── LiteLLM Path (no matrix, passthrough) ─────
                            import litellm
                            litellm.suppress_debug_info = True
                            from litellm import acompletion

                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                "api_key": self.api_key,
                                "drop_params": True, # Does NOT try out capability matrix. Trust in Litellm. If you want to do capabilitytesting set base_url even when using a provider.
                                "timeout": self.timeout,
                                **kwargs
                            }
                            if schema:
                                call_kwargs["response_format"] = schema

                            response = await acompletion(**call_kwargs)

                        # ── Success ────────────────────────────────────
                        if hasattr(response, 'usage') and response.usage:
                            GLOBAL_STATS.update(response.usage.model_dump())

                        # Lock strategy on first success
                        if discovering and not self._strategy_discovered:
                            self._strategy_discovered = True
                            self._discovery_succeeded = True
                            logger.info("   🔒 Strategy locked: '%s'", self.strategy.name)

                        # Cache hint (only log once to avoid spam) Catch it properly
                        if not self._cache_hit_logged:
                            cache_hit = getattr(response, '_hidden_params', {}).get('cache_hit', False)
                            if cache_hit:
                                logger.info("   💾 Cache hit detected — provider is caching responses.")
                                self._cache_hit_logged = True

                        response_str = response.choices[0].message.content
                        if cache_key:
                            self._session_cache[cache_key] = response_str
                            self._new_cache_entries += 1
                        return response_str

                    except (TypeError, AttributeError, KeyError, NameError, SyntaxError, ImportError) as e:
                        # Local coding bugs
                        logger.error("CODE BUG (not an API error): %s: %s", type(e).__name__, e)
                        logger.error("%s", traceback.format_exc())
                        self._save_and_raise(f"Code bug: {type(e).__name__}: {e}")

                    except Exception as e:
                        last_error = e
                        action = self._handle_api_error(e, attempt, max_retries)

                        if action != ErrorAction.SERVER_ERROR:
                            server_err_count = 0

                        if action == ErrorAction.FATAL:
                            self._save_and_raise(f"FATAL: {type(e).__name__} — {str(e)[:200]}")

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
                            await asyncio.sleep(60)
                            continue

                        elif action == ErrorAction.SERVER_ERROR:
                            server_err_count += 1
                            if server_err_count > 3:
                                self._save_and_raise(f"FATAL: Infrastructure failure. Aborting after 3 consecutive Server Errors. Last error: {str(e)[:1000]}")
                            wait_time = 5.0 * server_err_count 
                            logger.info("   ⏳ Server Error: sleeping %ss...", wait_time)
                            await asyncio.sleep(wait_time)
                            continue

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
                logger.error("🔴 FAILED after %d attempts. Last error: %s", attempt + 1, str(last_error)[:100])
                GLOBAL_STATS.log_error()
                raise LLMParseError(f"Exhausted {attempt + 1} retries: {str(last_error)[:200]}") from last_error

        finally:
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
                        self.save_cache()
                        raise LLMClientError(
                            f"No compatible strategy found for '{self.model}'. "
                            f"Exhausted all {len(RETRY_MATRIX)} strategies."
                        )



    # ── Fatal checkpoint support ──────────────────────────────

    def save_cache(self):
        """Dump the in-memory session cache to SQLite on crash."""
        if not getattr(self, '_cache_enabled', False) or getattr(self, '_new_cache_entries', 0) == 0:
            return
        try:
            with sqlite3.connect(self._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, response TEXT)")
                cursor.executemany("INSERT OR REPLACE INTO cache (hash, response) VALUES (?, ?)", list(self._session_cache.items()))
                conn.commit()
            logger.info("   💾 Saved %d total responses to %s for rescue.", len(self._session_cache), self._cache_file)
        except Exception:
            pass # Swallow errors on shutdown

    def _save_and_raise(self, message: str):
        """Log error, save partial results to crash cache, and raise LLMClientError."""
        self._fatal_error_occurred = True
        logger.error("💀 %s", message)
        self.save_cache()
        raise LLMClientError(message)

    async def _generate_safe(self, **task_args) -> str | LLMError:
        """Wrapper that catches per-item errors and returns them as values.

        Fatal errors (LLMClientError) propagate — they kill the batch.
        Per-item errors (ContextTooLong, ContentPolicy, Parsing) are returned
        as values so the worker can classify and retry them.
        """
        try:
            return await self.generate(**task_args)
        except (ContextTooLongError, ContentPolicyError, LLMParseError) as e:
            return e
        # LLMClientError and anything else propagates → kills batch


    async def generate_batch(self, tasks_data: list[dict], description="Processing", task: str = None) -> list[str | LLMError]:
        """
        Batch helper. Returns List[Union[str, LLMError]].
        Successful items are strings, failed items are LLMError instances.
        
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

        async def _run_and_update(args):
            completed = False
            try:
                res = await _run_safe(args)
                completed = True
                return res
            finally:
                if completed:
                    pbar.update(1)

        tasks = [asyncio.create_task(_run_and_update(args)) for args in tasks_data]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException as e:
            # Cancel all other tasks immediately!
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Wait for all tasks to be cancelled/done to avoid orphaned tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            self.save_cache()
            raise
        finally:
            pbar.close()

        # Scan for fatal LLMClientError that leaked through as a value (just in case)
        for r in results:
            if isinstance(r, LLMClientError) and not isinstance(r, (ContextTooLongError, ContentPolicyError)):
                self.save_cache()
                raise r

        return results


    async def check_connection(self):
        """Pre-flight check: verifies API reachability and authentication."""
        if not self.base_url:
            # LiteLLM mode — no direct endpoint to check, skip pre-flight
            self._connection_verified = True
            return

        logger.info("📡 Testing connection to %s/models...", self.base_url)
        try:
            models_response = await self.client.models.list()
            logger.info("   ✅ Connection confirmed. Server reachable")
            self._connection_verified = True

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
            logger.info("   ⚡ /models endpoint not available — skipping pre-flight check.")
            self._connection_verified = True

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
            logger.error("   Check: Is the URL correct? Is the server running? Firewall/proxy issues?")
            # TODO: Add CLI flag to skip pre-flight model check
            raise LLMClientError(f"Cannot connect to {self.base_url}") from e

        except Exception as e:
            logger.error("❌ FATAL: Unexpected error during connection check.")
            logger.error("   Type: %s", type(e).__name__)
            logger.error("   Error: %s", e)
            raise LLMClientError(f"{type(e).__name__} in check_connection: {e}") from e
