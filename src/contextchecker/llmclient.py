import asyncio
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from openai import (
    AsyncOpenAI,
    APIError, APIStatusError, APIConnectionError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, BadRequestError,
    NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError,
)
from tqdm.asyncio import tqdm_asyncio
import os
from contextchecker.stats import GLOBAL_STATS
from contextchecker import settings as default_config
from contextchecker.exceptions import LLMError, ContextTooLongError, ContentPolicyError, LLMParseError
from contextchecker.utils import build_compact_schema_example
from pydantic import ValidationError


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
    reasoning_effort: Optional[str] = None   # "low", "medium", "high" — OpenAI standard. None = don't send.
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
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, concurrency: int = 10):
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

        self._session_cache = {}
        self._new_cache_entries = 0
        self._cache_enabled = True
        self._cache_loaded_from_disk = False
        self._cache_file = ".rag_crash_cache.db"
        if os.path.exists(self._cache_file):
            try:
                import sqlite3
                with sqlite3.connect(self._cache_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'")
                    if cursor.fetchone():
                        cursor.execute("SELECT hash, response FROM cache")
                        for row in cursor.fetchall():
                            self._session_cache[row[0]] = row[1]
                        self._cache_loaded_from_disk = True
                        print(f"   ♻️  Found {len(self._session_cache)} cached responses in {self._cache_file}")
            except Exception as e:
                print(f"   ⚠️  Failed to load crash cache. No optional caching will take place. ({e})")
                self._cache_enabled = False

        sdk_mode = "OpenAI SDK" if self.base_url else "LiteLLM"
        print(f"   LLMClient initialized: {self.model} via {sdk_mode}" + (f" @ {base_url}" if base_url else ""))


    @property
    def strategy(self) -> RetryStrategy:
        """Current retry strategy."""
        return RETRY_MATRIX[self._strategy_index]


    def _next_strategy(self) -> bool:
        """Advance to next strategy. Returns True if advanced, False if at bottom."""
        if self._strategy_index < len(RETRY_MATRIX) - 1:
            self._strategy_index += 1
            print(f"   ⬇️  Next strategy: '{self.strategy.name}'")
            return True
        return False


    # ───────────────────────────────────────────────────────────────
    #  CENTRAL ERROR HANDLER
    #  Order matters! Subclasses MUST be checked before parents.
    # ───────────────────────────────────────────────────────────────

    def _handle_api_error(self, e: Exception, attempt: int = 0, max_retries: int = 0) -> ErrorAction:

        # ── FATAL: Auth / Permissions / Not Found ─────────────────

        if isinstance(e, AuthenticationError):
            print(f"\n⛔ AUTH ERROR ({self.model})")
            print(f"   API Key rejected or expired.")
            print(f"   Key: {self.api_key[:6]}...")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if isinstance(e, PermissionDeniedError):
            print(f"\n⛔ PERMISSION DENIED ({self.model})")
            print(f"   Your API key is valid but lacks access to this resource.")
            print(f"   Check your plan/tier or model permissions.")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if isinstance(e, NotFoundError):
            print(f"\n⛔ NOT FOUND: Model '{self.model}' does not exist on {self.base_url}")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if e.__class__.__name__ == 'BudgetExceededError':
            print(f"\n⛔ BUDGET EXCEEDED — LiteLLM proxy budget limit reached.")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        # ── SKIP: Per-item failures ───────────────────────────────

        if e.__class__.__name__ == 'ContextWindowExceededError':
            print(f"⚠️  CONTEXT WINDOW EXCEEDED ({self.model}): Input too long.")
            print(f"   Details: {str(e)[:300]}")
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'ContentPolicyViolationError':
            print(f"⚠️  CONTENT POLICY VIOLATION ({self.model}): Safety filter triggered.")
            print(f"   Details: {str(e)[:300]}")
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'UnsupportedParamsError':
            if self.base_url == None:
                print(f"⚠️  UNSUPPORTED PARAMS ({self.model}): {str(e)[:300]}")
                print(f"💡 HINT: If using a new model, litellm may not detect the provider/model combinations thus its capabilites. Update litellm with pip install --upgrade litellm. Or set the --checker/extractor-base-api to a provider that supports the model. Or set drop_params=True in the LLMClient.")

            return ErrorAction.SKIP

        if e.__class__.__name__ == 'JSONSchemaValidationError':
            print(f"⚠️  SCHEMA VALIDATION FAILED ({self.model}): {str(e)[:300]}")
            return ErrorAction.RETRY

        if isinstance(e, UnprocessableEntityError):
            print(f"⚠️  UNPROCESSABLE ENTITY ({self.model}): {str(e)[:300]}")
            return ErrorAction.SKIP

        # ── CONFIG ERROR: BadRequest base (after subclass checks!) ─

        if isinstance(e, BadRequestError):   
            error_text = ""
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_text = str(e.body).lower()
            else:
                error_text = str(e).lower()        
            if "invalid model" in error_text or "model name" in error_text or "model id" in error_text:  #hardcode lite llm invalid model error catch
                print(f"\n⛔ CRITICAL: Model Error for '{self.model}' - {e}")
                
                if "/" in self.model:
                    prefix, actual_model = self.model.split("/", 1)
                    print(f"💡 HINT: You are using the prefix '{prefix}/'.")
                    print(f"   When custom base_url: a possible error cause is that you MUST NOT use a provider prefix, since it is not using LiteLLM. The provider information is only for the LiteLLM SDK.")
                    print(f"   -> If that is the case: Change model to '{actual_model}' instead of '{self.model}'\n")
                    print(f"💡 When using litellm which means no custom base_url: You most likely used to wrong model id.")

                else:
                    print(f"💡 HINT: The model name was rejected by your base_url. Call `/v1/models` to check available models.\n")
                
                return ErrorAction.FATAL
        
            # Fallback 
            print(f"⚠️ BAD REQUEST ({self.model}): {str(e)[:300]}")
            # does not catch BAD REQUEST (google/gemini-3.1-flash-lite-preview): litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=google/gemini-3.1-flash-lite-preview
            return ErrorAction.SKIP

        # ── RETRY: Transient errors ───────────────────────────────

        retry_label = f"Attempt {attempt + 1}/{max_retries + 1}"

        if isinstance(e, RateLimitError):
            print(f"🔄 RATE LIMITED ({self.model}) — {retry_label}. Waiting 60s...")
            return ErrorAction.RATE_LIMIT

        if isinstance(e, APITimeoutError):
            print(f"🔄 TIMEOUT ({self.model}) — {retry_label}")
            return ErrorAction.SERVER_ERROR

        if isinstance(e, APIConnectionError):
            print(f"🔄 CONNECTION ERROR ({self.model}) — {retry_label}")
            return ErrorAction.SERVER_ERROR

        if isinstance(e, InternalServerError) or e.__class__.__name__ == 'ServiceUnavailableError':
            err_str = str(e).lower()
            if "unexpected keyword argument" in err_str: # catch litellm specific error
                print(f"⚠️  UNSUPPORTED PARAMS IN 500 ERROR ({self.model}).")
                return ErrorAction.SKIP
                
            print(f"🔄 SERVER ERROR ({self.model}) — {retry_label}") 
            return ErrorAction.SERVER_ERROR# Usually this would be a fatal error, but sometimes it is just a transient error. We retry only very few times then gracefully crash.

        if isinstance(e, ConflictError):
            print(f"🔄 CONFLICT ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, APIError):
            # Generic APIError fallback — treat as retryable

            # 1. Spezifischer Check auf 402 (Insufficient Credits / Payment Required)
            status_code = getattr(e, "status_code", None)
            
            if status_code == 402:
                print(f"\n⛔ CRITICAL ERROR (402): Out of Credits or Context too large for {self.model}.")
                print(f"    Error: {e}")
                return ErrorAction.FATAL

            # 2. Generic APIError fallback — treat as retryable (z.B. 500, 502)
            # gotta work this out better! litellm should crash most likely crash. openaisdk NOT
            print(f"🔄 API ERROR ({self.model}) — {retry_label}: {str(e)[:300]}")
            return ErrorAction.SERVER_ERROR

        if isinstance(e, ValidationError):
            print(f"⚠️ LOCAL SCHEMA VALIDATION FAILED ({self.model}): Model generated incomplete JSON.")
            return ErrorAction.RETRY 
        # ── UNKNOWN ───────────────────────────────────────────────

        print(f"💥 UNEXPECTED ERROR ({self.model}): {type(e).__name__}: {str(e)[:300]}")
        return ErrorAction.RETRY


    # ───────────────────────────────────────────────────────────────
    #  GENERATE
    # ───────────────────────────────────────────────────────────────

    async def generate(self, messages: List[Dict], schema: Any = None, max_retries=2, task: str = None, **kwargs) -> str:
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
        if self.base_url and not self._strategy_discovered:
            await self._discovery_lock.acquire()
            if self._strategy_discovered:
                # Someone else discovered while we waited — release and continue
                self._discovery_lock.release()
            else:
                discovering = True
                print(f"🔬 Discovering best strategy for {self.model}...")

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
                                # Falls extra_body gesetzt war, bereinigen wir das
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
                            print(f"   🔒 Strategy locked: '{self.strategy.name}'")

                        # Cache hint (only log once to avoid spam) Catch it properly
                        if not self._cache_hit_logged:
                            cache_hit = getattr(response, '_hidden_params', {}).get('cache_hit', False)
                            if cache_hit:
                                print(f"   💾 Cache hit detected — provider is caching responses.")
                                self._cache_hit_logged = True

                        response_str = response.choices[0].message.content
                        if cache_key:
                            self._session_cache[cache_key] = response_str
                            self._new_cache_entries += 1
                        return response_str

                    except (TypeError, AttributeError, KeyError, NameError, SyntaxError, ImportError) as e:
                        # Local coding bugs
                        import traceback
                        print(f"\n CODE BUG (not an API error): {type(e).__name__}: {e}")
                        traceback.print_exc()
                        self._save_and_die(f"Code bug: {type(e).__name__}: {e}")

                    except Exception as e:
                        last_error = e
                        action = self._handle_api_error(e, attempt, max_retries)

                        if action != ErrorAction.SERVER_ERROR:
                            server_err_count = 0

                        if action == ErrorAction.FATAL:
                            self._save_and_die(f"FATAL: {type(e).__name__} — {str(e)[:200]}")

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
                                print(f"   ⚠️ Schema Error. Retrying same strategy ({schema_retries + 1}/3)...")
                                schema_retries += 1
                                # add more verbose and detailed JSON description
                                continue
                            else:
                                print("   ❌ Model failed JSON schema 3 times. Downgrading strategy...")
                                schema_retries = 0  # reset before downgrade
                                is_capability_error = True

                        elif action == ErrorAction.SKIP:
                            GLOBAL_STATS.log_error()
                            # Raise typed exception instead of returning ""
                            if e.__class__.__name__ == 'ContextWindowExceededError':
                                raise ContextTooLongError(str(e)) from e
                            elif e.__class__.__name__ == 'ContentPolicyViolationError':
                                raise ContentPolicyError(str(e)) from e
                            else:
                                raise LLMParseError("", str(e)) from e

                        elif action == ErrorAction.RATE_LIMIT:
                            await asyncio.sleep(60)
                            continue

                        elif action == ErrorAction.SERVER_ERROR:
                            server_err_count += 1
                            if server_err_count > 3:
                                self._save_and_die(f"FATAL: Infrastructure failure. Aborting after 3 consecutive Server Errors. Last error: {str(e)[:1000]}")
                            wait_time = 5.0 * server_err_count 
                            print(f"   ⏳ Server Error: sleeping {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        elif action == ErrorAction.RETRY:
                            if attempt < max_retries:
                                wait_time = 0.5 * (attempt + 1)
                                print(f"   ⏳ Waiting {wait_time}s before retry...")
                                await asyncio.sleep(wait_time)
                                attempt += 1
                                continue
                            else:
                                break

                # All retries exhausted for this try. Breaking
                print(f"🔴 FAILED after {attempt + 1} attempts. Last error: {str(last_error)[:100]}")
                GLOBAL_STATS.log_error()
                raise LLMParseError("", f"Exhausted {attempt + 1} retries: {str(last_error)[:200]}") from last_error

        finally:
            # Release the discovery lock if we hold it
            if discovering:
                self._strategy_discovered = True  # lock at current level (already walked past incapable strategies)
                if self._discovery_lock.locked():
                    self._discovery_lock.release()
                if not self._discovery_succeeded:
                    sys.exit(1)



    # ── Fatal checkpoint support ──────────────────────────────

    def save_cache(self):
        """Dump the in-memory session cache to SQLite on crash."""
        if not getattr(self, '_cache_enabled', False) or getattr(self, '_new_cache_entries', 0) == 0:
            return
        try:
            import sqlite3
            with sqlite3.connect(self._cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, response TEXT)")
                cursor.executemany("INSERT OR REPLACE INTO cache (hash, response) VALUES (?, ?)", list(self._session_cache.items()))
                conn.commit()
            print(f"\n   💾 Saved {len(self._session_cache)} total responses to {self._cache_file} for rescue.")
        except Exception:
            pass # Swallow errors on shutdown

    def _save_and_die(self, message: str):
        """Print error, save partial results via callback, die."""
        print(f"\n💀 {message}")
        self.save_cache()
        sys.exit(1)

    async def _generate_safe(self, **task_args) -> Union[str, LLMError]:
        """Wrapper that catches LLMErrors and returns them instead of raising."""
        try:
            return await self.generate(**task_args)
        except LLMError as e:
            return e


    async def generate_batch(self, tasks_data: List[Dict], description="Processing", task: str = None) -> List[Union[str, LLMError]]:
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

        try:
            results = await tqdm_asyncio.gather(
                *[_run_safe(args) for args in tasks_data],
                desc="  "+description
            )
        except asyncio.CancelledError:
            self.save_cache()
            raise

        return results


    async def check_connection(self):
        """Pre-flight check: verifies API reachability and authentication."""
        if not self.base_url:
            # LiteLLM mode — no direct endpoint to check, skip pre-flight
            print(f"📡 LiteLLM mode ({self.model}) — skipping pre-flight connection check.")
            self._connection_verified = True
            return

        print(f"📡 Testing connection to {self.base_url}/models...")
        try:
            await self.client.models.list()
            print("   ✅ Connection confirmed. Server reachable")
            self._connection_verified = True

        except AuthenticationError as e:
            print(f"\n❌ FATAL: Authentication Failed.")
            print(f"   Key: {self.api_key[:6]}...")
            print(f"   Error: {e}")
            print("FATAL: Auth Error — check your API key.")
            os._exit(1)

        except PermissionDeniedError as e:
            print(f"\n❌ FATAL: Permission Denied.")
            print(f"   Your key is valid but cannot access this endpoint.")
            print(f"   Error: {e}")
            print("FATAL: Permission Denied — check your API plan/tier.")
            os._exit(1)

        except NotFoundError as e:
            # /v1/models may not exist on all custom endpoints
            print(f"   ⚡ /models endpoint not available — skipping pre-flight check.")
            self._connection_verified = True

        except APIConnectionError as e:
            print(f"\n❌ FATAL: Cannot connect to API endpoint.")
            print(f"   URL: {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Check: Is the URL correct? Is the server running? Firewall/proxy issues?")
            print(f"   Skip modell check with ---------------------------------------------------------------arg")
            print(f"FATAL: Cannot connect to {self.base_url}")
            os._exit(1)

        except APITimeoutError as e:
            print(f"\n❌ FATAL: Connection timed out during pre-flight check.")
            print(f"   URL: {self.base_url}")
            print(f"   Error: {e}")
            print(f"FATAL: Timeout connecting to {self.base_url}")
            os._exit(1)

        except Exception as e:
            print(f"\n❌ FATAL: Unexpected error during connection check.")
            print(f"   Type: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            print(f"FATAL: {type(e).__name__} in check_connection: {e}")
            os._exit(1)
