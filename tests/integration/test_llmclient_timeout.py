"""Timeouts are a per-item skip, never a batch abort.

A retry round resends the same payload, so an item too slow to answer stays
too slow — one retry, then the item carries a `timeout` cause downstream.
A connection that cannot be opened at all is a different signal and keeps
aborting the run.

Run with:
    pytest tests/integration/test_llmclient_timeout.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APITimeoutError, APIConnectionError

from contextchecker.llmclient import LLMClient, TIMEOUT_RETRIES
from contextchecker.exceptions import (
    LLMClientError, LLMTimeoutError, ContextTooLongError, ContentPolicyError,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

BASE_URL = "http://fake/v1"
MODEL = "test-model"


def _timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", BASE_URL))


def _conn_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", BASE_URL))


def _ok(content: str = '{"ok": true}') -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    r.usage.model_dump.return_value = {"prompt_tokens": 1, "completion_tokens": 1}
    r._hidden_params = {}
    return r


def _client() -> LLMClient:
    with patch("contextchecker.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model=MODEL, base_url=BASE_URL)
    c._connection_verified = True
    c._strategy_discovered = True
    c._discovery_succeeded = True
    c.client.chat.completions.create = AsyncMock()
    return c


def _tasks(n: int) -> list[dict]:
    return [{"messages": [{"role": "user", "content": f"item {i}"}]} for i in range(n)]


@pytest.fixture(autouse=True)
def _reset_process_state():
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()
    yield
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()


class TestRetryBudget:

    async def test_one_retry_then_gives_up(self):
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout()] * 5
        with pytest.raises(LLMTimeoutError):
            await c.generate([{"role": "user", "content": "hi"}])
        assert c.client.chat.completions.create.call_count == TIMEOUT_RETRIES + 1

    async def test_retry_can_succeed(self):
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout(), _ok()]
        out = await c.generate([{"role": "user", "content": "hi"}])
        assert out == '{"ok": true}'
        assert c.client.chat.completions.create.call_count == 2

    async def test_timeout_never_sets_the_fatal_flag(self):
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout()] * 5
        with pytest.raises(LLMTimeoutError):
            await c.generate([{"role": "user", "content": "hi"}])
        assert c._fatal_error_occurred is False


class TestBlastRadius:
    """The two guards that would silently turn a skip back into a batch kill."""

    async def test_generate_safe_returns_it_as_a_value(self):
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout()] * 5
        res = await c._generate_safe(messages=[{"role": "user", "content": "hi"}])
        assert isinstance(res, LLMTimeoutError)

    async def test_one_slow_item_does_not_kill_the_batch(self):
        c = _client()
        calls = {"n": 0}

        async def side_effect(**kwargs):
            calls["n"] += 1
            if kwargs["messages"][0]["content"] == "item 7":
                raise _timeout()
            return _ok()

        c.client.chat.completions.create = AsyncMock(side_effect=side_effect)
        results = await c.generate_batch(_tasks(20))

        assert len(results) == 20
        failed = [r for r in results if isinstance(r, Exception)]
        assert len(failed) == 1
        assert isinstance(failed[0], LLMTimeoutError)
        assert sum(1 for r in results if r == '{"ok": true}') == 19

    async def test_batch_leak_scan_does_not_reraise_it(self):
        """generate_batch re-raises stray LLMClientErrors — but not this one."""
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout()] * 99
        results = await c.generate_batch(_tasks(3))
        assert all(isinstance(r, LLMTimeoutError) for r in results)

    async def test_every_item_timing_out_still_completes(self):
        """No breaker: timeouts alone never abort, however many."""
        c = _client()
        c.client.chat.completions.create.side_effect = [_timeout()] * 99
        results = await c.generate_batch(_tasks(30))
        assert len(results) == 30
        assert all(isinstance(r, LLMTimeoutError) for r in results)


class TestConnectionStillFatal:
    """Being unreachable is the offline signal and must keep aborting."""

    async def test_connection_errors_still_abort_the_run(self):
        c = _client()
        c.client.chat.completions.create.side_effect = [_conn_error()] * 99
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMClientError) as exc:
                await c.generate_batch(_tasks(2))
        assert not isinstance(exc.value, LLMTimeoutError)
        assert c._fatal_error_occurred is True


class TestExceptionHierarchy:

    async def test_sits_next_to_the_other_per_item_errors(self):
        assert issubclass(LLMTimeoutError, LLMClientError)
        assert not issubclass(LLMTimeoutError, (ContextTooLongError, ContentPolicyError))
