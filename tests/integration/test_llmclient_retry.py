"""
Integration tests for LLMClient rate-limit handling and the drop_params A/B probe.

Rate limits (429) must never drop an item: the request is retried indefinitely,
honoring a server Retry-After when present (or a jittered fallback otherwise),
and aborting only if the server asks to wait longer than the configured cap.

drop_params is a LiteLLM-proxy-only field. Against a direct endpoint it 400s, so
the client learns the endpoint type via an A/B probe: on the first drop_params
rejection it retries the SAME strategy without drop_params (keeping reasoning),
and only advances the matrix if that also fails.

All tests mock the AsyncOpenAI SDK — zero real network, and asyncio.sleep is
patched so back-offs don't actually wait.

Run with:
    pytest tests/integration/test_llmclient_retry.py -v
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import RateLimitError, BadRequestError

from contextchecker.llmclient import LLMClient, RETRY_MATRIX
from contextchecker.exceptions import LLMClientError


BASE_URL = "http://fake/v1"
MODEL = "test-model"


# ── Helpers ──────────────────────────────────────────────────────


def _fake_response(content: str = '{"ok": true}') -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.model_dump.return_value = {"prompt_tokens": 1, "completion_tokens": 1}
    response._hidden_params = {}
    return response


def _httpx_response(status_code: int = 429, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers if headers is not None else {}
    return resp


def _rate_limit_error(retry_after=None) -> RateLimitError:
    headers = {} if retry_after is None else {"retry-after": str(retry_after)}
    return RateLimitError(
        message="rate limited",
        response=_httpx_response(429, headers),
        body=None,
    )


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(
        message=message,
        response=_httpx_response(400),
        body={"error": {"message": message}},
    )


def _make_client(tmp_path, *, discovered: bool) -> LLMClient:
    cache_file = str(tmp_path / "cache.db")
    with patch("contextchecker.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model=MODEL, base_url=BASE_URL, cache_file=cache_file)
    c._connection_verified = True
    if discovered:
        c._strategy_discovered = True
        c._discovery_succeeded = True
        c._strategy_index = 0  # 'Reasoning + Schema' — carries reasoning_effort
    c.client.chat.completions.parse = AsyncMock()
    return c


@pytest.fixture(autouse=True)
def _reset_process_state():
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    yield
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()


# ── Rate limits ──────────────────────────────────────────────────


@pytest.mark.integration
class TestRateLimit:

    async def test_honors_server_retry_after(self, tmp_path):
        """A Retry-After header is used verbatim as the back-off."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = [
            _rate_limit_error(retry_after=5),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c.client.chat.completions.parse.call_count == 2
        assert mock_sleep.call_args.args[0] == 5.0
        assert c._rate_limit_wait == 5.0

    async def test_fallback_wait_is_jittered(self, tmp_path):
        """No Retry-After → jittered fallback within ±10% of RATE_LIMIT_WAIT (60)."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = [
            _rate_limit_error(retry_after=None),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await c.generate([{"role": "user", "content": "hi"}])

        waited = mock_sleep.call_args.args[0]
        assert 54.0 <= waited <= 66.0  # 60 ± 10%

    async def test_retry_after_over_cap_aborts(self, tmp_path):
        """Retry-After beyond RATE_LIMIT_MAX_WAIT (300) is fatal, not an infinite wait."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = _rate_limit_error(retry_after=9999)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMClientError):
                await c.generate([{"role": "user", "content": "hi"}])

        # Aborted on the first 429 — no retry.
        assert c.client.chat.completions.parse.call_count == 1

    async def test_rate_limit_never_drops_item(self, tmp_path):
        """429s don't count toward max_retries — the item is retried until it clears."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = [
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            out = await c.generate([{"role": "user", "content": "hi"}], max_retries=2)

        assert out == '{"ok": true}'
        assert c.client.chat.completions.parse.call_count == 6  # 5 × 429 + 1 success

    async def test_console_message_once_per_window_no_attempt_label(self, tmp_path, caplog):
        """One clear message per episode; the misleading 'Attempt X/N' label is gone."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = [
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _rate_limit_error(retry_after=1),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.WARNING, logger="contextchecker"):
                await c.generate([{"role": "user", "content": "hi"}])

        msgs = [r.getMessage() for r in caplog.records]
        rate_msgs = [m for m in msgs if "Rate limited by" in m]
        assert len(rate_msgs) == 1                       # one per episode, not per 429
        assert not any("Attempt" in m for m in msgs)     # bogus label removed


# ── drop_params A/B probe ────────────────────────────────────────


@pytest.mark.integration
class TestDropParamsProbe:

    async def test_proxy_keeps_drop_params(self, tmp_path):
        """drop_params accepted → learned as a proxy; the field is sent and kept."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.return_value = _fake_response()

        await c.generate([{"role": "user", "content": "hi"}])

        assert c._drop_params_supported is True
        kwargs = c.client.chat.completions.parse.call_args.kwargs
        assert kwargs["extra_body"]["drop_params"] is True
        assert kwargs["reasoning_effort"] == "low"

    async def test_direct_endpoint_drops_param_but_keeps_reasoning(self, tmp_path):
        """drop_params rejected → retried WITHOUT it on the same strategy, reasoning intact."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.parse.side_effect = [
            _bad_request("Validation: Unsupported parameter(s): `drop_params`"),
            _fake_response(),
        ]

        out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c._drop_params_supported is False
        assert c.client.chat.completions.parse.call_count == 2
        assert c._strategy_index == 0  # did NOT advance/lose reasoning

        first, second = c.client.chat.completions.parse.call_args_list
        assert first.kwargs["extra_body"]["drop_params"] is True
        assert "drop_params" not in second.kwargs.get("extra_body", {})
        assert second.kwargs["reasoning_effort"] == "low"  # reasoning retained

    async def test_fails_at_both_advances_matrix(self, tmp_path):
        """drop_params off AND still 400 → real capability gap → advance the matrix."""
        c = _make_client(tmp_path, discovered=False)  # let discovery walk
        c.client.chat.completions.parse.side_effect = [
            _bad_request("Unsupported parameter(s): `drop_params`"),  # strat 0 + drop_params
            _bad_request("reasoning_effort not supported"),            # strat 0 no drop_params
            _fake_response(),                                          # strat 1 (Schema Only)
        ]

        out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c._drop_params_supported is False
        assert c.client.chat.completions.parse.call_count == 3
        assert c._strategy_index == 1               # advanced past reasoning
        assert RETRY_MATRIX[1].reasoning_effort is None
