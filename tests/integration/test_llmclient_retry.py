"""
Integration tests for LLMClient rate-limit handling and the drop_params A/B probe.

Rate limits (429) must never drop an item: the request is retried indefinitely,
honoring a server Retry-After when present (or a jittered fallback otherwise),
and aborting only if the server asks to wait longer than the configured cap.

A parse failure is not a transient server condition: the response arrived and was
billed, it just wasn't valid JSON. Those retry immediately, with no back-off —
while genuinely transient errors keep theirs.

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

from openai import RateLimitError, BadRequestError, ConflictError
from pydantic import BaseModel, ValidationError

from contextchecker.llmclient import LLMClient, RETRY_MATRIX
from contextchecker.exceptions import LLMClientError, LLMParseError


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


class _Probe(BaseModel):
    ok: bool


def _validation_error() -> ValidationError:
    """A real pydantic ValidationError, as .parse() raises on unparseable output."""
    try:
        _Probe.model_validate_json("Entailment: [1]")
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


def _conflict_error() -> ConflictError:
    return ConflictError(message="conflict", response=_httpx_response(409), body=None)


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(
        message=message,
        response=_httpx_response(400),
        body={"error": {"message": message}},
    )


def _make_client(tmp_path, *, discovered: bool) -> LLMClient:
    with patch("contextchecker.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model=MODEL, base_url=BASE_URL)
    c._connection_verified = True
    if discovered:
        c._strategy_discovered = True
        c._discovery_succeeded = True
        c._strategy_index = 0  # 'Reasoning + Schema' — carries reasoning_effort
    c.client.chat.completions.create = AsyncMock()
    return c


@pytest.fixture(autouse=True)
def _reset_process_state():
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()
    yield
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()


# ── Rate limits ──────────────────────────────────────────────────


@pytest.mark.integration
class TestRateLimit:

    async def test_honors_server_retry_after(self, tmp_path):
        """A Retry-After header is used verbatim as the back-off."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
            _rate_limit_error(retry_after=5),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c.client.chat.completions.create.call_count == 2
        assert mock_sleep.call_args.args[0] == 5.0
        assert c._rate_limit_wait == 5.0

    async def test_fallback_wait_is_jittered(self, tmp_path):
        """No Retry-After → jittered fallback within ±10% of RATE_LIMIT_WAIT (60)."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
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
        c.client.chat.completions.create.side_effect = _rate_limit_error(retry_after=9999)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMClientError):
                await c.generate([{"role": "user", "content": "hi"}])

        # Aborted on the first 429 — no retry.
        assert c.client.chat.completions.create.call_count == 1

    async def test_rate_limit_never_drops_item(self, tmp_path):
        """429s don't count toward max_retries — the item is retried until it clears."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
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
        assert c.client.chat.completions.create.call_count == 6  # 5 × 429 + 1 success

    async def test_console_message_once_per_window_no_attempt_label(self, tmp_path, caplog):
        """One clear message per episode; the misleading 'Attempt X/N' label is gone."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
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


# ── Parse failures ───────────────────────────────────────────────


@pytest.mark.integration
class TestParseRetryNoBackoff:

    async def test_parse_failure_retries_without_sleeping(self, tmp_path):
        """Unparseable output is resampled immediately — no back-off."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
            _validation_error(),
            _validation_error(),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c.client.chat.completions.create.call_count == 3
        mock_sleep.assert_not_called()

    async def test_transient_error_still_backs_off(self, tmp_path):
        """Regression guard for the RETRY/RESAMPLE split: a real transient
        error (409) keeps the back-off that parse failures lost."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
            _conflict_error(),
            _fake_response(),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert mock_sleep.call_args.args[0] == 0.5

    async def test_attempt_accounting_is_unchanged(self, tmp_path):
        """Dropping the sleep must not change how many attempts an item gets."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [_validation_error()] * 3

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMParseError, match="Exhausted 3 retries"):
                await c.generate([{"role": "user", "content": "hi"}])

        assert c.client.chat.completions.create.call_count == 3

    async def test_exhausted_parse_failure_is_silent(self, tmp_path, caplog):
        """The expected case is counted on the bar and in stats — not printed."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [_validation_error()] * 3

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.ERROR, logger="contextchecker"):
                with pytest.raises(LLMParseError):
                    await c.generate([{"role": "user", "content": "hi"}])

        assert caplog.records == []

    async def test_mixed_sequence_ending_in_parse_failure_reports(self, tmp_path, caplog):
        """Gating on the last error alone would silence this — a transient error
        occurred, so the item is not the expected all-parse-failure case."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
            _conflict_error(),
            _validation_error(),
            _validation_error(),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.ERROR, logger="contextchecker"):
                with pytest.raises(LLMParseError):
                    await c.generate([{"role": "user", "content": "hi"}])

        assert any("FAILED after 3 attempts" in r.getMessage() for r in caplog.records)

    async def test_exhausted_transient_error_still_reports(self, tmp_path, caplog):
        """A non-parse exhaustion is unusual — it keeps its line."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [_conflict_error()] * 3

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.ERROR, logger="contextchecker"):
                with pytest.raises(LLMParseError):
                    await c.generate([{"role": "user", "content": "hi"}])

        assert any("FAILED after 3 attempts" in r.getMessage() for r in caplog.records)


# ── drop_params A/B probe ────────────────────────────────────────


@pytest.mark.integration
class TestDropParamsProbe:

    async def test_proxy_keeps_drop_params(self, tmp_path):
        """drop_params accepted → learned as a proxy; the field is sent and kept."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.return_value = _fake_response()

        await c.generate([{"role": "user", "content": "hi"}])

        assert c._drop_params_supported is True
        kwargs = c.client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["drop_params"] is True
        assert kwargs["reasoning_effort"] == "low"

    async def test_direct_endpoint_drops_param_but_keeps_reasoning(self, tmp_path):
        """drop_params rejected → retried WITHOUT it on the same strategy, reasoning intact."""
        c = _make_client(tmp_path, discovered=True)
        c.client.chat.completions.create.side_effect = [
            _bad_request("Validation: Unsupported parameter(s): `drop_params`"),
            _fake_response(),
        ]

        out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c._drop_params_supported is False
        assert c.client.chat.completions.create.call_count == 2
        assert c._strategy_index == 0  # did NOT advance/lose reasoning

        first, second = c.client.chat.completions.create.call_args_list
        assert first.kwargs["extra_body"]["drop_params"] is True
        assert "drop_params" not in second.kwargs.get("extra_body", {})
        assert second.kwargs["reasoning_effort"] == "low"  # reasoning retained

    async def test_learning_is_shared_across_clients(self, tmp_path):
        """A second client for the same endpoint inherits the learned drop_params
        state — no per-client re-learn / concurrent stampede on a cache hit."""
        # Client 1 learns it's a direct endpoint (drop_params rejected).
        c1 = _make_client(tmp_path, discovered=True)
        c1.client.chat.completions.create.side_effect = [
            _bad_request("Unsupported parameter(s): `drop_params`"),
            _fake_response(),
        ]
        await c1.generate([{"role": "user", "content": "hi"}])
        assert c1._drop_params_supported is False
        assert LLMClient._DROP_PARAMS_CACHE[(BASE_URL, MODEL)] is False

        # Client 2 is fresh but seeds from the process cache in __init__ — so it
        # sends WITHOUT drop_params on the very first request (no rejected wave).
        c2 = _make_client(tmp_path, discovered=True)
        assert c2._drop_params_supported is False
        c2.client.chat.completions.create.return_value = _fake_response()
        await c2.generate([{"role": "user", "content": "hi"}])

        assert c2.client.chat.completions.create.call_count == 1
        kwargs = c2.client.chat.completions.create.call_args.kwargs
        assert "drop_params" not in kwargs.get("extra_body", {})

    async def test_fails_at_both_advances_matrix(self, tmp_path):
        """drop_params off AND still 400 → real capability gap → advance the matrix."""
        c = _make_client(tmp_path, discovered=False)  # let discovery walk
        c.client.chat.completions.create.side_effect = [
            _bad_request("Unsupported parameter(s): `drop_params`"),  # strat 0 + drop_params
            _bad_request("reasoning_effort not supported"),            # strat 0 no drop_params
            _fake_response(),                                          # strat 1 (Schema Only)
        ]

        out = await c.generate([{"role": "user", "content": "hi"}])

        assert out == '{"ok": true}'
        assert c._drop_params_supported is False
        assert c.client.chat.completions.create.call_count == 3
        assert c._strategy_index == 1               # advanced past reasoning
        assert RETRY_MATRIX[1].reasoning_effort is None
