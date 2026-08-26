"""
Integration tests for LLMClient fatal error handling.

Tests verify that fatal API errors (auth, connection, budget, infrastructure)
correctly propagate as LLMClientError, stop the batch, and trigger cache saves.

All tests mock the AsyncOpenAI SDK and litellm.acompletion — zero real network calls.

Run with:
    pytest tests/integration/test_llmclient_fatal.py -v
"""

import httpx
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from openai import (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    BadRequestError,
)

from contextchecker.llmclient import LLMClient
from contextchecker.exceptions import LLMClientError


# ── Helpers ──────────────────────────────────────────────────────


def _fake_response(content: str = '{"triplets": []}') -> MagicMock:
    """Build a fake OpenAI ChatCompletion response object.

    Mimics the structure that the real SDK returns:
        response.choices[0].message.content  → the LLM output string
        response.usage.model_dump()          → token counts dict
        response._hidden_params              → provider metadata (cache hints)
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    response._hidden_params = {}
    return response


def _fake_httpx_response(status_code: int = 500) -> MagicMock:
    """Build a fake httpx.Response for OpenAI SDK exception constructors.

    OpenAI's APIStatusError subclasses (AuthenticationError, etc.) require
    an httpx.Response object with at least .status_code and .headers.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    return resp


def _fake_httpx_request() -> MagicMock:
    """Build a fake httpx.Request for connection-level exception constructors.

    APIConnectionError and APITimeoutError require an httpx.Request object.
    """
    req = MagicMock()
    req.url = "http://fake/v1/chat/completions"
    return req


def _make_tasks(n: int) -> list[dict]:
    """Build n minimal batch tasks for generate_batch."""
    return [
        {"messages": [{"role": "user", "content": f"item {i}"}]}
        for i in range(n)
    ]


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path):
    """Create an LLMClient with a mocked AsyncOpenAI backend.

    Preflight connection check and strategy discovery are both skipped
    so tests can focus purely on error handling during generate/generate_batch.
    """
    with patch("contextchecker.llmclient.AsyncOpenAI"):
        c = LLMClient(
            api_key="test-key-abc123",
            model="test-model",
            base_url="http://fake/v1",
        )
    # Skip preflight and discovery — tested separately
    c._connection_verified = True
    c._strategy_discovered = True
    c._discovery_succeeded = True
    # Replace the auto-generated mock with a proper AsyncMock
    c.client.chat.completions.create = AsyncMock()
    return c


@pytest.fixture(params=["openai", "litellm"])
def client_and_mock(request, tmp_path):
    """Fixture parameterizing the client across OpenAI and LiteLLM paths."""
    mode = request.param
    if mode == "openai":
        with patch("contextchecker.llmclient.AsyncOpenAI"):
            c = LLMClient(
                api_key="test-key-abc123",
                model="test-model",
                base_url="http://fake/v1",
            )
        c._connection_verified = True
        c._strategy_discovered = True
        c._discovery_succeeded = True
        c.client.chat.completions.create = AsyncMock()
        yield c, c.client.chat.completions.create
    elif mode == "litellm":
        c = LLMClient(
            api_key="test-key-abc123",
            model="google/gemini-2.0-flash",
            base_url=None,
        )
        c._connection_verified = True
        c._strategy_discovered = True
        c._discovery_succeeded = True
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            yield c, mock_acompletion


# ── Group 1a: Fatal API errors stop the batch ────────────────────


@pytest.mark.integration
class TestFatalErrorsStopBatch:
    """Fatal errors (auth, permission, budget) must raise LLMClientError.

    The batch should not silently swallow these — they indicate the entire
    run is doomed, not just one item.
    """

    async def test_auth_error_raises_llmclient_error(self, client_and_mock):
        """AuthenticationError → LLMClientError with 'FATAL' in message."""
        client, mock_call = client_and_mock
        mock_call.side_effect = AuthenticationError(
            message="Incorrect API key provided: test-k***23.",
            response=_fake_httpx_response(401),
            body=None,
        )

        with pytest.raises(LLMClientError, match="FATAL"):
            await client.generate_batch(_make_tasks(3))

    async def test_permission_denied_raises_llmclient_error(self, client_and_mock):
        """PermissionDeniedError → LLMClientError with 'FATAL' in message."""
        client, mock_call = client_and_mock
        mock_call.side_effect = PermissionDeniedError(
            message="You don't have access to this model.",
            response=_fake_httpx_response(403),
            body=None,
        )

        with pytest.raises(LLMClientError, match="FATAL"):
            await client.generate_batch(_make_tasks(3))

    async def test_model_not_found_raises_llmclient_error(self, client_and_mock):
        """NotFoundError (model doesn't exist) → LLMClientError with 'FATAL'."""
        client, mock_call = client_and_mock
        mock_call.side_effect = NotFoundError(
            message="The model 'gpt-99' does not exist.",
            response=_fake_httpx_response(404),
            body=None,
        )

        with pytest.raises(LLMClientError, match="FATAL"):
            await client.generate_batch(_make_tasks(3))

    async def test_405_is_fatal_not_retried(self, client_and_mock):
        """405 → immediate LLMClientError, no server-error backoff.

        The endpoint refuses the method outright, so retrying can never succeed.
        Guards against it falling back into the retryable APIError branch.
        """
        client, mock_call = client_and_mock
        err = APIError(
            message="Method Not Allowed",
            request=httpx.Request("POST", "http://fake/v1/chat/completions"),
            body=None,
        )
        err.status_code = 405
        mock_call.side_effect = err

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(LLMClientError, match="FATAL"):
                await client.generate_batch(_make_tasks(1))

        # No 5s/10s/15s server-error backoff — it must fail on the first error.
        assert 5.0 not in [c.args[0] for c in mock_sleep.call_args_list]

    async def test_405_message_points_at_the_base_url(self, client_and_mock, caplog):
        """The operator needs to be told which URL is wrong, not 'infrastructure'."""
        client, mock_call = client_and_mock
        err = APIError(
            message="Method Not Allowed",
            request=httpx.Request("POST", "http://fake/v1/chat/completions"),
            body=None,
        )
        err.status_code = 405
        mock_call.side_effect = err

        with caplog.at_level("ERROR"):
            with pytest.raises(LLMClientError):
                await client.generate_batch(_make_tasks(1))

        assert "ENDPOINT ERROR (405)" in caplog.text
        assert "Infrastructure failure" not in caplog.text
        if client.base_url:
            assert client.base_url in caplog.text

    async def test_bad_request_is_not_treated_as_an_endpoint_error(self, client_and_mock, caplog):
        """400 must never reach the 405 branch.

        Regression guard for the 405 fix: keying on a 4xx range instead of 405
        alone would turn every unsupported-strategy probe into a fatal endpoint
        error and break discovery.
        """
        client, mock_call = client_and_mock
        mock_call.side_effect = BadRequestError(
            message="unsupported parameter: reasoning_effort",
            response=_fake_httpx_response(400),
            body=None,
        )

        with caplog.at_level("ERROR"):
            await client.generate_batch(_make_tasks(1))

        assert "ENDPOINT ERROR (405)" not in caplog.text

    async def test_unknown_model_bad_request_raises_llmclient_error(self, client_and_mock):
        """BadRequestError for invalid/unknown model name → LLMClientError with 'FATAL'."""
        client, mock_call = client_and_mock
        body = {
            "error": {
                "message": "{'error': '/chat/completions: Invalid model name passed in model=gemini-2.0-flash. Call `/v1/models` to view available models for your key.'}",
                "type": "None",
            }
        }
        mock_call.side_effect = BadRequestError(
            message="Error code: 400 - " + str(body),
            response=_fake_httpx_response(400),
            body=body,
        )

        with pytest.raises(LLMClientError, match="FATAL"):
            await client.generate_batch(_make_tasks(3))

    async def test_litellm_invalid_provider_raises_llmclient_error(self, client_and_mock):
        """BadRequestError for LiteLLM provider missing -> LLMClientError with 'FATAL'."""
        client, mock_call = client_and_mock
        body = {
            "error": "LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=google/gemini-3.1-flash-lite-preview"
        }
        mock_call.side_effect = BadRequestError(
            message=body["error"],
            response=_fake_httpx_response(400),
            body=body,
        )

        with pytest.raises(LLMClientError, match="FATAL"):
            await client.generate_batch(_make_tasks(3))

    async def test_auth_error_mid_batch_still_fatal(self, client_and_mock):
        """Auth error on request 3 of 5 → still raises LLMClientError.

        With async gather, all 5 tasks start concurrently. The auth error
        on any one of them should propagate and kill the batch. We can't
        assert exact call count (async timing), but the raise is guaranteed.
        """
        client, mock_call = client_and_mock
        success = _fake_response()
        mock_call.side_effect = [
            success,  # task 0: succeeds
            success,  # task 1: succeeds
            AuthenticationError(
                message="key revoked mid-run",
                response=_fake_httpx_response(401),
                body=None,
            ),
            success,  # task 3: may or may not fire (async timing)
            success,  # task 4: may or may not fire (async timing)
        ]

        with pytest.raises(LLMClientError):
            await client.generate_batch(_make_tasks(5))

    async def test_server_error_4_consecutive_is_fatal(self, client_and_mock):
        """4 consecutive InternalServerError → LLMClientError('Infrastructure failure').

        The client tolerates up to 3 consecutive server errors with progressive
        backoff (5s, 10s, 15s). The 4th triggers a fatal abort.
        """
        client, mock_call = client_and_mock
        mock_call.side_effect = InternalServerError(
            message="Internal server error",
            response=_fake_httpx_response(500),
            body=None,
        )

        # Patch asyncio.sleep to skip the 5s/10s/15s backoff waits
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(LLMClientError, match="Infrastructure failure"):
                await client.generate_batch(_make_tasks(1))

            # Verify progressive backoff was attempted: 5s, 10s, 15s
            sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
            assert 5.0 in sleep_values
            assert 10.0 in sleep_values
            assert 15.0 in sleep_values


# ── Group 1b: Fatal errors in preflight ──────────────────────────


@pytest.mark.integration
class TestPreflightFatalErrors:
    """Connection check errors should raise LLMClientError before any batch work."""

    async def test_connection_refused(self, client):
        """APIConnectionError during check_connection → LLMClientError."""
        client._connection_verified = False  # force preflight to run
        client.client.models.list = AsyncMock(
            side_effect=APIConnectionError(
                message="Connection refused",
                request=_fake_httpx_request(),
            )
        )

        with pytest.raises(LLMClientError, match="Cannot connect"):
            await client.generate_batch(_make_tasks(3))

        # Verify no actual LLM calls were made — died before the batch
        client.client.chat.completions.create.assert_not_called()

    async def test_timeout_during_preflight(self, client):
        """APITimeoutError during check_connection → LLMClientError."""
        client._connection_verified = False
        client.client.models.list = AsyncMock(
            side_effect=APITimeoutError(request=_fake_httpx_request())
        )

        with pytest.raises(LLMClientError, match="Timeout"):
            await client.generate_batch(_make_tasks(3))

        client.client.chat.completions.create.assert_not_called()

    async def test_auth_error_during_preflight(self, client):
        """AuthenticationError during check_connection → LLMClientError."""
        client._connection_verified = False
        client.client.models.list = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid key",
                response=_fake_httpx_response(401),
                body=None,
            )
        )

        with pytest.raises(LLMClientError, match="Authentication failed"):
            await client.generate_batch(_make_tasks(3))

        client.client.chat.completions.create.assert_not_called()

    async def test_404_preflight_is_not_fatal(self, client):
        """NotFoundError on /models endpoint → graceful skip, batch proceeds.

        Many custom endpoints (vLLM, Ollama) don't serve /v1/models.
        The client should skip the preflight check and continue.
        """
        client._connection_verified = False
        client.client.models.list = AsyncMock(
            side_effect=NotFoundError(
                message="Not Found",
                response=_fake_httpx_response(404),
                body=None,
            )
        )
        client.client.chat.completions.create.return_value = _fake_response()

        # Should NOT raise — batch proceeds normally after skipping preflight
        results = await client.generate_batch(_make_tasks(2))
        assert len(results) == 2




# ── Group 1d: Fatal abort is honest — no fake progress, no doomed calls ──


@pytest.mark.integration
class TestFatalAbortReporting:
    """Once one task hits a fatal error, sibling tasks must abort quietly.

    Regression guard for the bug where post-fatal peers were downgraded to a
    per-item SKIP: SKIP returns a value (not a raise), which marked the task
    'completed' and ticked the progress bar — so a total failure rendered as
    '16/17 done'. Peers must instead propagate the fatal so the batch stops
    fast and the bar only ever counts genuinely-completed work.
    """

    async def test_fatal_flag_aborts_before_calling_llm(self, client_and_mock):
        """If the fatal flag is already set, generate() must not fire a request.

        Catches tasks that were queued behind the discovery lock / semaphore
        when a sibling died — they should bail out before wasting an API call.
        """
        client, mock_call = client_and_mock
        client._fatal_error_occurred = True
        mock_call.return_value = _fake_response()  # would succeed if ever called

        with pytest.raises(LLMClientError):
            await client.generate([{"role": "user", "content": "hi"}])

        mock_call.assert_not_called()

    async def test_inflight_error_after_fatal_propagates_as_fatal(self, client_and_mock):
        """An in-flight error arriving after the fatal flag flips must raise
        LLMClientError — NOT be reclassified into a per-item ParsingError value.

        ParsingError is a sibling of LLMClientError, so `pytest.raises(
        LLMClientError)` fails on the old (buggy) SKIP-to-value path and passes
        only once the post-fatal guard propagates the abort.
        """
        client, mock_call = client_and_mock

        def _flag_then_raise(*args, **kwargs):
            # Simulate a sibling task flipping the fatal flag mid-flight, then
            # this task's own (otherwise per-item) error arriving afterward.
            client._fatal_error_occurred = True
            raise BadRequestError(
                message="some generic bad request",
                response=_fake_httpx_response(400),
                body={"error": "generic"},
            )

        mock_call.side_effect = _flag_then_raise

        with pytest.raises(LLMClientError):
            await client.generate([{"role": "user", "content": "hi"}])

    async def test_fatal_batch_does_not_advance_progress_bar(self, client_and_mock):
        """A wholly-fatal batch must not tick the progress bar for any task.

        Every task fails with the same fatal auth error. The batch must raise
        and the bar's update() must never be called — total failure should
        not look like partial success.
        """
        client, mock_call = client_and_mock
        mock_call.side_effect = AuthenticationError(
            message="bad key",
            response=_fake_httpx_response(401),
            body=None,
        )

        with patch("contextchecker.llmclient.tqdm") as mock_tqdm:
            pbar = MagicMock()
            mock_tqdm.return_value = pbar

            with pytest.raises(LLMClientError):
                await client.generate_batch(_make_tasks(5))

            pbar.update.assert_not_called()
