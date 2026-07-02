"""
Unit tests for LLMClient process-level strategy-cache adoption.

A strategy discovered for one (base_url, model) must be reusable by any other
client for the same endpoint+model — including a client that was *constructed
before* discovery happened (the eager-construction case: an atomizer built at
boot whose sibling extractor discovers first, regression for bug #3).

No real network: AsyncOpenAI is mocked and the SDK parse call is an AsyncMock.

Run with:
    pytest tests/unit/test_llmclient_strategy_cache.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextchecker.llmclient import LLMClient, RETRY_MATRIX


BASE_URL = "http://fake/v1"
MODEL = "test-model"


def _fake_response(content: str = '{"ok": true}') -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.model_dump.return_value = {"prompt_tokens": 1, "completion_tokens": 1}
    response._hidden_params = {}
    return response


def _make_client(tmp_path, base_url=BASE_URL, model=MODEL) -> LLMClient:
    """Build an LLMClient with a mocked SDK backend, preflight skipped."""
    cache_file = str(tmp_path / f"{model}.db")
    with patch("contextchecker.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model=model, base_url=base_url, cache_file=cache_file)
    c._connection_verified = True  # skip preflight
    c.client.chat.completions.parse = AsyncMock(return_value=_fake_response())
    return c


@pytest.fixture(autouse=True)
def _reset_process_state():
    """Isolate the class-level caches between tests (shared mutable state)."""
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()
    yield
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()


class TestStrategyCacheAdoption:
    """Helper-level and end-to-end coverage of cache adoption."""

    def test_adopt_helper_applies_cached_index(self, tmp_path):
        """The helper adopts a cached index and marks the client discovered."""
        c = _make_client(tmp_path)
        assert not c._strategy_discovered
        LLMClient._STRATEGY_CACHE[(BASE_URL, MODEL)] = 2

        adopted = c._try_adopt_cached_strategy()

        assert adopted is True
        assert c._strategy_index == 2
        assert c._strategy_discovered is True
        assert c._discovery_succeeded is True
        assert c.strategy is RETRY_MATRIX[2]

    def test_adopt_helper_noop_when_cache_empty(self, tmp_path):
        c = _make_client(tmp_path)
        assert c._try_adopt_cached_strategy() is False
        assert c._strategy_discovered is False

    def test_adopt_helper_noop_when_already_discovered(self, tmp_path):
        """An already-discovered client must not be overwritten by the cache."""
        c = _make_client(tmp_path)
        c._strategy_discovered = True
        LLMClient._STRATEGY_CACHE[(BASE_URL, MODEL)] = 3

        assert c._try_adopt_cached_strategy() is False
        assert c._strategy_index == 0  # untouched

    async def test_constructed_before_discovery_adopts_at_request_time(self, tmp_path):
        """Core #3 case: a client built while the cache was empty still adopts a
        strategy a sibling discovers later — at request time, without re-discovering.
        """
        # Client built first, cache empty → not discovered.
        client = _make_client(tmp_path)
        assert not client._strategy_discovered

        # Sibling discovers a NON-default strategy and populates the process cache.
        non_default = 2
        assert non_default != 0
        LLMClient._STRATEGY_CACHE[(BASE_URL, MODEL)] = non_default

        await client.generate([{"role": "user", "content": "hi"}])

        # Adopted the cached index — did NOT re-discover from the top (index 0).
        assert client._strategy_discovered is True
        assert client._strategy_index == non_default
        client.client.chat.completions.parse.assert_called_once()

    def test_constructed_after_discovery_adopts_in_init(self, tmp_path):
        """A warm cache at construction time is adopted in __init__."""
        LLMClient._STRATEGY_CACHE[(BASE_URL, MODEL)] = 1
        c = _make_client(tmp_path)
        assert c._strategy_discovered is True
        assert c._strategy_index == 1

    def test_cache_is_keyed_by_base_url_and_model(self, tmp_path):
        """A strategy for one model must not leak to a different model."""
        LLMClient._STRATEGY_CACHE[(BASE_URL, "model-a")] = 2
        c = _make_client(tmp_path, model="model-b")
        assert c._strategy_discovered is False
        assert c._try_adopt_cached_strategy() is False
