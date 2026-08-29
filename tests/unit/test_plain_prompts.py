"""Unit tests for the plain-prompt path — the Unguided Decoding rung.

Run with:
    pytest tests/unit/test_plain_prompts.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError

from contextchecker import settings
from contextchecker.exceptions import LLMClientError
from contextchecker.llmclient import LLMClient
from contextchecker.models import (
    AtomizationPayload,
    CheckingPayload,
    ExtractionPayload,
)
from contextchecker.workers.atomizer import Atomizer
from contextchecker.workers.checker import Checker
from contextchecker.workers.extractor import Extractor

BASE_URL = "http://fake/v1"


@pytest.fixture(autouse=True)
def _clear_process_caches():
    LLMClient._STRATEGY_CACHE.clear()
    LLMClient._VERIFIED_ENDPOINTS.clear()
    LLMClient._DROP_PARAMS_CACHE.clear()
    yield


def _response(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    r.usage.model_dump.return_value = {"prompt_tokens": 1, "completion_tokens": 1}
    r._hidden_params = {}
    return r


def _bad_request() -> BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return BadRequestError("Unsupported parameter", response=response, body=None)


def wire(worker, content: str) -> list[dict]:
    """Reject every format parameter, forcing the walk to Unguided Decoding."""
    sent: list[dict] = []

    async def create(**kwargs):
        sent.append(kwargs)
        if "response_format" in kwargs or "reasoning_effort" in kwargs:
            raise _bad_request()
        return _response(content)

    worker.client.client = MagicMock()
    worker.client.client.chat.completions.create = AsyncMock(side_effect=create)
    worker.client.client.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return sent


class TestExtractor:
    @pytest.mark.asyncio
    async def test_plain_prompt_goes_on_the_wire(self):
        w = Extractor(api_key="k", model="m", base_url=BASE_URL)
        sent = wire(w, '{"triplets": []}')

        await w.extract_batch([ExtractionPayload(text="Goethe wrote Faust.")])

        assert w.client.strategy.name == "Unguided Decoding"
        assert sent[-1]["messages"] == w._build_messages("Goethe wrote Faust.", "plain")

    @pytest.mark.asyncio
    async def test_the_shape_is_substituted_into_the_prompt(self):
        w = Extractor(api_key="k", model="m", base_url=BASE_URL)
        prompt = w._build_messages("x", "plain")[-1]["content"]

        assert "{{schema}}" not in prompt
        assert '"subject": "<string>"' in prompt

    @pytest.mark.asyncio
    async def test_a_missing_plain_prompt_aborts(self, monkeypatch):
        prompts = dict(settings.PROMPTS)
        del prompts["extractor_prompt_plain"]
        monkeypatch.setattr(settings, "PROMPTS", prompts)

        w = Extractor(api_key="k", model="m", base_url=BASE_URL)
        wire(w, '{"triplets": []}')

        with pytest.raises(LLMClientError):
            await w.extract_batch([ExtractionPayload(text="Goethe wrote Faust.")])


class TestChecker:
    @pytest.mark.asyncio
    async def test_plain_prompt_goes_on_the_wire(self):
        w = Checker(api_key="k", model="m", base_url=BASE_URL)
        sent = wire(w, '{"explanation": "ok", "verdict": "Entailment"}')
        payload = CheckingPayload(claim="c", reference=["r"], item_index=0, claim_index=0)

        await w.check_batch([payload])

        assert w.client.strategy.name == "Unguided Decoding"
        assert sent[-1]["messages"] == w._build_messages("c", ["r"], "plain")

    @pytest.mark.asyncio
    async def test_the_shape_is_substituted_into_the_prompt(self):
        w = Checker(api_key="k", model="m", base_url=BASE_URL)
        prompt = w._build_messages("c", ["r"], "plain")[-1]["content"]

        assert "{{schema}}" not in prompt
        assert "Entailment | Contradiction | Neutral" in prompt

    @pytest.mark.asyncio
    async def test_a_missing_plain_prompt_aborts(self, monkeypatch):
        prompts = dict(settings.PROMPTS)
        del prompts["checker_prompt_plain"]
        monkeypatch.setattr(settings, "PROMPTS", prompts)

        w = Checker(api_key="k", model="m", base_url=BASE_URL)
        wire(w, '{"explanation": "ok", "verdict": "Entailment"}')
        payload = CheckingPayload(claim="c", reference=["r"], item_index=0, claim_index=0)

        with pytest.raises(LLMClientError):
            await w.check_batch([payload])


class TestAtomizer:
    @staticmethod
    def _payload() -> AtomizationPayload:
        return AtomizationPayload(
            subject="basalt", predicate="forms from", object="lava and magma",
            response="Basalt forms from lava and magma.",
            item_index=0, triplet_index=0,
        )

    @pytest.mark.asyncio
    async def test_plain_prompt_goes_on_the_wire(self):
        w = Atomizer(api_key="k", model="m", base_url=BASE_URL)
        sent = wire(w, '{"reasoning": "r", "is_atomic": true, "split": []}')

        await w.atomize_batch([self._payload()])

        assert w.client.strategy.name == "Unguided Decoding"
        assert sent[-1]["messages"] == w._build_messages(self._payload(), "plain")

    @pytest.mark.asyncio
    async def test_the_shape_is_substituted_into_the_prompt(self):
        w = Atomizer(api_key="k", model="m", base_url=BASE_URL)
        prompt = w._build_messages(self._payload(), "plain")[-1]["content"]

        assert "{{schema}}" not in prompt
        assert '"is_atomic": "<boolean>"' in prompt

    @pytest.mark.asyncio
    async def test_a_missing_plain_prompt_aborts(self, monkeypatch):
        prompts = dict(settings.PROMPTS)
        del prompts["atomizer_prompt_plain"]
        monkeypatch.setattr(settings, "PROMPTS", prompts)

        w = Atomizer(api_key="k", model="m", base_url=BASE_URL)
        wire(w, '{"reasoning": "r", "is_atomic": true, "split": []}')

        with pytest.raises(LLMClientError):
            await w.atomize_batch([self._payload()])
