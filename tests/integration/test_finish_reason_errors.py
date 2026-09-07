"""finish_reason failures belong in the existing per-item buckets.

The SDK reports two conditions through `finish_reason` rather than an error
code: the answer hit the output cap, or the safety filter stopped it. Both are
deterministic — the same request produces the same result — so they are skipped
per item instead of retried, and they carry the cause that already exists for
their kind.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import LengthFinishReasonError, ContentFilterFinishReasonError

from claimlens.llmclient import LLMClient, ErrorAction
from claimlens.exceptions import (
    ContextTooLongError, ContentPolicyError, LLMParseError, LLMClientError,
    FinishReasonLengthError,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

BASE_URL = "http://fake/v1"


def _response(content: str, finish_reason: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = finish_reason
    r.usage.model_dump.return_value = {"prompt_tokens": 40, "completion_tokens": 10}
    r._hidden_params = {}
    return r


class _Schema(MagicMock):
    pass


def _client(discovered: bool = True) -> LLMClient:
    with patch("claimlens.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model="m", base_url=BASE_URL)
    c._connection_verified = True
    if discovered:
        c._strategy_discovered = True
        c._discovery_succeeded = True
    c._strategy_index = 1  # 'Schema Only' — a rung that validates
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


class TestClassification:

    async def test_output_cap_is_a_skip_not_a_retry(self):
        c = _client()
        action = c._handle_api_error(LengthFinishReasonError(completion=MagicMock()))
        assert action is ErrorAction.SKIP

    async def test_safety_filter_is_a_skip_not_a_retry(self):
        c = _client()
        assert c._handle_api_error(ContentFilterFinishReasonError()) is ErrorAction.SKIP


class TestEndToEnd:
    """From a real cut-off response to the typed error the worker receives."""

    async def test_cut_off_answer_becomes_finish_reason_length(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": tr', "length")
        with pytest.raises(FinishReasonLengthError):
            await c.generate([{"role": "user", "content": "hi"}], schema=Out)

    async def test_filtered_answer_becomes_content_policy(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client()
        c.client.chat.completions.create.return_value = _response("", "content_filter")
        with pytest.raises(ContentPolicyError):
            await c.generate([{"role": "user", "content": "hi"}], schema=Out)

    async def test_sent_once_not_three_times(self):
        """Deterministic: retrying a cut-off answer buys the same cut-off answer."""
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": tr', "length")
        with pytest.raises(FinishReasonLengthError):
            await c.generate([{"role": "user", "content": "hi"}], schema=Out)
        assert c.client.chat.completions.create.call_count == 1


class TestDiscoveryIsNotBlamed:
    """Both are LLMClientError subclasses, so the pioneer request keeps its cause.

    A plain ParsingError would be rewritten to "No compatible strategy found"
    by the discovery teardown — the strategy was never the problem.
    """

    async def test_pioneer_cut_off_keeps_its_own_cause(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client(discovered=False)
        c.client.chat.completions.create.return_value = _response('{"ok": tr', "length")
        with pytest.raises(LLMClientError) as exc:
            await c.generate([{"role": "user", "content": "hi"}], schema=Out)
        assert isinstance(exc.value, FinishReasonLengthError)
        assert "No compatible strategy" not in str(exc.value)


class TestReachesTheTree:

    async def test_worker_records_it_as_finish_reason_length(self):
        from claimlens.stats import PhaseStats
        from claimlens.workers.extractor import Extractor

        with patch("claimlens.workers.extractor.LLMClient"):
            ex = Extractor(api_key="k", model="m")
        stats = PhaseStats()
        ex._classify([FinishReasonLengthError("cut off")], stats)
        assert stats.finish_reason_length == 1
        assert stats.error_causes == {0: "finish_reason_length"}


class TestBlastRadius:
    """The two guards that silently turn a per-item skip back into a batch kill."""

    async def test_generate_safe_returns_it_as_a_value(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": tr', "length")
        res = await c._generate_safe(messages=[{"role": "user", "content": "hi"}], schema=Out)
        assert isinstance(res, FinishReasonLengthError)

    async def test_one_cut_off_item_does_not_kill_the_batch(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            ok: bool

        c = _client()

        async def side_effect(**kwargs):
            cut = kwargs["messages"][0]["content"] == "item 3"
            return _response('{"ok": tr' if cut else '{"ok": true}',
                             "length" if cut else "stop")

        c.client.chat.completions.create = AsyncMock(side_effect=side_effect)
        results = await c.generate_batch(
            [{"messages": [{"role": "user", "content": f"item {i}"}], "schema": Out}
             for i in range(6)]
        )
        assert len(results) == 6
        assert sum(isinstance(r, FinishReasonLengthError) for r in results) == 1
        assert c._fatal_error_occurred is False


class TestSeparateFromContextTooLong:
    """Two different failures: the prompt did not fit vs the answer did not."""

    async def test_own_counter_and_own_tree_row(self, caplog):
        import logging
        from claimlens.stats import PhaseStats, log_api_parsing

        s = PhaseStats(first_pass_count=5, first_pass_ok=1,
                       context_too_long=2, finish_reason_length=2)
        with caplog.at_level(logging.INFO, logger="claimlens"):
            log_api_parsing(pending=5, stats=s)
        assert "2 context too long" in caplog.text
        assert "2 finish reason length" in caplog.text
        assert s.total_permanent == 4

    async def test_checker_marks_the_verdict_with_its_own_cause(self):
        from claimlens.models import CheckingPayload
        from claimlens.workers.checker import Checker

        with patch("claimlens.workers.checker.LLMClient"):
            ck = Checker(api_key="k", model="m")
        ck.client.generate_batch = AsyncMock(
            return_value=[FinishReasonLengthError("cut off")])
        ck.client.last_batch_requests = 1

        out = await ck.check_batch([CheckingPayload(claim="c", reference=["r"],
                                                    item_index=0, claim_index=0)])
        assert out[0].verdict is None
        assert out[0].error == "finish_reason_length"
        assert ck.last_stats.finish_reason_length == 1

    async def test_atomizer_counts_it_apart(self):
        from claimlens.models import AtomizationPayload
        from claimlens.workers.atomizer import Atomizer

        with patch("claimlens.workers.atomizer.LLMClient"):
            a = Atomizer(api_key="k", model="m")
        a.client.generate_batch = AsyncMock(
            return_value=[FinishReasonLengthError("cut off")])
        a.client.last_batch_requests = 1
        p = AtomizationPayload(subject="s", predicate="p", object="o",
                               response="r", item_index=0, triplet_index=0)
        await a.atomize_batch([p])
        assert a.last_stats.finish_reason_length == 1
        assert a.last_stats.context_too_long == 0


class TestMaxTokensSetting:
    """LLM_MAX_TOKENS is the lever that makes finish_reason_length reachable."""

    async def test_unset_means_the_parameter_is_not_sent(self):
        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": true}', "stop")
        c.max_tokens = None
        await c.generate([{"role": "user", "content": "hi"}])
        assert "max_tokens" not in c.client.chat.completions.create.call_args.kwargs

    async def test_set_value_reaches_the_request(self):
        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": true}', "stop")
        c.max_tokens = 500
        await c.generate([{"role": "user", "content": "hi"}])
        assert c.client.chat.completions.create.call_args.kwargs["max_tokens"] == 500

    async def test_caller_keeps_the_last_word(self):
        """A worker passing its own budget is not overridden by the global one."""
        c = _client()
        c.client.chat.completions.create.return_value = _response('{"ok": true}', "stop")
        c.max_tokens = 500
        await c.generate([{"role": "user", "content": "hi"}], max_tokens=7)
        assert c.client.chat.completions.create.call_args.kwargs["max_tokens"] == 7
