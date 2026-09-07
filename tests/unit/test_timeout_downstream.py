"""A timed-out item must reach the report as `timeout`, never as `parse_failure`.

The retry rounds relabel whatever survives them, so a timeout that entered
those rounds would arrive downstream wearing the wrong cause. It is classified
as permanent instead — same bucket as context_too_long.
"""

from unittest.mock import AsyncMock, patch

import pytest

from claimlens.exceptions import LLMTimeoutError, ContextTooLongError
from claimlens.models import ExtractionPayload, AtomizationPayload, CheckingPayload
from claimlens.stats import PhaseStats
from claimlens.workers.extractor import Extractor
from claimlens.workers.atomizer import Atomizer
from claimlens.workers.checker import Checker


def _extractor() -> Extractor:
    with patch("claimlens.workers.extractor.LLMClient"):
        return Extractor(api_key="k", model="m")


def _checker(**kw) -> Checker:
    with patch("claimlens.workers.checker.LLMClient"):
        return Checker(api_key="k", model="m", **kw)


def _atomizer() -> Atomizer:
    with patch("claimlens.workers.atomizer.LLMClient"):
        return Atomizer(api_key="k", model="m")


class TestExtractor:

    def test_classified_as_permanent_with_its_own_cause(self):
        stats = PhaseStats()
        results, retry = _extractor()._classify([LLMTimeoutError("zu langsam")], stats)
        assert stats.timeout == 1
        assert stats.error_causes == {0: "timeout"}
        assert results[0] == []

    def test_never_enters_the_retry_rounds(self):
        """Entering them would relabel it 'parse_failure' at the end."""
        stats = PhaseStats()
        _, retry = _extractor()._classify([LLMTimeoutError("zu langsam")], stats)
        assert retry == []
        assert stats.parse_error == 0

    @pytest.mark.asyncio
    async def test_cause_survives_a_full_batch_with_retry_rounds(self):
        ex = _extractor()
        ex.client.generate_batch = AsyncMock(return_value=[LLMTimeoutError("zu langsam")])
        ex.client.last_batch_requests = 1
        results = await ex.extract_batch([ExtractionPayload(text="T")])
        assert results == [[]]
        assert ex.last_stats.error_causes == {0: "timeout"}


class TestChecker:

    @pytest.mark.asyncio
    async def test_single_mode_marks_the_verdict(self):
        ck = _checker()
        ck.client.generate_batch = AsyncMock(return_value=[LLMTimeoutError("zu langsam")])
        ck.client.last_batch_requests = 1
        out = await ck.check_batch([CheckingPayload(claim="c", reference=["r"],
                                                    item_index=0, claim_index=0)])
        assert out[0].verdict is None
        assert out[0].error == "timeout"
        assert ck.last_stats.timeout == 1

    @pytest.mark.asyncio
    async def test_joint_mode_marks_every_claim_in_the_chunk(self):
        ck = _checker()
        ck.client.generate_batch = AsyncMock(return_value=[LLMTimeoutError("zu langsam")])
        ck.client.last_batch_requests = 1
        out = await ck.check_joint_batch([([(1, "a"), (2, "b"), (3, "c")], ["ref"])])
        assert set(out[0]) == {1, 2, 3}
        assert all(v.verdict is None and v.error == "timeout" for v in out[0].values())
        assert ck.last_stats.timeout == 1


class TestAtomizer:

    @pytest.mark.asyncio
    async def test_counted_and_not_retried(self):
        a = _atomizer()
        a.client.generate_batch = AsyncMock(return_value=[LLMTimeoutError("zu langsam")])
        a.client.last_batch_requests = 1
        p = AtomizationPayload(subject="s", predicate="p", object="o",
                               response="r", item_index=0, triplet_index=0)
        out = await a.atomize_batch([p])
        assert len(out) == 1
        assert a.last_stats.timeout == 1
        assert a.last_stats.parse_error == 0


class TestStatsDisplay:

    def test_counted_as_permanent_not_retryable(self):
        s = PhaseStats(first_pass_count=3, timeout=2, context_too_long=1)
        assert s.total_permanent == 3

    def test_shown_in_the_api_block(self, caplog):
        import logging
        from claimlens.stats import log_api_parsing
        s = PhaseStats(first_pass_count=2, timeout=2)
        with caplog.at_level(logging.INFO, logger="claimlens"):
            log_api_parsing(pending=2, stats=s)
        assert "2 timed out" in caplog.text
