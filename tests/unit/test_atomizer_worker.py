"""
Tests for the Atomizer worker — classification, retry, and batch logic.

Mirrors the test patterns in test_extraction_service.py for the Extractor worker.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from contextchecker.workers.atomizer import (
    Atomizer,
    AtomicTriplet,
    AtomizationResult,
    RetryRoundConfig,
)
from contextchecker.models import AtomizationPayload
from contextchecker.exceptions import (
    ContextTooLongError,
    ContentPolicyError,
    ParsingError,
)
from contextchecker.stats import PhaseStats


# ── Helpers ──────────────────────────────────────────────────────────────────

def _atomizer(**kwargs) -> Atomizer:
    """Build an Atomizer with mocked LLMClient."""
    defaults = dict(
        api_key="test-key",
        model="test-model",
    )
    defaults.update(kwargs)
    with patch("contextchecker.workers.atomizer.LLMClient"):
        return Atomizer(**defaults)


def _payload(triplet: str = "cats are animals", item_idx: int = 0, tri_idx: int = 0):
    return AtomizationPayload(triplet=triplet, item_index=item_idx, triplet_index=tri_idx)


def _result_json(triplets: list[dict]) -> str:
    """Build a valid AtomizationResult JSON string."""
    return json.dumps({"triplets": triplets})


def _atomic(s: str, p: str, o: str) -> dict:
    return {"subject": s, "predicate": p, "object": o}


# ── Test _build_messages ─────────────────────────────────────────────────────

class TestBuildMessages:
    def test_substitutes_triplet(self):
        a = _atomizer()
        messages = a._build_messages("cats are animals")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "cats are animals" in messages[1]["content"]

    def test_system_message_content(self):
        a = _atomizer()
        messages = a._build_messages("dogs like bones")
        assert "atomic" in messages[0]["content"].lower() or "split" in messages[0]["content"].lower()


# ── Test _build_task ─────────────────────────────────────────────────────────

class TestBuildTask:
    def test_first_pass_temperature(self):
        a = _atomizer()
        task = a._build_task(_payload())
        assert task["temperature"] == 0.0
        assert task["schema"] == AtomizationResult

    def test_retry_round_temperature(self):
        a = _atomizer()
        rc = RetryRoundConfig(temperature=0.5)
        task = a._build_task(_payload(), round_config=rc)
        assert task["temperature"] == 0.5


# ── Test _build_fallbacks ────────────────────────────────────────────────────

class TestBuildFallbacks:
    def test_three_word_triplet(self):
        payloads = [_payload("cats are animals")]
        fallbacks = Atomizer._build_fallbacks(payloads)
        assert len(fallbacks) == 1
        assert fallbacks[0][0].subject == "cats"
        assert fallbacks[0][0].predicate == "are"
        assert fallbacks[0][0].object == "animals"

    def test_multi_word_object(self):
        """Object can contain spaces — split on first two spaces only."""
        payloads = [_payload("cats like warm places")]
        fallbacks = Atomizer._build_fallbacks(payloads)
        assert fallbacks[0][0].subject == "cats"
        assert fallbacks[0][0].predicate == "like"
        assert fallbacks[0][0].object == "warm places"

    def test_single_word_fallback(self):
        """Can't split a single word — whole string becomes subject."""
        payloads = [_payload("cats")]
        fallbacks = Atomizer._build_fallbacks(payloads)
        assert fallbacks[0][0].subject == "cats"
        assert fallbacks[0][0].predicate == ""


# ── Test _classify ───────────────────────────────────────────────────────────

class TestClassify:
    def test_success(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        responses = [_result_json([_atomic("a", "b", "c")])]

        results, retries = a._classify(responses, originals, stats)
        assert len(results[0]) == 1
        assert results[0][0].subject == "a"
        assert stats.success == 1
        assert len(retries) == 0

    def test_split_into_multiple(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        responses = [_result_json([
            _atomic("cats", "are", "animals"),
            _atomic("dogs", "are", "animals"),
        ])]

        results, retries = a._classify(responses, originals, stats)
        assert len(results[0]) == 2
        assert stats.total_items == 2

    def test_context_too_long_keeps_original(self):
        a = _atomizer()
        stats = PhaseStats()
        orig = [AtomicTriplet(subject="x", predicate="y", object="z")]
        originals = [orig]
        responses = [ContextTooLongError("too long")]

        results, retries = a._classify(responses, originals, stats)
        assert results[0][0].subject == "x"  # kept original
        assert stats.context_too_long == 1
        assert len(retries) == 0

    def test_content_policy_keeps_original(self):
        a = _atomizer()
        stats = PhaseStats()
        orig = [AtomicTriplet(subject="x", predicate="y", object="z")]
        originals = [orig]
        responses = [ContentPolicyError("blocked")]

        results, retries = a._classify(responses, originals, stats)
        assert results[0][0].subject == "x"
        assert stats.content_policy == 1

    def test_parse_error_retryable(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        responses = [ParsingError("bad json")]

        results, retries = a._classify(responses, originals, stats)
        assert retries == [0]
        assert stats.parse_error == 1

    def test_malformed_json_retryable(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        responses = ["not valid json"]

        results, retries = a._classify(responses, originals, stats)
        assert retries == [0]
        assert stats.parse_error == 1

    def test_empty_result_keeps_original(self):
        a = _atomizer()
        stats = PhaseStats()
        orig = [AtomicTriplet(subject="x", predicate="y", object="z")]
        originals = [orig]
        responses = [_result_json([])]

        results, retries = a._classify(responses, originals, stats)
        assert results[0][0].subject == "x"  # kept original
        assert stats.empty == 1


# ── Test _apply_retries ──────────────────────────────────────────────────────

class TestApplyRetries:
    def test_recovered(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        results = [originals[0][:]]
        responses = [_result_json([_atomic("a", "b", "c")])]

        round_result, remaining = a._apply_retries(
            responses, [0], results, originals, stats,
        )
        assert round_result.recovered == 1
        assert round_result.still_failed == 0
        assert len(remaining) == 0
        assert results[0][0].subject == "a"

    def test_still_failed(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [[AtomicTriplet(subject="x", predicate="y", object="z")]]
        results = [originals[0][:]]
        responses = [ParsingError("bad")]

        round_result, remaining = a._apply_retries(
            responses, [0], results, originals, stats,
        )
        assert round_result.still_failed == 1
        assert remaining == [0]


# ── Test atomize_batch integration ───────────────────────────────────────────

class TestAtomizeBatch:
    @pytest.mark.asyncio
    async def test_batch_success(self):
        a = _atomizer()
        payloads = [
            _payload("cats are animals"),
            _payload("dogs are animals"),
        ]

        a.client.generate_batch = AsyncMock(return_value=[
            _result_json([_atomic("cats", "are", "animals")]),
            _result_json([_atomic("dogs", "are", "animals")]),
        ])

        results = await a.atomize_batch(payloads)
        assert len(results) == 2
        assert results[0][0].subject == "cats"
        assert results[1][0].subject == "dogs"

    @pytest.mark.asyncio
    async def test_batch_with_failure_keeps_original(self):
        a = _atomizer(max_retries=0)
        payloads = [
            _payload("cats are animals"),
            _payload("dogs are animals"),
        ]

        a.client.generate_batch = AsyncMock(return_value=[
            _result_json([_atomic("cats", "are", "animals")]),
            ParsingError("bad json"),
        ])

        results = await a.atomize_batch(payloads)
        assert len(results) == 2
        assert results[0][0].subject == "cats"
        # Failed item keeps original (fallback)
        assert results[1][0].subject == "dogs"
        assert results[1][0].predicate == "are"
        assert results[1][0].object == "animals"
