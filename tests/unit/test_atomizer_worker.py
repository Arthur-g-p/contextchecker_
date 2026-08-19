"""Unit tests for the Atomizer worker — classification, retry, and batch logic."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from contextchecker.workers.atomizer import (
    Atomizer,
    AtomicTriplet,
    AtomizationDecision,
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
        a = Atomizer(**defaults)
        # Set a default template for messages rendering
        a._prompt_template = "subject: {{subject}}, predicate: {{predicate}}, object: {{object}}, response: {{response}}"
        return a


def _payload(s: str = "cats", p: str = "are", o: str = "animals", response: str = "t"):
    return AtomizationPayload(subject=s, predicate=p, object=o, response=response, item_index=0, triplet_index=0)


def _result_json(reasoning: str = "ok", is_atomic: bool = True, split: list[dict] | None = None) -> str:
    """Build a valid AtomizationDecision JSON string."""
    return json.dumps({
        "reasoning": reasoning,
        "is_atomic": is_atomic,
        "split": split or []
    })


def _atomic(s: str, p: str, o: str) -> dict:
    return {"subject": s, "predicate": p, "object": o}


# ── Test _build_messages ─────────────────────────────────────────────────────

class TestBuildMessages:
    def test_substitutes_triplet(self):
        a = _atomizer()
        p = _payload(s="cats", p="are", o="animals")
        messages = a._build_messages(p)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "cats" in messages[1]["content"]
        assert "are" in messages[1]["content"]
        assert "animals" in messages[1]["content"]

    def test_system_message_content(self):
        a = _atomizer()
        p = _payload(s="dogs", p="like", o="bones")
        messages = a._build_messages(p)
        assert "atomic" in messages[0]["content"].lower() or "split" in messages[0]["content"].lower()


# ── Test _build_task ─────────────────────────────────────────────────────────

class TestBuildTask:
    def test_first_pass_temperature(self):
        a = _atomizer()
        task = a._build_task(_payload())
        assert task["temperature"] == 0.0
        assert task["schema"] == AtomizationDecision

    def test_retry_round_temperature(self):
        a = _atomizer()
        rc = RetryRoundConfig(temperature=0.5)
        task = a._build_task(_payload(), round_config=rc)
        assert task["temperature"] == 0.5


# ── Test _build_fallbacks ────────────────────────────────────────────────────

class TestBuildFallbacks:
    def test_creates_correct_number_of_fallbacks(self):
        payloads = [_payload("s1"), _payload("s2")]
        fallbacks = Atomizer._build_fallbacks(payloads)
        assert len(fallbacks) == 2
        for f in fallbacks:
            assert isinstance(f, AtomizationDecision)
            assert f.is_atomic is True
            assert len(f.split) == 0
            assert "fallback" in f.reasoning


# ── Test _classify ───────────────────────────────────────────────────────────

class TestClassify:
    def test_success(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [AtomizationDecision(reasoning="fallback", is_atomic=True, split=[])]
        responses = [_result_json(reasoning="already atomic", is_atomic=True)]

        results, retries = a._classify(responses, originals, stats)
        assert len(results) == 1
        assert results[0].reasoning == "already atomic"
        assert results[0].is_atomic is True
        assert stats.success == 1
        assert len(retries) == 0

    def test_split_into_multiple(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [AtomizationDecision(reasoning="fallback", is_atomic=True, split=[])]
        responses = [_result_json(
            reasoning="needs split",
            is_atomic=False,
            split=[
                _atomic("cats", "are", "animals"),
                _atomic("dogs", "are", "animals"),
            ]
        )]

        results, retries = a._classify(responses, originals, stats)
        assert len(results) == 1
        assert results[0].is_atomic is False
        assert len(results[0].split) == 2
        assert results[0].split[0].subject == "cats"
        assert stats.success == 1
        assert len(retries) == 0

    def test_context_too_long_keeps_original(self):
        a = _atomizer()
        stats = PhaseStats()
        orig = AtomizationDecision(reasoning="orig reasoning", is_atomic=True, split=[])
        originals = [orig]
        responses = [ContextTooLongError("too long")]

        results, retries = a._classify(responses, originals, stats)
        assert results[0] is orig
        assert stats.context_too_long == 1
        assert len(retries) == 0

    def test_content_policy_keeps_original(self):
        a = _atomizer()
        stats = PhaseStats()
        orig = AtomizationDecision(reasoning="orig reasoning", is_atomic=True, split=[])
        originals = [orig]
        responses = [ContentPolicyError("blocked")]

        results, retries = a._classify(responses, originals, stats)
        assert results[0] is orig
        assert stats.content_policy == 1
        assert len(retries) == 0

    def test_parse_error_retryable(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [AtomizationDecision(reasoning="orig reasoning", is_atomic=True, split=[])]
        responses = [ParsingError("bad json")]

        results, retries = a._classify(responses, originals, stats)
        assert retries == [0]
        assert stats.parse_error == 1

    def test_malformed_json_retryable(self):
        a = _atomizer()
        stats = PhaseStats()
        originals = [AtomizationDecision(reasoning="orig", is_atomic=True, split=[])]
        responses = ["not valid json"]

        results, retries = a._classify(responses, originals, stats)
        assert retries == [0]
        assert stats.parse_error == 1


# ── Test _apply_retries ──────────────────────────────────────────────────────

class TestApplyRetries:
    def test_recovered(self):
        a = _atomizer()
        stats = PhaseStats()
        stats.failed_indices = [0]
        originals = [AtomizationDecision(reasoning="orig", is_atomic=True, split=[])]
        results = list(originals)
        responses = [_result_json(reasoning="recovered reasoning", is_atomic=True)]

        round_result, remaining = a._apply_retries(
            responses, [0], results, originals, stats,
        )
        assert round_result.recovered == 1
        assert round_result.still_failed == 0
        assert len(remaining) == 0
        assert results[0].reasoning == "recovered reasoning"
        assert 0 not in stats.failed_indices

    def test_still_failed(self):
        a = _atomizer()
        stats = PhaseStats()
        stats.failed_indices = [0]
        originals = [AtomizationDecision(reasoning="orig", is_atomic=True, split=[])]
        results = list(originals)
        responses = [ParsingError("bad")]

        round_result, remaining = a._apply_retries(
            responses, [0], results, originals, stats,
        )
        assert round_result.still_failed == 1
        assert remaining == [0]
        assert 0 in stats.failed_indices


# ── Test atomize_batch integration ───────────────────────────────────────────

class TestAtomizeBatch:
    @pytest.mark.asyncio
    async def test_batch_success(self):
        a = _atomizer()
        payloads = [
            _payload("cats", "are", "animals"),
            _payload("dogs", "are", "animals"),
        ]

        a.client.generate_batch = AsyncMock(return_value=[
            _result_json(reasoning="r1", is_atomic=True),
            _result_json(reasoning="r2", is_atomic=True),
        ])

        results = await a.atomize_batch(payloads)
        assert len(results) == 2
        assert results[0].reasoning == "r1"
        assert results[1].reasoning == "r2"

    @pytest.mark.asyncio
    async def test_batch_with_failure_keeps_original(self):
        a = _atomizer()
        payloads = [
            _payload("cats", "are", "animals"),
            _payload("dogs", "are", "animals"),
        ]

        # First pass, then one response per retry round — the failing item never
        # recovers, so it must end on the fallback decision.
        a.client.generate_batch = AsyncMock(side_effect=[
            [_result_json(reasoning="r1", is_atomic=True), ParsingError("bad json")],
            [ParsingError("bad json")],
            [ParsingError("bad json")],
        ])

        results = await a.atomize_batch(payloads)
        assert len(results) == 2
        assert results[0].reasoning == "r1"
        # Failed item keeps original fallback decision
        assert results[1].reasoning == "not processed (fallback)"
        assert results[1].is_atomic is True
