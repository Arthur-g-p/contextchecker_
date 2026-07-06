"""
Unit tests for ExtractionService._validate(), _filter(), and _serialize() dedup.

Tests the service's input validation, skip logic, and the destructive
dedup pass directly. The Extractor worker is mocked out at construction —
these tests never touch the LLM.
"""

import pytest
from unittest.mock import patch, MagicMock

from contextchecker.services.extraction import ExtractionService
from contextchecker.workers.extractor import Triplet
from contextchecker.exceptions import InvalidInputError, FilterError


FAKE_API_KEY = "test-key-12345"
MODEL = "gemini-2.0-flash"
KG_KEY = f"{MODEL}_response_kg"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_api_key(monkeypatch):
    """Every test gets a valid API key so the service can be constructed."""
    monkeypatch.setattr("contextchecker.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)


@pytest.fixture
def service():
    """ExtractionService with the Extractor worker stubbed out (dedup on)."""
    with patch("contextchecker.services.extraction.Extractor"):
        return ExtractionService(model=MODEL)


@pytest.fixture
def service_no_dedup():
    """ExtractionService with dedup disabled (the evaluator's path)."""
    with patch("contextchecker.services.extraction.Extractor"):
        return ExtractionService(model=MODEL, dedup=False)


def _trip(s, p, o):
    return Triplet(subject=s, predicate=p, object=o)


# ── Validation fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def all_valid():
    """Every item has a 'response' key."""
    return [
        {"response": "Paris is in France"},
        {"response": "The sky is blue"},
    ]


@pytest.fixture
def mixed_valid_invalid():
    """Some items have 'response', some don't."""
    return [
        {"response": "Paris is in France"},
        {"question": "Where is Paris?"},       # no response
        {"answer": "somewhere"},               # no response
        {"response": "The sky is blue"},
    ]


@pytest.fixture
def all_invalid():
    """No item has a 'response' key."""
    return [
        {"question": "Where is Paris?"},
        {"answer": "I don't know"},
    ]


# ── Filter fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def already_processed():
    """Items that already have the kg_key — should be skipped."""
    return [
        {"response": "Paris is in France", KG_KEY: [{"subject": "Paris", "predicate": "is in", "object": "France"}]},
        {"response": "The sky is blue", KG_KEY: [{"subject": "Sky", "predicate": "has color", "object": "blue"}]},
    ]


@pytest.fixture
def mix_processed_and_pending():
    """One already done, one still needs extraction."""
    return [
        {"response": "Paris is in France", KG_KEY: [{"subject": "Paris", "predicate": "is in", "object": "France"}]},
        {"response": "Water boils at 100 degrees"},
    ]


# ── Validation tests ─────────────────────────────────────────────────────────

class TestValidate:

    def test_all_valid_passes(self, service, all_valid):
        """All items have 'response' → all returned."""
        result = service._validate(all_valid)
        assert len(result) == 2

    def test_mixed_skips_invalid(self, service, mixed_valid_invalid):
        """Items without 'response' are dropped, valid ones survive."""
        result = service._validate(mixed_valid_invalid)
        assert len(result) == 2
        assert all("response" in item for item in result)

    def test_all_invalid_raises(self, service, all_invalid):
        """Zero valid items → InvalidInputError."""
        with pytest.raises(InvalidInputError, match="No items contain"):
            service._validate(all_invalid)

    def test_empty_list_raises(self, service):
        """Empty input → InvalidInputError."""
        with pytest.raises(InvalidInputError, match="No items contain"):
            service._validate([])

    def test_single_valid_item(self, service):
        """One valid item in the list → returned as-is."""
        data = [{"response": "Hello"}]
        result = service._validate(data)
        assert len(result) == 1
        assert result[0]["response"] == "Hello"

    def test_response_key_with_empty_string(self, service):
        """Item has 'response' key but value is empty — still passes validation.
        (Abstention detection is _filter's job, not _validate's.)"""
        data = [{"response": ""}]
        result = service._validate(data)
        assert len(result) == 1


# ── Filter tests ─────────────────────────────────────────────────────────────

class TestFilter:

    def test_all_already_processed_returns_empty(self, service, already_processed):
        """Every item has kg_key, nothing to extract → returns empty lists."""
        pending, abstained, skipped = service._filter(already_processed)
        assert len(pending) == 0
        assert len(abstained) == 0
        assert skipped == 2

    def test_already_processed_skipped(self, service, mix_processed_and_pending):
        """Processed items are skipped, pending ones returned."""
        pending, abstained, skipped = service._filter(mix_processed_and_pending)
        assert len(pending) == 1
        assert skipped == 1
        assert pending[0]["response"] == "Water boils at 100 degrees"

    def test_pending_items_returned(self, service, all_valid):
        """No kg_key, no abstention → all items are pending."""
        pending, abstained, skipped = service._filter(all_valid)
        assert len(pending) == 2
        assert len(abstained) == 0
        assert skipped == 0

    def test_processed_items_data_untouched(self, service, already_processed):
        """Existing kg_key data must not be modified (just skipped)."""
        original_triplets = already_processed[0][KG_KEY].copy()
        service._filter(already_processed)
        assert already_processed[0][KG_KEY] == original_triplets

    def test_abstention_separated_from_pending(self, service):
        """Abstentions go to the abstained bucket, real text to pending."""
        data = [
            {"response": "I don't know"},             # abstention
            {"response": "The Earth orbits the Sun"},  # pending
        ]
        pending, abstained, skipped = service._filter(data)
        assert len(pending) == 1
        assert len(abstained) == 1
        assert skipped == 0
        assert pending[0]["response"] == "The Earth orbits the Sun"

    def test_only_abstentions_no_error(self, service):
        """All items are abstentions → no FilterError (they still need [] written)."""
        data = [
            {"response": "I don't know"},
            {"response": ""},
        ]
        pending, abstained, skipped = service._filter(data)
        assert len(pending) == 0
        assert len(abstained) == 2
        assert skipped == 0


# ── Serialize / dedup tests ──────────────────────────────────────────────────

class TestSerializeDedup:

    def test_default_is_dedup_on(self, service):
        assert service._dedup is True

    def test_dedup_on_removes_exact_duplicates(self, service):
        items = [{}]
        results = [[_trip("a", "b", "c"), _trip("a", "b", "c"), _trip("d", "e", "f")]]
        removed = service._serialize(items, results)
        assert removed == 1
        assert len(items[0][KG_KEY]) == 2

    def test_dedup_on_keeps_first_occurrence(self, service):
        items = [{}]
        results = [[_trip("a", "b", "c"), _trip("a", "b", "c")]]
        service._serialize(items, results)
        assert items[0][KG_KEY] == [{"subject": "a", "predicate": "b", "object": "c"}]

    def test_dedup_off_keeps_duplicates(self, service_no_dedup):
        items = [{}]
        results = [[_trip("a", "b", "c"), _trip("a", "b", "c")]]
        removed = service_no_dedup._serialize(items, results)
        assert removed == 0
        assert len(items[0][KG_KEY]) == 2

    def test_removed_count_aggregates_across_items(self, service):
        items = [{}, {}]
        results = [
            [_trip("a", "b", "c"), _trip("a", "b", "c")],
            [_trip("x", "y", "z"), _trip("x", "y", "z"), _trip("x", "y", "z")],
        ]
        removed = service._serialize(items, results)
        assert removed == 3

    def test_no_duplicates_removes_nothing(self, service):
        items = [{}]
        results = [[_trip("a", "b", "c"), _trip("d", "e", "f")]]
        removed = service._serialize(items, results)
        assert removed == 0
        assert len(items[0][KG_KEY]) == 2


# ── Outcome-marker tests (error vs abstention disambiguation) ────────────────

ERROR_KEY = f"{MODEL}_extraction_error"


class TestOutcomeMarkers:
    """[] on disk is never ambiguous: errors carry {model}_extraction_error,
    empty-without-error carries is_abstention."""

    def test_error_cause_marks_item_and_empties_kg(self, service):
        items = [{}, {}]
        results = [[_trip("a", "b", "c")], []]
        service._serialize(items, results, error_causes={1: "parse_failure"})
        assert items[0][KG_KEY] == [{"subject": "a", "predicate": "b", "object": "c"}]
        assert ERROR_KEY not in items[0]
        assert items[1][KG_KEY] == []
        assert items[1][ERROR_KEY] == "parse_failure"
        assert "is_abstention" not in items[1]

    def test_empty_without_error_is_llm_abstention(self, service):
        items = [{}]
        results = [[]]
        service._serialize(items, results)
        assert items[0][KG_KEY] == []
        assert items[0]["is_abstention"] is True
        assert items[0]["abstention_source"] == "llm"
        assert ERROR_KEY not in items[0]

    def test_stale_markers_cleared_on_new_result(self, service):
        """A re-run that succeeds must drop the previous error/abstention keys."""
        items = [{
            ERROR_KEY: "parse_failure",
            "is_abstention": True,
            "abstention_source": "llm",
        }]
        results = [[_trip("a", "b", "c")]]
        service._serialize(items, results)
        assert ERROR_KEY not in items[0]
        assert "is_abstention" not in items[0]
        assert "abstention_source" not in items[0]
        assert len(items[0][KG_KEY]) == 1

    def test_filter_repends_errored_items(self, service):
        """An error marker is not a result: the item is re-extracted on re-runs."""
        valid = [
            {"response": "Paris is in France", KG_KEY: [], ERROR_KEY: "parse_failure"},
            {"response": "The sky is blue", KG_KEY: [
                {"subject": "Sky", "predicate": "has color", "object": "blue"}]},
        ]
        pending, abstained, skipped = service._filter(valid)
        assert len(pending) == 1
        assert pending[0][ERROR_KEY] == "parse_failure"
        assert skipped == 1
        assert abstained == []


# ── Worker cause-recording tests ─────────────────────────────────────────────

class TestWorkerErrorCauses:
    """The Extractor worker records WHY each batch index failed on
    last_stats.error_causes so the service can persist it."""

    def test_classify_records_permanent_causes(self):
        from contextchecker.workers.extractor import Extractor
        from contextchecker.exceptions import ContextTooLongError, ContentPolicyError
        from contextchecker.stats import PhaseStats

        with patch("contextchecker.workers.extractor.LLMClient"):
            extractor = Extractor(api_key=FAKE_API_KEY, model=MODEL)

        stats = PhaseStats()
        responses = [
            ContextTooLongError("too long"),
            ContentPolicyError("blocked"),
            '{"triplets": []}',
        ]
        results, retry = extractor._classify(responses, stats)
        assert stats.error_causes == {0: "context_too_long", 1: "content_policy"}
        assert retry == []
        assert results[0] == [] and results[1] == []

    def test_exhausted_retries_marked_as_parse_failure(self):
        import asyncio
        from contextchecker.workers.extractor import Extractor
        from contextchecker.models import ExtractionPayload

        with patch("contextchecker.workers.extractor.LLMClient"):
            extractor = Extractor(api_key=FAKE_API_KEY, model=MODEL, max_retries=1)

        async def always_invalid(tasks, **kwargs):
            return [ValueError("not json") for _ in tasks]

        extractor.client.generate_batch = always_invalid
        results = asyncio.run(
            extractor.extract_batch([ExtractionPayload(text="some text")])
        )
        assert results == [[]]
        assert extractor.last_stats.error_causes == {0: "parse_failure"}
