"""
Unit tests for ExtractionService._validate() and _filter().

Tests the service's input validation and skip logic directly.
The Extractor worker is mocked out at construction — these tests
never touch the LLM.
"""

import pytest
from unittest.mock import patch, MagicMock

from contextchecker.services.extraction import ExtractionService
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
    """ExtractionService with the Extractor worker stubbed out."""
    with patch("contextchecker.services.extraction.Extractor"):
        return ExtractionService(model=MODEL)


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
