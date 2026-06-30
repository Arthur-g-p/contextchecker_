"""
Unit tests for normalization and shared context/question canonicalization helpers.

Covers:
- _canonicalize_keys (BaseService)
- _normalize_triplets (CheckingService helper)
"""

import pytest

from contextchecker.services.base import BaseService
from contextchecker.utils import canonicalize_triplets


# ── Test Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def legacy_item():
    """Item in legacy format (all old-style keys)."""
    return {
        "query": "What is the capital of France?",
        "context": ["France is a country in Europe.", "Paris is the capital of France."],
        "response": "The capital of France is Paris.",
        "claude2_response_kg": [
            {"triplet": ["France", "has capital", "Paris"], "human_label": "Entailment"},
            {"triplet": ["Paris", "is in", "Europe"], "human_label": "Neutral"}
        ]
    }


@pytest.fixture
def canonical_item():
    """Item in canonical format (all new-style keys)."""
    return {
        "question": "What is the capital of France?",
        "reference": ["France is a country in Europe.", "Paris is the capital of France."],
        "response": "The capital of France is Paris.",
        "claude2_response_kg": [
            {"subject": "France", "predicate": "has capital", "object": "Paris", "human_label": "Entailment"}
        ]
    }


@pytest.fixture
def mixed_item():
    """Item with mixed legacy and canonical elements in the same list."""
    return {
        "context": ["Some passage."],
        "question": "Already canonical question",
        "claude2_response_kg": [
            {"triplet": ["A", "B", "C"], "human_label": "Entailment"},
            {"subject": "X", "predicate": "Y", "object": "Z", "human_label": "Contradiction"}
        ]
    }


@pytest.fixture
def single_string_ref_item():
    """Item with reference as a single string instead of a list."""
    return {
        "reference": "Single passage as a string, not a list.",
        "claude2_response_kg": [
            {"triplet": ["A", "B", "C"], "human_label": "Entailment"}
        ]
    }


# ── _canonicalize_keys tests ─────────────────────────────────────────────────

class TestCanonicalizeKeys:

    def test_canonicalize_context_to_reference(self, legacy_item):
        """'context' key gets renamed to 'reference'."""
        data = [legacy_item]
        BaseService._canonicalize_keys(data)
        assert "reference" in data[0]
        assert "context" not in data[0]
        assert data[0]["reference"] == ["France is a country in Europe.", "Paris is the capital of France."]

    def test_canonicalize_query_to_question(self, legacy_item):
        """'query' key gets renamed to 'question'."""
        data = [legacy_item]
        BaseService._canonicalize_keys(data)
        assert "question" in data[0]
        assert "query" not in data[0]
        assert data[0]["question"] == "What is the capital of France?"

    def test_canonicalize_preserves_existing_reference(self):
        """If both 'context' AND 'reference' exist, 'reference' is NOT overwritten."""
        data = [{"reference": ["original"], "context": ["alias"]}]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == ["original"]
        assert data[0]["context"] == ["alias"]

    def test_canonicalize_preserves_existing_question(self):
        """If both 'question' AND 'query' exist, 'question' is NOT overwritten."""
        data = [{"question": "original", "query": "alias"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["question"] == "original"
        assert data[0]["query"] == "alias"

    def test_canonicalize_no_side_effects(self):
        """Items without context/query are untouched."""
        data = [{"response": "hello"}]
        BaseService._canonicalize_keys(data)
        assert data[0] == {"response": "hello"}

    def test_canonicalize_empty_list(self):
        """Empty list — no crash."""
        data = []
        BaseService._canonicalize_keys(data)
        assert data == []

    def test_canonicalize_mixed_items(self, legacy_item, canonical_item):
        """Multiple items with different key shapes in the same list."""
        data = [
            legacy_item,
            canonical_item,
            {"response": "untouched"}
        ]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == ["France is a country in Europe.", "Paris is the capital of France."]
        assert data[0]["question"] == "What is the capital of France?"
        assert data[1]["reference"] == ["France is a country in Europe.", "Paris is the capital of France."]
        assert "reference" not in data[2]

    def test_canonicalize_mutation_in_place(self, legacy_item):
        """Canonicalization mutates the original list items in-place."""
        data = [legacy_item]
        original_item = data[0]
        BaseService._canonicalize_keys(data)

