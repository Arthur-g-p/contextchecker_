"""
Unit tests for normalization and shared context/question canonicalization helpers.

Covers:
- _canonicalize_keys (BaseService)
- _normalize_triplets (CheckingService helper)
"""

import pytest

from contextchecker.services.base import BaseService
from contextchecker.services.checking import _normalize_triplets


# ── _canonicalize_keys tests ─────────────────────────────────────────────────

class TestCanonicalizeKeys:

    def test_context_renamed_to_reference(self):
        """'context' → 'reference' when 'reference' is absent."""
        data = [{"context": ["passage one"], "response": "answer"}]
        BaseService._canonicalize_keys(data)
        assert "reference" in data[0]
        assert "context" not in data[0]
        assert data[0]["reference"] == ["passage one"]

    def test_query_renamed_to_question(self):
        """'query' → 'question' when 'question' is absent."""
        data = [{"query": "what is X?", "response": "X is Y"}]
        BaseService._canonicalize_keys(data)
        assert "question" in data[0]
        assert "query" not in data[0]
        assert data[0]["question"] == "what is X?"

    def test_both_aliases_renamed(self):
        """Both 'context' and 'query' are renamed in the same item."""
        data = [{"context": ["ref"], "query": "q text", "response": "r"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == ["ref"]
        assert data[0]["question"] == "q text"
        assert "context" not in data[0]
        assert "query" not in data[0]

    def test_reference_already_exists_no_overwrite(self):
        """If 'reference' already exists, 'context' is NOT renamed (no overwrite)."""
        data = [{"reference": ["original"], "context": ["alias"]}]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == ["original"]
        assert data[0]["context"] == ["alias"]  # untouched

    def test_question_already_exists_no_overwrite(self):
        """If 'question' already exists, 'query' is NOT renamed."""
        data = [{"question": "original", "query": "alias"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["question"] == "original"
        assert data[0]["query"] == "alias"  # untouched

    def test_canonical_keys_already_present(self):
        """Items with canonical keys are untouched."""
        data = [{"reference": ["ref"], "question": "q", "response": "r"}]
        BaseService._canonicalize_keys(data)
        assert data[0] == {"reference": ["ref"], "question": "q", "response": "r"}

    def test_no_relevant_keys(self):
        """Items with neither alias nor canonical key — untouched."""
        data = [{"response": "hello"}]
        BaseService._canonicalize_keys(data)
        assert data[0] == {"response": "hello"}

    def test_empty_list(self):
        """Empty list — no crash."""
        data = []
        BaseService._canonicalize_keys(data)
        assert data == []

    def test_mixed_items(self):
        """Multiple items with different key shapes."""
        data = [
            {"context": ["ref1"], "response": "r1"},
            {"reference": ["ref2"], "query": "q2", "response": "r2"},
            {"response": "r3"},
        ]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == ["ref1"]
        assert data[1]["reference"] == ["ref2"]
        assert data[1]["question"] == "q2"
        assert "reference" not in data[2]

    def test_mutation_in_place(self):
        """Canonicalization mutates the original list, not a copy."""
        data = [{"context": ["ref"]}]
        original_item = data[0]
        BaseService._canonicalize_keys(data)
        assert data[0] is original_item  # same object


# ── _normalize_triplets tests ────────────────────────────────────────────────

class TestNormalizeTriplets:

    def test_legacy_array_format(self):
        """{"triplet": [s, p, o]} → {"subject": s, "predicate": p, "object": o}."""
        kg = [{"triplet": ["France", "has capital", "Paris"]}]
        _normalize_triplets(kg)
        assert kg[0] == {"subject": "France", "predicate": "has capital", "object": "Paris"}

    def test_legacy_with_human_label_preserved(self):
        """Extra keys like 'human_label' survive normalization."""
        kg = [{"triplet": ["X", "is", "Y"], "human_label": "Entailment"}]
        _normalize_triplets(kg)
        assert kg[0]["subject"] == "X"
        assert kg[0]["predicate"] == "is"
        assert kg[0]["object"] == "Y"
        assert kg[0]["human_label"] == "Entailment"
        assert "triplet" not in kg[0]

    def test_canonical_format_untouched(self):
        """Already canonical dicts are left as-is."""
        original = {"subject": "A", "predicate": "B", "object": "C"}
        kg = [original.copy()]
        _normalize_triplets(kg)
        assert kg[0] == original

    def test_mixed_formats(self):
        """Mix of legacy and canonical in the same list."""
        kg = [
            {"triplet": ["X", "is", "Y"]},
            {"subject": "A", "predicate": "B", "object": "C"},
        ]
        _normalize_triplets(kg)
        assert kg[0] == {"subject": "X", "predicate": "is", "object": "Y"}
        assert kg[1] == {"subject": "A", "predicate": "B", "object": "C"}

    def test_empty_list(self):
        """Empty kg list — no crash."""
        kg = []
        _normalize_triplets(kg)
        assert kg == []

    def test_triplet_key_with_subject_already_present(self):
        """If both 'triplet' and 'subject' exist, leave as-is (don't overwrite)."""
        kg = [{"triplet": ["X", "Y", "Z"], "subject": "existing"}]
        _normalize_triplets(kg)
        assert kg[0]["subject"] == "existing"
        assert kg[0]["triplet"] == ["X", "Y", "Z"]  # not popped

    def test_mutation_in_place(self):
        """Normalization mutates the original list items."""
        kg = [{"triplet": ["A", "B", "C"]}]
        original_item = kg[0]
        _normalize_triplets(kg)
        assert kg[0] is original_item  # same object

    def test_multiple_legacy_items(self):
        """Multiple legacy triplets all get normalized."""
        kg = [
            {"triplet": ["A", "is", "B"], "human_label": "Entailment"},
            {"triplet": ["C", "has", "D"], "human_label": "Contradiction"},
            {"triplet": ["E", "near", "F"], "human_label": "Neutral"},
        ]
        _normalize_triplets(kg)
        assert all("subject" in c for c in kg)
        assert all("triplet" not in c for c in kg)
        assert all("human_label" in c for c in kg)
