"""
Unit tests for _canonicalize_keys (BaseService) and _normalize_triplets (CheckingService).

Tests the normalization logic that runs before any service-specific validation.
"""

import pytest
from unittest.mock import patch

from contextchecker.services.base import BaseService
from contextchecker.services.checking import _normalize_triplets, CheckingService
from contextchecker.exceptions import InvalidInputError


# ── _canonicalize_keys tests ─────────────────────────────────────────────────

class TestCanonicalizeKeys:

    def test_context_renamed_to_reference(self):
        """'context' → 'reference' when 'reference' is absent."""
        data = [{"context": "some text", "response": "answer"}]
        BaseService._canonicalize_keys(data)
        assert "reference" in data[0]
        assert "context" not in data[0]
        assert data[0]["reference"] == "some text"

    def test_query_renamed_to_question(self):
        """'query' → 'question' when 'question' is absent."""
        data = [{"query": "what is X?", "response": "X is Y"}]
        BaseService._canonicalize_keys(data)
        assert "question" in data[0]
        assert "query" not in data[0]
        assert data[0]["question"] == "what is X?"

    def test_both_aliases_renamed(self):
        """Both 'context' and 'query' are renamed in the same item."""
        data = [{"context": "ref text", "query": "q text", "response": "r"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == "ref text"
        assert data[0]["question"] == "q text"
        assert "context" not in data[0]
        assert "query" not in data[0]

    def test_reference_already_exists_no_overwrite(self):
        """If 'reference' already exists, 'context' is NOT renamed (no overwrite)."""
        data = [{"reference": "original", "context": "alias"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == "original"
        assert data[0]["context"] == "alias"  # untouched

    def test_question_already_exists_no_overwrite(self):
        """If 'question' already exists, 'query' is NOT renamed."""
        data = [{"question": "original", "query": "alias"}]
        BaseService._canonicalize_keys(data)
        assert data[0]["question"] == "original"
        assert data[0]["query"] == "alias"  # untouched

    def test_canonical_keys_already_present(self):
        """Items with canonical keys are untouched."""
        data = [{"reference": "ref", "question": "q", "response": "r"}]
        BaseService._canonicalize_keys(data)
        assert data[0] == {"reference": "ref", "question": "q", "response": "r"}

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
            {"context": "ref1", "response": "r1"},
            {"reference": "ref2", "query": "q2", "response": "r2"},
            {"response": "r3"},
        ]
        BaseService._canonicalize_keys(data)
        assert data[0]["reference"] == "ref1"
        assert data[1]["reference"] == "ref2"
        assert data[1]["question"] == "q2"
        assert "reference" not in data[2]

    def test_mutation_in_place(self):
        """Canonicalization mutates the original list, not a copy."""
        data = [{"context": "ref"}]
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


# ── Integration: CheckingService._validate with normalization ────────────────

FAKE_API_KEY = "test-key-12345"
EXTRACTOR_MODEL = "gemini-2.0-flash"
KG_KEY = f"{EXTRACTOR_MODEL}_response_kg"


class TestCheckingValidation:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    @pytest.fixture
    def service(self):
        with patch("contextchecker.services.checking.Checker"):
            return CheckingService(model="checker-model", extractor_model=EXTRACTOR_MODEL)

    def test_legacy_triplets_normalized_during_validation(self, service):
        """_validate normalizes legacy triplet format in-place."""
        data = [
            {
                "reference": "some ref",
                KG_KEY: [{"triplet": ["X", "is", "Y"], "human_label": "Entailment"}],
            }
        ]
        valid = service._validate(data)
        # After validation, triplets should be canonical
        assert valid[0][KG_KEY][0]["subject"] == "X"
        assert "triplet" not in valid[0][KG_KEY][0]
        assert valid[0][KG_KEY][0]["human_label"] == "Entailment"

    def test_all_empty_kg_raises(self, service):
        """All items have empty _response_kg → InvalidInputError."""
        data = [
            {"reference": "ref1", KG_KEY: []},
            {"reference": "ref2", KG_KEY: []},
        ]
        with pytest.raises(InvalidInputError, match="empty"):
            service._validate(data)

    def test_missing_reference_dropped(self, service):
        """Items without 'reference' are dropped."""
        data = [
            {KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},  # no reference
            {"reference": "ref", KG_KEY: [{"subject": "X", "predicate": "Y", "object": "Z"}]},
        ]
        valid = service._validate(data)
        assert len(valid) == 1
        assert valid[0]["reference"] == "ref"

    def test_missing_kg_key_dropped(self, service):
        """Items without _response_kg are dropped."""
        data = [
            {"reference": "ref"},  # no kg_key
            {"reference": "ref2", KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        valid = service._validate(data)
        assert len(valid) == 1

    def test_context_alias_works_after_canonicalize(self, service):
        """'context' is renamed to 'reference' by _canonicalize_keys before _validate."""
        data = [
            {"context": "some ref", KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        # Simulate what run() does: canonicalize then validate
        service._canonicalize_keys(data)
        valid = service._validate(data)
        assert len(valid) == 1
        assert valid[0]["reference"] == "some ref"
