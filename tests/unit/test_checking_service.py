"""
Unit tests for CheckingService methods and helper functions.

Covers:
- _format_reference, _reference_word_count (checker worker)
- _effective_joint_num (checking service)
- _triplet_to_text (CheckingService)
- _flat_to_map (CheckingService)
- _serialize (CheckingService)
- _warn_oversized_references (CheckingService)
- CheckingService._validate integration
- CheckingService._filter
- _build_payloads (CheckingService)
"""

import pytest
from unittest.mock import patch

from contextchecker.services.checking import (
    _effective_joint_num,
    CheckingService,
    DEFAULT_MAX_WORDS,
    CONTEXT_BUDGET_RATIO,
)
from contextchecker.workers.checker import (
    Verdict,
    _format_reference,
    _reference_word_count,
)
from contextchecker.exceptions import InvalidInputError, FilterError


FAKE_API_KEY = "test-key-12345"
EXTRACTOR_MODEL = "gemini-2.0-flash"
KG_KEY = f"{EXTRACTOR_MODEL}_response_kg"


# ── _format_reference tests ─────────────────────────────────────────────────

class TestFormatReference:

    def test_single_passage(self):
        """Single passage returned as-is, no numbering."""
        assert _format_reference(["Hello world"]) == "Hello world"

    def test_multiple_passages(self):
        """Multiple passages numbered [Passage 1], [Passage 2], etc."""
        result = _format_reference(["First", "Second", "Third"])
        assert "[Passage 1] First" in result
        assert "[Passage 2] Second" in result
        assert "[Passage 3] Third" in result

    def test_empty_list(self):
        """Edge case: empty list."""
        result = _format_reference([])
        assert result == ""

    def test_two_passages_newline_separated(self):
        """Passages are joined with newlines."""
        result = _format_reference(["A", "B"])
        assert result == "[Passage 1] A\n[Passage 2] B"


# ── _reference_word_count tests ──────────────────────────────────────────────

class TestReferenceWordCount:

    def test_single_passage(self):
        assert _reference_word_count(["hello world"]) == 2

    def test_multiple_passages(self):
        assert _reference_word_count(["hello world", "foo bar baz"]) == 5

    def test_empty_list(self):
        assert _reference_word_count([]) == 0

    def test_empty_string_passage(self):
        assert _reference_word_count([""]) == 0


# ── _effective_joint_num tests ───────────────────────────────────────────────

class TestEffectiveJointNum:

    def test_small_reference_no_reduction(self):
        """Short reference — joint_num unchanged."""
        ref = ["Small ref"]
        claims = ["claim one", "claim two", "claim three"]
        result = _effective_joint_num(ref, claims, joint_num=10, max_words=6000)
        assert result == 10

    def test_huge_reference_reduces_to_one(self):
        """Reference alone exceeds budget — falls back to 1."""
        ref = [" ".join(["word"] * 5000)]  # 5000 words
        claims = ["claim"]
        result = _effective_joint_num(ref, claims, joint_num=10, max_words=6000)
        assert result == 1

    def test_moderate_reference_partial_reduction(self):
        """Reference eats into budget, fewer claims fit."""
        # Budget: 6000 * 0.75 = 4500. Ref = 4000 words. Available = 4500 - 4000 - 100 = 400.
        # Each claim ~2 words. Fits ~200 claims. joint_num=10 wins.
        ref = [" ".join(["word"] * 4000)]
        claims = ["claim one"] * 5
        result = _effective_joint_num(ref, claims, joint_num=10, max_words=6000)
        assert result == 10  # 200 fit, capped at 10

    def test_tight_budget_reduces_below_joint_num(self):
        """Budget is tight — effective num drops below joint_num."""
        # Budget: 500 * 0.75 = 375. Ref = 300 words. Available = 375 - 300 - 100 = -25 → 1
        ref = [" ".join(["word"] * 300)]
        claims = ["claim"] * 5
        result = _effective_joint_num(ref, claims, joint_num=10, max_words=500)
        assert result == 1

    def test_no_claims_returns_joint_num(self):
        """Empty claims list → return joint_num as-is."""
        result = _effective_joint_num(["ref"], [], joint_num=10, max_words=6000)
        assert result == 10

    def test_multiple_passages_word_count(self):
        """Reference is list of passages — total word count used."""
        # 3 passages × 100 words = 300 words total
        ref = [" ".join(["word"] * 100) for _ in range(3)]
        claims = ["claim"] * 5
        result = _effective_joint_num(ref, claims, joint_num=10, max_words=6000)
        assert result == 10  # plenty of room


# ── _triplet_to_text tests ───────────────────────────────────────────────────

class TestTripletToText:

    def test_basic(self):
        t = {"subject": "France", "predicate": "has capital", "object": "Paris"}
        assert CheckingService._triplet_to_text(t) == "France has capital Paris"

    def test_single_word_fields(self):
        t = {"subject": "A", "predicate": "is", "object": "B"}
        assert CheckingService._triplet_to_text(t) == "A is B"


# ── _flat_to_map tests ──────────────────────────────────────────────────────

class TestFlatToMap:

    def test_basic_mapping(self):
        """Flat payloads+verdicts → nested dict."""
        from contextchecker.models import CheckingPayload

        payloads = [
            CheckingPayload(claim="c1", reference=["r"], item_index=0, claim_index=0),
            CheckingPayload(claim="c2", reference=["r"], item_index=0, claim_index=1),
            CheckingPayload(claim="c3", reference=["r"], item_index=1, claim_index=0),
        ]
        verdicts = [Verdict.ENTAILMENT, Verdict.CONTRADICTION, Verdict.NEUTRAL]

        result = CheckingService._flat_to_map(payloads, verdicts)
        assert result == {
            0: {0: Verdict.ENTAILMENT, 1: Verdict.CONTRADICTION},
            1: {0: Verdict.NEUTRAL},
        }

    def test_with_none_verdicts(self):
        """None verdicts (failures) are preserved."""
        from contextchecker.models import CheckingPayload

        payloads = [
            CheckingPayload(claim="c1", reference=["r"], item_index=0, claim_index=0),
            CheckingPayload(claim="c2", reference=["r"], item_index=0, claim_index=1),
        ]
        verdicts = [Verdict.ENTAILMENT, None]

        result = CheckingService._flat_to_map(payloads, verdicts)
        assert result == {0: {0: Verdict.ENTAILMENT, 1: None}}

    def test_empty_inputs(self):
        result = CheckingService._flat_to_map([], [])
        assert result == {}


# ── _serialize tests ─────────────────────────────────────────────────────────

class TestSerialize:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    @pytest.fixture
    def service(self):
        with patch("contextchecker.services.checking.Checker"):
            return CheckingService(model="checker-model", extractor_model=EXTRACTOR_MODEL)

    def test_basic_serialize(self, service):
        """Verdicts written onto each triplet as string values."""
        items = [
            {KG_KEY: [
                {"subject": "A", "predicate": "is", "object": "B"},
                {"subject": "C", "predicate": "is", "object": "D"},
            ]},
        ]
        verdicts_map = {0: {0: Verdict.ENTAILMENT, 1: Verdict.CONTRADICTION}}
        service._serialize(items, verdicts_map)
        assert items[0][KG_KEY][0]["checker-model_checker_verdict"] == "Entailment"
        assert items[0][KG_KEY][1]["checker-model_checker_verdict"] == "Contradiction"
        # The legacy root-level parallel list is never written anymore
        assert "checker-model_checker_verdicts" not in items[0]

    def test_serialize_with_none_gaps(self, service):
        """None verdicts (gaps/failures) written as None on the triplet."""
        items = [
            {KG_KEY: [
                {"subject": "A", "predicate": "is", "object": "B"},
                {"subject": "C", "predicate": "is", "object": "D"},
            ]},
        ]
        verdicts_map = {0: {0: Verdict.ENTAILMENT, 1: None}}
        service._serialize(items, verdicts_map)
        assert items[0][KG_KEY][0]["checker-model_checker_verdict"] == "Entailment"
        assert items[0][KG_KEY][1]["checker-model_checker_verdict"] is None

    def test_serialize_missing_item_in_map(self, service):
        """Item missing from verdicts_map → triplets left untouched."""
        items = [
            {KG_KEY: [{"subject": "A", "predicate": "is", "object": "B"}]},
        ]
        verdicts_map = {}  # empty
        service._serialize(items, verdicts_map)
        assert "checker-model_checker_verdict" not in items[0][KG_KEY][0]


# ── _warn_oversized_references tests ─────────────────────────────────────────

class TestWarnOversizedReferences:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    def test_no_warning_when_max_words_none(self, caplog):
        """No warning when max_words is None (single mode, unset)."""
        with patch("contextchecker.services.checking.Checker"):
            service = CheckingService(
                model="checker", extractor_model=EXTRACTOR_MODEL,
                joint=False, max_words=None,
            )
        items = [{"reference": [" ".join(["word"] * 10000)], KG_KEY: []}]
        service._warn_oversized_references(items)
        assert "too large" not in caplog.text

    def test_warning_when_reference_exceeds_budget(self, caplog):
        """Warning emitted when reference exceeds the word budget."""
        import logging
        with patch("contextchecker.services.checking.Checker"):
            service = CheckingService(
                model="checker", extractor_model=EXTRACTOR_MODEL,
                max_words=100,
            )
        items = [{"reference": [" ".join(["word"] * 200)], KG_KEY: []}]
        with caplog.at_level(logging.WARNING, logger="contextchecker.services.checking"):
            service._warn_oversized_references(items)
        assert "too large" in caplog.text

    def test_no_warning_when_within_budget(self, caplog):
        """No warning when reference is within budget."""
        import logging
        with patch("contextchecker.services.checking.Checker"):
            service = CheckingService(
                model="checker", extractor_model=EXTRACTOR_MODEL,
                max_words=6000,
            )
        items = [{"reference": ["short ref"], KG_KEY: []}]
        with caplog.at_level(logging.WARNING, logger="contextchecker.services.checking"):
            service._warn_oversized_references(items)
        assert "too large" not in caplog.text


# ── CheckingService._validate integration ────────────────────────────────────

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
                "reference": ["some ref"],
                KG_KEY: [{"triplet": ["X", "is", "Y"], "human_label": "Entailment"}],
            }
        ]
        valid = service._validate(data)
        assert valid[0][KG_KEY][0]["subject"] == "X"
        assert "triplet" not in valid[0][KG_KEY][0]
        assert valid[0][KG_KEY][0]["human_label"] == "Entailment"

    def test_all_empty_kg_raises(self, service):
        """All items have empty _response_kg → InvalidInputError."""
        data = [
            {"reference": ["ref1"], KG_KEY: []},
            {"reference": ["ref2"], KG_KEY: []},
        ]
        with pytest.raises(InvalidInputError, match="empty"):
            service._validate(data)

    def test_missing_reference_dropped(self, service):
        """Items without 'reference' are dropped."""
        data = [
            {KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
            {"reference": ["ref"], KG_KEY: [{"subject": "X", "predicate": "Y", "object": "Z"}]},
        ]
        valid = service._validate(data)
        assert len(valid) == 1
        assert valid[0]["reference"] == ["ref"]

    def test_missing_kg_key_dropped(self, service):
        """Items without _response_kg are dropped."""
        data = [
            {"reference": ["ref"]},
            {"reference": ["ref2"], KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        valid = service._validate(data)
        assert len(valid) == 1

    def test_context_alias_works_after_canonicalize(self, service):
        """'context' is renamed to 'reference' by _canonicalize_keys before _validate."""
        data = [
            {"context": ["some ref"], KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        service._canonicalize_keys(data)
        valid = service._validate(data)
        assert len(valid) == 1
        assert valid[0]["reference"] == ["some ref"]

    def test_all_items_missing_both_keys_raises(self, service):
        """No items have required keys → InvalidInputError."""
        data = [{"foo": "bar"}, {"baz": "qux"}]
        with pytest.raises(InvalidInputError):
            service._validate(data)


# ── CheckingService._filter tests ────────────────────────────────────────────

class TestCheckingFilter:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    @pytest.fixture
    def service(self):
        with patch("contextchecker.services.checking.Checker"):
            return CheckingService(model="checker-model", extractor_model=EXTRACTOR_MODEL)

    def test_pending_items_returned(self, service):
        """Items without verdict key are pending."""
        valid = [
            {"reference": ["r"], KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        pending, skipped = service._filter(valid)
        assert len(pending) == 1
        assert skipped["already_checked"] == 0
        assert skipped["empty_claims"] == 0

    def test_already_checked_skipped(self, service):
        """Items whose triplets all carry a verdict are skipped."""
        valid = [
            {
                "reference": ["r"],
                KG_KEY: [{"subject": "A", "predicate": "B", "object": "C",
                          "checker-model_checker_verdict": "Entailment"}],
            },
            {"reference": ["r"], KG_KEY: [{"subject": "X", "predicate": "Y", "object": "Z"}]},
        ]
        pending, skipped = service._filter(valid)
        assert len(pending) == 1
        assert skipped["already_checked"] == 1
        assert skipped["empty_claims"] == 0

    def test_legacy_root_verdicts_list_ignored(self, service):
        """The retired root-level {model}_checker_verdicts list no longer
        counts as checked — such items are re-checked."""
        valid = [
            {
                "reference": ["r"],
                KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}],
                "checker-model_checker_verdicts": ["Entailment"],
            },
        ]
        pending, skipped = service._filter(valid)
        assert len(pending) == 1
        assert skipped["already_checked"] == 0

    def test_empty_kg_skipped(self, service):
        """Items with empty _response_kg (abstentions) are skipped."""
        valid = [
            {"reference": ["r"], KG_KEY: []},
            {"reference": ["r"], KG_KEY: [{"subject": "A", "predicate": "B", "object": "C"}]},
        ]
        pending, skipped = service._filter(valid)
        assert len(pending) == 1
        assert skipped["already_checked"] == 0
        assert skipped["empty_claims"] == 1

    def test_all_skipped_raises(self, service):
        """All items already checked → FilterError."""
        valid = [
            {
                "reference": ["r"],
                KG_KEY: [{"subject": "A", "predicate": "B", "object": "C",
                          "checker-model_checker_verdict": "Entailment"}],
            },
        ]
        with pytest.raises(FilterError):
            service._filter(valid)

    def test_triplet_level_verdict_already_checked(self, service):
        """Item is skipped if all triplets contain a non-None verdict."""
        valid = [
            {
                "reference": ["r"],
                KG_KEY: [
                    {"subject": "A", "predicate": "B", "object": "C", "checker-model_checker_verdict": "Entailment"},
                    {"subject": "X", "predicate": "Y", "object": "Z", "checker-model_checker_verdict": "Neutral"},
                ],
            }
        ]
        with pytest.raises(FilterError):
            service._filter(valid)

    def test_triplet_level_partial_verdict_is_pending(self, service):
        """Item is NOT skipped if at least one triplet is missing a verdict or has a None verdict."""
        valid = [
            {
                "reference": ["r"],
                KG_KEY: [
                    {"subject": "A", "predicate": "B", "object": "C", "checker-model_checker_verdict": "Entailment"},
                    {"subject": "X", "predicate": "Y", "object": "Z"}, # missing
                ],
            },
            {
                "reference": ["r"],
                KG_KEY: [
                    {"subject": "A", "predicate": "B", "object": "C", "checker-model_checker_verdict": "Entailment"},
                    {"subject": "X", "predicate": "Y", "object": "Z", "checker-model_checker_verdict": None}, # None
                ],
            }
        ]
        pending, skipped = service._filter(valid)
        assert len(pending) == 2
        assert skipped["already_checked"] == 0
        assert skipped["empty_claims"] == 0


# ── _build_payloads tests ────────────────────────────────────────────────────

class TestBuildPayloads:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    @pytest.fixture
    def service(self):
        with patch("contextchecker.services.checking.Checker"):
            return CheckingService(model="checker-model", extractor_model=EXTRACTOR_MODEL)

    def test_single_item_single_claim(self, service):
        pending = [
            {"reference": ["ref passage"], KG_KEY: [{"subject": "A", "predicate": "is", "object": "B"}]},
        ]
        payloads = service._build_payloads(pending)
        assert len(payloads) == 1
        assert payloads[0].claim == "A is B"
        assert payloads[0].reference == ["ref passage"]
        assert payloads[0].item_index == 0
        assert payloads[0].claim_index == 0

    def test_multiple_items_multiple_claims(self, service):
        pending = [
            {"reference": ["r1"], KG_KEY: [
                {"subject": "A", "predicate": "is", "object": "B"},
                {"subject": "C", "predicate": "has", "object": "D"},
            ]},
            {"reference": ["r2"], KG_KEY: [
                {"subject": "X", "predicate": "near", "object": "Y"},
            ]},
        ]
        payloads = service._build_payloads(pending)
        assert len(payloads) == 3
        assert payloads[0].item_index == 0 and payloads[0].claim_index == 0
        assert payloads[1].item_index == 0 and payloads[1].claim_index == 1
        assert payloads[2].item_index == 1 and payloads[2].claim_index == 0

    def test_reference_is_list(self, service):
        """Reference in payload is list[str], not concatenated."""
        pending = [
            {"reference": ["p1", "p2"], KG_KEY: [{"subject": "A", "predicate": "is", "object": "B"}]},
        ]
        payloads = service._build_payloads(pending)
        assert payloads[0].reference == ["p1", "p2"]


class TestClaimLevelResumption:

    @pytest.fixture(autouse=True)
    def _patch_api_key(self, monkeypatch):
        monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)

    @pytest.fixture
    def service(self):
        with patch("contextchecker.services.checking.Checker"):
            return CheckingService(model="checker-model", extractor_model=EXTRACTOR_MODEL)

    def test_single_mode_builds_payloads_only_for_unchecked(self, service):
        """_build_payloads only includes triplets without non-None verdicts."""
        pending = [
            {
                "reference": ["r1"],
                KG_KEY: [
                    {"subject": "A", "predicate": "is", "object": "B", "checker-model_checker_verdict": "Entailment"},
                    {"subject": "C", "predicate": "has", "object": "D"}, # unchecked
                    {"subject": "E", "predicate": "near", "object": "F", "checker-model_checker_verdict": None}, # None/failed -> unchecked
                ]
            }
        ]
        payloads = service._build_payloads(pending)
        assert len(payloads) == 2
        assert payloads[0].claim == "C has D"
        assert payloads[0].claim_index == 1
        assert payloads[1].claim == "E near F"
        assert payloads[1].claim_index == 2

    @pytest.mark.anyio
    async def test_joint_mode_packages_only_unchecked(self, service):
        """_execute_joint packages and checks only the unchecked claims."""
        pending = [
            {
                "reference": ["r1"],
                KG_KEY: [
                    {"subject": "A", "predicate": "is", "object": "B", "checker-model_checker_verdict": "Entailment"},
                    {"subject": "C", "predicate": "has", "object": "D"}, # unchecked (orig idx 1)
                    {"subject": "E", "predicate": "near", "object": "F", "checker-model_checker_verdict": None}, # unchecked (orig idx 2)
                ]
            }
        ]

        from unittest.mock import AsyncMock
        from contextchecker.workers.checker import ClaimVerdict, Verdict
        mock_results = [{
            2: ClaimVerdict(verdict=Verdict.CONTRADICTION, explanation="Explanation 1"),
            3: ClaimVerdict(verdict=Verdict.NEUTRAL, explanation="Explanation 2")
        }]

        mock_check = AsyncMock(return_value=mock_results)
        with patch.object(service._checker, "check_joint_batch", mock_check):
            verdicts_map = await service._execute_joint(pending)
            
            # Verify mock call parameters: expected claim_ids are 2 and 3 (1-based from orig_idx + 1)
            mock_check.assert_called_once()
            called_chunks = mock_check.call_args[0][0]
            assert len(called_chunks) == 1
            numbered_claims = called_chunks[0][0]
            assert numbered_claims == [(2, "C has D"), (3, "E near F")]

            # Verify returned verdicts map back to original indices 1 and 2
            assert verdicts_map == {
                0: {
                    1: ClaimVerdict(verdict=Verdict.CONTRADICTION, explanation="Explanation 1"),
                    2: ClaimVerdict(verdict=Verdict.NEUTRAL, explanation="Explanation 2")
                }
            }

    def test_serialize_merges_and_preserves_verdicts(self, service):
        """_serialize updates new verdicts, preserves existing ones, and fills legacy root list."""
        from contextchecker.workers.checker import ClaimVerdict, Verdict

        items = [
            {
                KG_KEY: [
                    {
                        "subject": "A", "predicate": "is", "object": "B",
                        "checker-model_checker_verdict": "Entailment",
                        "checker-model_checker_explanation": "Existing explanation"
                    },
                    {
                        "subject": "C", "predicate": "has", "object": "D"
                    }
                ]
            }
        ]

        verdicts_map = {
            0: {
                1: ClaimVerdict(verdict=Verdict.CONTRADICTION, explanation="New explanation")
            }
        }

        service._serialize(items, verdicts_map)

        # First triplet should be unchanged
        assert items[0][KG_KEY][0]["checker-model_checker_verdict"] == "Entailment"
        assert items[0][KG_KEY][0]["checker-model_checker_explanation"] == "Existing explanation"

        # Second triplet should be updated
        assert items[0][KG_KEY][1]["checker-model_checker_verdict"] == "Contradiction"
        assert items[0][KG_KEY][1]["checker-model_checker_explanation"] == "New explanation"

        # The retired root-level parallel list is never written
        assert "checker-model_checker_verdicts" not in items[0]
