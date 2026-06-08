"""Unit tests for CheckerEvaluator internals.

All tests mock CheckingService so no network calls are made.
Tests cover: GT preparation, verdict stripping, comparison logic,
metric computation, and edge cases (parse errors, empty GT, missing keys).
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from contextchecker.eval.checkereval import CheckerEvaluator, LABELS
from contextchecker.models import CheckerEvalResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_gt_item(
    triplets: list[tuple[str, str, str]],
    labels: list[str],
    reference: str | list[str] = "Some reference text.",
    gt_key: str = "claude2_response_kg",
    extra_keys: dict | None = None,
) -> dict:
    """Build a minimal GT item for testing."""
    kg = []
    for (s, p, o), label in zip(triplets, labels):
        kg.append({
            "triplet": [s, p, o],
            "human_label": label,
        })
    item = {
        gt_key: kg,
        "reference": reference,
        "question": "test question",
        "response": "test response",
    }
    if extra_keys:
        item.update(extra_keys)
    return item


@pytest.fixture
def sample_items():
    """3 valid items with known labels."""
    return [
        _make_gt_item(
            [("dogs", "are", "animals"), ("cats", "are", "pets")],
            ["Entailment", "Entailment"],
        ),
        _make_gt_item(
            [("sky", "is", "green")],
            ["Contradiction"],
        ),
        _make_gt_item(
            [("water", "might be", "wet")],
            ["Neutral"],
        ),
    ]


# ── Test _prepare_gt ─────────────────────────────────────────────────────────

class TestPrepareGt:
    """Tests for the GT validation and classification step."""

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_valid_items(self, mock_svc_cls, sample_items):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        evaluable, gt_map, skip = evaluator._prepare_gt(sample_items)

        assert len(evaluable) == 3
        assert gt_map[0] == {0: "Entailment", 1: "Entailment"}
        assert gt_map[1] == {0: "Contradiction"}
        assert gt_map[2] == {0: "Neutral"}
        assert skip["missing_gt"] == 0
        assert skip["missing_context"] == 0
        assert skip["empty_gt"] == 0
        assert skip["unlabeled_claims"] == 0

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_missing_gt_key(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [
            {"reference": "text", "some_other_key": []},
            _make_gt_item([("a", "b", "c")], ["Entailment"]),
        ]
        evaluable, gt_map, skip = evaluator._prepare_gt(items)

        assert len(evaluable) == 1
        assert skip["missing_gt"] == 1

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_missing_reference(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [
            {
                "claude2_response_kg": [
                    {"triplet": ["a", "b", "c"], "human_label": "Entailment"}
                ],
                # no reference key
            },
            _make_gt_item([("x", "y", "z")], ["Neutral"]),
        ]
        evaluable, gt_map, skip = evaluator._prepare_gt(items)

        assert len(evaluable) == 1
        assert skip["missing_context"] == 1

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_empty_gt_after_filter(self, mock_svc_cls):
        """GT key exists but no triplet has human_label."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [
            {
                "claude2_response_kg": [
                    {"triplet": ["a", "b", "c"]},  # no human_label
                ],
                "reference": "text",
            },
            _make_gt_item([("x", "y", "z")], ["Entailment"]),
        ]
        evaluable, gt_map, skip = evaluator._prepare_gt(items)

        assert len(evaluable) == 1
        assert skip["empty_gt"] == 1

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_all_items_invalid_raises(self, mock_svc_cls):
        """If no items are evaluable, raise InvalidInputError."""
        from contextchecker.exceptions import InvalidInputError

        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [{"reference": "text"}]  # no GT key

        with pytest.raises(InvalidInputError):
            evaluator._prepare_gt(items)

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_canonical_format_supported(self, mock_svc_cls):
        """Canonical format (subject/predicate/object) works too."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [{
            "claude2_response_kg": [{
                "subject": "dogs",
                "predicate": "are",
                "object": "animals",
                "human_label": "Entailment",
            }],
            "reference": "text",
        }]
        evaluable, gt_map, skip = evaluator._prepare_gt(items)

        assert len(evaluable) == 1
        assert gt_map[0] == {0: "Entailment"}

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_partial_human_labels(self, mock_svc_cls):
        """Only some triplets have human_label — index tracking must be correct."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"

        items = [{
            "claude2_response_kg": [
                {"triplet": ["a", "b", "c"]},                         # no label
                {"triplet": ["d", "e", "f"]},                         # no label
                {"triplet": ["g", "h", "i"], "human_label": "Entailment"},
            ],
            "reference": "text",
        }]
        evaluable, gt_map, skip = evaluator._prepare_gt(items)

        assert len(evaluable) == 1
        # Label must be at index 2, NOT index 0
        assert gt_map[0] == {2: "Entailment"}
        assert skip["unlabeled_claims"] == 2  # 2 triplets without human_label


# ── Test _prepare_for_service ────────────────────────────────────────────────

class TestPrepareForService:
    """Tests for GT aliasing and verdict stripping."""

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_aliases_gt_key(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"
        evaluator._explanation_key = "gpt4o_checker_explanation"
        evaluator._verdicts_list_key = "gpt4o_checker_verdicts"

        item = _make_gt_item([("a", "b", "c")], ["Entailment"])
        # All triplets labeled → gt_labels_map has sequential indices
        gt_map = {0: {0: "Entailment"}}
        evaluator._prepare_for_service([item], gt_map)

        assert "_gt_eval_response_kg" in item
        # Should contain only the labeled triplet
        assert len(item["_gt_eval_response_kg"]) == 1

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_strips_existing_verdicts(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"
        evaluator._explanation_key = "gpt4o_checker_explanation"
        evaluator._verdicts_list_key = "gpt4o_checker_verdicts"

        item = _make_gt_item([("a", "b", "c")], ["Entailment"])
        # Add pre-existing verdicts
        item["claude2_response_kg"][0]["gpt4o_checker_verdict"] = "Neutral"
        item["claude2_response_kg"][0]["gpt4o_checker_explanation"] = "old"
        item["gpt4o_checker_verdicts"] = ["Neutral"]

        gt_map = {0: {0: "Entailment"}}
        evaluator._prepare_for_service([item], gt_map)

        assert "gpt4o_checker_verdict" not in item["_gt_eval_response_kg"][0]
        assert "gpt4o_checker_explanation" not in item["_gt_eval_response_kg"][0]
        assert "gpt4o_checker_verdicts" not in item

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_filters_unlabeled_and_remaps(self, mock_svc_cls):
        """Only labeled triplets go to service, indices get remapped."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._gt_key = "claude2_response_kg"
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"
        evaluator._explanation_key = "gpt4o_checker_explanation"
        evaluator._verdicts_list_key = "gpt4o_checker_verdicts"

        # 3 triplets, only index 0 and 2 have labels
        item = _make_gt_item(
            [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")],
            ["Entailment", None, "Contradiction"],
        )
        # Manually fix: _make_gt_item sets human_label for all, but we want index 1 unlabeled
        del item["claude2_response_kg"][1]["human_label"]

        gt_map = {0: {0: "Entailment", 2: "Contradiction"}}  # sparse indices

        evaluator._prepare_for_service([item], gt_map)

        # Service only gets 2 triplets (the labeled ones)
        assert len(item["_gt_eval_response_kg"]) == 2
        assert item["_gt_eval_response_kg"][0]["triplet"] == ["a", "b", "c"]
        assert item["_gt_eval_response_kg"][1]["triplet"] == ["g", "h", "i"]

        # Indices remapped: {0: "Ent", 2: "Con"} → {0: "Ent", 1: "Con"}
        assert gt_map == {0: {0: "Entailment", 1: "Contradiction"}}


# ── Test _compare ────────────────────────────────────────────────────────────

class TestCompare:
    """Tests for the 1:1 verdict comparison step."""

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_basic_comparison(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"

        items = [{
            "_gt_eval_response_kg": [
                {"gpt4o_checker_verdict": "Entailment"},
                {"gpt4o_checker_verdict": "Contradiction"},
            ],
        }]
        gt_map = {0: {0: "Entailment", 1: "Entailment"}}

        gt_flat, pred_flat, errors = evaluator._compare(items, gt_map)

        assert gt_flat == ["Entailment", "Entailment"]
        assert pred_flat == ["Entailment", "Contradiction"]
        assert errors == 0

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_parse_errors_excluded(self, mock_svc_cls):
        """None verdicts are excluded from metrics and counted."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"

        items = [{
            "_gt_eval_response_kg": [
                {"gpt4o_checker_verdict": "Entailment"},
                {"gpt4o_checker_verdict": None},  # parse error
                {"gpt4o_checker_verdict": "Neutral"},
            ],
        }]
        gt_map = {0: {0: "Entailment", 1: "Contradiction", 2: "Neutral"}}

        gt_flat, pred_flat, errors = evaluator._compare(items, gt_map)

        assert len(gt_flat) == 2
        assert len(pred_flat) == 2
        assert errors == 1
        # Only the non-None entries
        assert gt_flat == ["Entailment", "Neutral"]
        assert pred_flat == ["Entailment", "Neutral"]

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_partial_labels_correct_alignment(self, mock_svc_cls):
        """Only some triplets have GT labels — comparison uses correct indices."""
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)
        evaluator._service_kg_key = "_gt_eval_response_kg"
        evaluator._verdict_key = "gpt4o_checker_verdict"

        # 3 triplets checked by service, but only index 2 has a human_label
        items = [{
            "_gt_eval_response_kg": [
                {"gpt4o_checker_verdict": "Neutral"},         # idx 0 — no GT
                {"gpt4o_checker_verdict": "Contradiction"},   # idx 1 — no GT
                {"gpt4o_checker_verdict": "Entailment"},      # idx 2 — has GT
            ],
        }]
        gt_map = {0: {2: "Entailment"}}  # only index 2

        gt_flat, pred_flat, errors = evaluator._compare(items, gt_map)

        # Must compare index 2's verdict, NOT index 0's
        assert gt_flat == ["Entailment"]
        assert pred_flat == ["Entailment"]
        assert errors == 0


# ── Test _build_result ───────────────────────────────────────────────────────

class TestBuildResult:
    """Tests for metric computation."""

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_perfect_accuracy(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)

        gt = ["Entailment", "Contradiction", "Neutral"]
        pred = ["Entailment", "Contradiction", "Neutral"]
        skip = {"missing_gt": 0, "missing_context": 0, "empty_gt": 0}

        result = evaluator._build_result(gt, pred, 0, 1, skip)

        assert result.accuracy == 1.0
        assert result.total_claims == 3
        assert result.parse_errors == 0
        assert result.confusion_matrix["matrix"] == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_partial_accuracy(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)

        gt = ["Entailment", "Entailment", "Contradiction", "Neutral"]
        pred = ["Entailment", "Neutral", "Contradiction", "Entailment"]
        skip = {"missing_gt": 0, "missing_context": 0, "empty_gt": 0}

        result = evaluator._build_result(gt, pred, 0, 2, skip)

        assert result.accuracy == 0.5  # 2/4
        assert result.total_claims == 4
        assert result.total_items == 2

    @patch("contextchecker.eval.checkereval.CheckingService")
    def test_report_contains_all_labels(self, mock_svc_cls):
        evaluator = CheckerEvaluator.__new__(CheckerEvaluator)

        gt = ["Entailment", "Contradiction"]
        pred = ["Entailment", "Neutral"]
        skip = {"missing_gt": 0, "missing_context": 0, "empty_gt": 0}

        result = evaluator._build_result(gt, pred, 0, 1, skip)

        assert "Entailment" in result.report
        assert "Contradiction" in result.report
        assert "Neutral" in result.report
        assert "macro avg" in result.report
