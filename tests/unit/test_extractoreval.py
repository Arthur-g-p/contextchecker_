"""Unit tests for ExtractorEvaluator — validation, classification, LLM matching, result building."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from contextchecker.eval.extractoreval import (
    ExtractorEvaluator,
    _ItemBucket,
    _ItemMatchResult,
)
from contextchecker.exceptions import InvalidInputError
from contextchecker.models import ExtractorEvalResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

GT_KEY = "claude2_response_kg"
PRED_KEY = "test-model_response_kg"

def _make_triplet(s, p, o, human_label=None, aliases=None):
    """Build a GT triplet dict in legacy format."""
    t = {"triplet": [s, p, o]}
    if human_label:
        t["human_label"] = human_label
    if aliases:
        t["aliases"] = aliases
    return t


def _make_canonical_triplet(s, p, o):
    """Build a predicted triplet dict in canonical format."""
    return {"subject": s, "predicate": p, "object": o}


def _make_item(gt_triplets=None, pred_triplets=None, item_id="test", response="r"):
    """Build a test item."""
    item = {"id": item_id, "question": "q", "response": response, "reference": ["ref"]}
    if gt_triplets is not None:
        item[GT_KEY] = gt_triplets
    if pred_triplets is not None:
        item[PRED_KEY] = pred_triplets
    return item


def _evaluator(**kwargs):
    """Build an evaluator with mocked ExtractionService (no API key needed)."""
    defaults = dict(
        extractor_model="test-model",
        gt_key=GT_KEY,
        checker_model="test-checker",
    )
    defaults.update(kwargs)

    with patch("contextchecker.eval.extractoreval.ExtractionService"):
        return ExtractorEvaluator(**defaults)


# ── Test _validate ───────────────────────────────────────────────────────────

class TestValidate:
    """_validate only drops items missing response — GT is not required."""

    def test_valid_items_pass(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1

    def test_missing_gt_NOT_dropped(self):
        """Items with no GT survive validation — they are the wrongful_answer trap."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=None),
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 2

    def test_empty_gt_NOT_dropped(self):
        """Items with empty GT list also survive validation."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1

    def test_missing_response_dropped(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")], response=""),
        ]
        with pytest.raises(InvalidInputError):
            ev._validate(data)

    def test_all_missing_response_raises(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")], response=""),
            _make_item(gt_triplets=None, response=""),
        ]
        with pytest.raises(InvalidInputError, match="No evaluable items"):
            ev._validate(data)


# ── Test _classify (post-extraction) ─────────────────────────────────────────

class TestClassify:
    """_classify should sort items into four buckets after extraction."""

    def test_valid_item(self):
        """Item with both GT and predictions → to_compare."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.to_compare) == 1
        assert len(buckets.wrongful_answer) == 0
        assert len(buckets.wrongful_abstention) == 0
        assert len(buckets.correct_abstention) == 0

    def test_wrongful_answer(self):
        """Item with predictions but no GT → wrongful answer."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=None,
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.wrongful_answer) == 1

    def test_wrongful_abstention(self):
        """Item with GT but no predictions → wrongful abstention."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=None,
        )]
        buckets = ev._classify(data)
        assert len(buckets.wrongful_abstention) == 1

    def test_skipped(self):
        """Item with neither GT nor predictions → correct_abstention."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=None, pred_triplets=None)]
        buckets = ev._classify(data)
        assert len(buckets.correct_abstention) == 1

    def test_empty_gt_treated_as_no_gt(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[],
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.wrongful_answer) == 1

    def test_empty_pred_treated_as_no_pred(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=[],
        )]
        buckets = ev._classify(data)
        assert len(buckets.wrongful_abstention) == 1

    def test_mixed_classification(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                       pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            _make_item(gt_triplets=None, pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            _make_item(gt_triplets=[_make_triplet("d", "e", "f")], pred_triplets=None),
            _make_item(gt_triplets=None, pred_triplets=None),
        ]
        buckets = ev._classify(data)
        assert len(buckets.to_compare) == 1
        assert len(buckets.wrongful_answer) == 1
        assert len(buckets.wrongful_abstention) == 1
        assert len(buckets.correct_abstention) == 1


# ── Test _triplet_to_str ─────────────────────────────────────────────────────

class TestTripletToStr:
    def test_canonical_format(self):
        t = {"subject": "dogs", "predicate": "are", "object": "animals"}
        assert ExtractorEvaluator._triplet_to_str(t) == "dogs are animals"

    def test_legacy_format(self):
        t = {"triplet": ["dogs", "are", "animals"]}
        assert ExtractorEvaluator._triplet_to_str(t) == "dogs are animals"

    def test_canonical_preferred_over_legacy(self):
        t = {"subject": "A", "predicate": "B", "object": "C", "triplet": ["X", "Y", "Z"]}
        assert ExtractorEvaluator._triplet_to_str(t) == "A B C"


# ── Test _build_result ───────────────────────────────────────────────────────

class TestBuildResult:
    """Test the encapsulated _build_result method."""

    def test_basic_metrics(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            wrongful_answer=[],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        item_results = [_ItemMatchResult(tp=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0
        assert result.to_compare_items == 1
        assert result.correct_abstention == 0

    def test_abstention_fn_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            wrongful_answer=[],
            wrongful_abstention=[
                _make_item(gt_triplets=[_make_triplet("d", "e", "f"),
                                        _make_triplet("g", "h", "i")]),
            ],
            correct_abstention=[],
        )
        item_results = [_ItemMatchResult(tp=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        assert result.tp == 1
        assert result.fn == 2  # penalty from abstention
        assert result.abstention_errors["wrongful_abstention_fn_penalty"] == 2

    def test_wrongful_answer_fp_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            wrongful_answer=[
                _make_item(gt_triplets=None,
                           pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            ],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        item_results = [_ItemMatchResult(tp=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        assert result.tp == 1
        assert result.fp == 1  # penalty from wrongful answer
        assert result.abstention_errors["wrongful_answer_fp_penalty"] == 1

    def test_zero_items(self):
        ev = _evaluator()
        buckets = _ItemBucket(to_compare=[], wrongful_answer=[], wrongful_abstention=[], correct_abstention=[])
        result = ev._build_result([], buckets, total_items=0)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0


# ── Test _build_disagreements ────────────────────────────────────────────────

class TestBuildDisagreements:
    def test_perfect_match_excluded(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[_make_item(
                gt_triplets=[_make_triplet("a", "b", "c")],
                pred_triplets=[_make_canonical_triplet("a", "b", "c")],
            )],
            wrongful_answer=[],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        results = [_ItemMatchResult(tp=1, fp=0, fn=0, false_positives=[], false_negatives=[])]
        disagreements = ev._build_disagreements(buckets, results)
        assert len(disagreements) == 0

    def test_fp_produces_disagreement(self):
        ev = _evaluator()
        fp_detail = {"pred_triplet": {"subject": "x"}, "verdict": "no comparison made.", "reason": "No match"}
        buckets = _ItemBucket(
            to_compare=[_make_item(
                gt_triplets=[_make_triplet("a", "b", "c")],
                pred_triplets=[_make_canonical_triplet("x", "y", "z")],
            )],
            wrongful_answer=[],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        results = [_ItemMatchResult(tp=0, fp=1, fn=1,
                                     false_positives=[fp_detail],
                                     false_negatives=[])]
        disagreements = ev._build_disagreements(buckets, results)
        assert len(disagreements) == 1
        assert "error_type" not in disagreements[0]
        assert disagreements[0]["fp"] == 1

    def test_wrongful_answer_in_disagreements(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[],
            wrongful_answer=[_make_item(
                gt_triplets=None,
                pred_triplets=[_make_canonical_triplet("a", "b", "c")],
            )],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        disagreements = ev._build_disagreements(buckets, [])
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "wrongful_answer"
        assert disagreements[0]["false_positives"][0]["verdict"] == "no comparison made."

    def test_wrongful_abstention_in_disagreements(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[],
            wrongful_answer=[],
            wrongful_abstention=[_make_item(
                gt_triplets=[_make_triplet("a", "b", "c"), _make_triplet("d", "e", "f")],
                pred_triplets=None,
            )],
            correct_abstention=[],
        )
        disagreements = ev._build_disagreements(buckets, [])
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "wrongful_abstention"
        assert disagreements[0]["fn"] == 2
        assert disagreements[0]["false_negatives"][0]["verdict"] == "no comparison made."


# ── Integration-lite tests ───────────────────────────────────────────────────

class TestEvaluateIntegration:
    """Test evaluate() flow with mocked extraction service."""

    def test_perfect_score_with_mocked_extraction(self):
        """Extraction produces exact matches → P=R=F1=1.0."""
        ev = _evaluator()
        
        async def mock_match_all_llm(items):
            return [_ItemMatchResult(tp=1, fp=0, fn=0, false_positives=[], false_negatives=[])]
        ev._match_all_llm = AsyncMock(side_effect=mock_match_all_llm)

        data = [
            _make_item(
                gt_triplets=[_make_triplet("a", "b", "c")],
                item_id="1",
            ),
        ]

        # Mock extraction: service.run() adds pred triplets to items
        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = [_make_canonical_triplet("a", "b", "c")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_data_post"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            result, disagreements = ev.run_sync(data)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert len(disagreements) == 0

    def test_wrongful_abstention_from_extraction(self):
        """Extraction produces empty results → wrongful abstention."""
        ev = _evaluator()

        data = [
            _make_item(
                gt_triplets=[_make_triplet("a", "b", "c"), _make_triplet("d", "e", "f")],
                item_id="1",
            ),
        ]

        # Mock extraction: service.run() produces nothing (abstention)
        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = []

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_data_post"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            result, disagreements = ev.run_sync(data)

        assert result.tp == 0
        assert result.fn == 2
        assert result.abstention_errors["wrongful_abstention"] == 1
        assert result.abstention_errors["wrongful_abstention_fn_penalty"] == 2
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "wrongful_abstention"

    def test_wrongful_answer_from_extraction(self):
        """Item has no GT but extraction produces triplets → wrongful answer."""
        ev = _evaluator()

        data = [
            _make_item(
                gt_triplets=None,
                item_id="1",
            ),
        ]

        # Mock extraction: service.run() produces triplets on a no-GT item
        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = [_make_canonical_triplet("x", "y", "z")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_data_post"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            result, disagreements = ev.run_sync(data)

        assert result.abstention_errors["wrongful_answer"] == 1
        assert result.abstention_errors["wrongful_answer_fp_penalty"] == 1
        assert result.fp == 1
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "wrongful_answer"
        assert disagreements[0]["false_positives"][0]["verdict"] == "no comparison made."
