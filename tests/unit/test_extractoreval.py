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


# ── Test pred_key / gt_key collision guard ───────────────────────────────────

class TestPredKeyCollisionGuard:
    """The derived predicted-key (f'{extractor_model}_response_kg') must never
    equal the GT key — otherwise extraction targets the GT slot and the
    evaluator matches ground truth against itself, reporting a false perfect score."""

    def test_collision_via_extractor_model_raises(self):
        """extractor_model='claude2' derives 'claude2_response_kg', colliding with the default gt_key."""
        with pytest.raises(InvalidInputError, match="collides"):
            _evaluator(extractor_model="claude2", gt_key="claude2_response_kg")

    def test_collision_via_gt_key_raises(self):
        """A gt_key set to match the derived predicted-key also collides."""
        with pytest.raises(InvalidInputError, match="collides"):
            _evaluator(extractor_model="test-model", gt_key="test-model_response_kg")

    def test_no_collision_constructs_ok(self):
        """Distinct extractor_model and gt_key construct without raising."""
        ev = _evaluator(extractor_model="gemini-3.1", gt_key="claude2_response_kg")
        assert ev._pred_key == "gemini-3.1_response_kg"
        assert ev._pred_key != ev._gt_key


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

    def test_legacy_dict_raises(self):
        # Triplets are canonicalized at ingestion; an un-canonicalized legacy
        # dict reaching this method is a contract violation, not a fallback.
        with pytest.raises(KeyError):
            ExtractorEvaluator._triplet_to_str({"triplet": ["dogs", "are", "animals"]})


# ── Test _measure_duplicates (orthogonal, read-only) ──────────────────────────

class TestMeasureDuplicates:
    def test_none_when_no_predictions(self):
        ev = _evaluator()
        valid = [_make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")])]
        assert ev._measure_duplicates(valid) is None

    def test_counts_and_lists_duplicates(self):
        ev = _evaluator()
        valid = [_make_item(item_id="x", pred_triplets=[
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("d", "e", "f"),
        ])]
        d = ev._measure_duplicates(valid)
        assert d["predicted_claims"] == 3
        assert d["duplicate_claims"] == 1
        assert d["unique_claims"] == 2
        assert d["duplicate_rate"] == round(1 / 3, 4)
        assert d["items"][0]["id"] == "x"
        assert d["items"][0]["duplicates"] == ["a b c"]

    def test_no_duplicates_gives_empty_listing(self):
        ev = _evaluator()
        valid = [_make_item(pred_triplets=[
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("d", "e", "f"),
        ])]
        d = ev._measure_duplicates(valid)
        assert d["duplicate_claims"] == 0
        assert d["items"] == []

    def test_read_only_never_mutates_predictions(self):
        ev = _evaluator()
        preds = [_make_canonical_triplet("a", "b", "c"), _make_canonical_triplet("a", "b", "c")]
        valid = [_make_item(pred_triplets=preds)]
        ev._measure_duplicates(valid)
        assert len(preds) == 2  # the eval never deduplicates its data


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
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.tp_recall == 1
        assert result.tp_precision == 1
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
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        assert result.tp_recall == 1
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
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        assert result.tp_precision == 1
        assert result.fp == 1  # penalty from wrongful answer
        assert result.abstention_errors["wrongful_answer_fp_penalty"] == 1

    def test_zero_items(self):
        ev = _evaluator()
        buckets = _ItemBucket(to_compare=[], wrongful_answer=[], wrongful_abstention=[], correct_abstention=[])
        result = ev._build_result([], buckets, total_items=0)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0


# ── Test extraction-error handling (tooling failures, never abstentions) ─────

ERROR_KEY = "test-model_extraction_error"


class TestExtractionErrors:
    """[] caused by a tooling failure must be excluded from ALL metrics and
    reported as an error rate — not scored as (wrongful/correct) abstention."""

    def test_errored_items_get_own_bucket(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        normal = _make_item(gt_triplets=[_make_triplet("d", "e", "f")],
                            pred_triplets=[_make_canonical_triplet("d", "e", "f")])

        buckets = ev._classify([errored, normal])
        assert buckets.extraction_error == [errored]
        # Crucially NOT scored as wrongful abstention despite GT + empty pred
        assert buckets.wrongful_abstention == []
        assert buckets.to_compare == [normal]

    def test_error_rate_computed_and_no_fn_penalty(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c"),
                                          _make_triplet("d", "e", "f")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        buckets = _ItemBucket(
            to_compare=[],
            wrongful_answer=[],
            wrongful_abstention=[],
            correct_abstention=[_make_item()],
            extraction_error=[errored],
        )
        result = ev._build_result([], buckets, total_items=2)
        # The errored item's 2 GT triplets added NO FN penalty
        assert result.fn == 0
        assert result.extraction_errors == {
            "count": 1,
            "rate": 0.5,
            "by_cause": {"parse_failure": 1},
        }

    def test_errored_items_listed_in_disagreements(self):
        ev = _evaluator()
        errored = _make_item(item_id="broken-1")
        errored[ERROR_KEY] = "context_too_long"
        buckets = _ItemBucket(
            to_compare=[], wrongful_answer=[], wrongful_abstention=[],
            correct_abstention=[], extraction_error=[errored],
        )
        disagreements = ev._build_disagreements(buckets, [])
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "extraction_error"
        assert disagreements[0]["cause"] == "context_too_long"
        assert disagreements[0]["id"] == "broken-1"


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
        results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]
        disagreements = ev._build_disagreements(buckets, results)
        assert len(disagreements) == 0

    def test_fp_produces_disagreement(self):
        ev = _evaluator()
        fp_detail = {"pred_triplet": {"subject": "x"}, "verdict": "no comparison made.", "reason": "No match"}
        buckets = _ItemBucket(
            to_compare=[_make_item(
                gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                pred_triplets=[_make_canonical_triplet("x", "y", "z")],
            )],
            wrongful_answer=[],
            wrongful_abstention=[],
            correct_abstention=[],
        )
        results = [_ItemMatchResult(tp_recall=0, tp_precision=0, fp=1, fn=1,
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
                gt_triplets=[_make_canonical_triplet("a", "b", "c"), _make_canonical_triplet("d", "e", "f")],
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
            return [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]
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

        assert result.tp_recall == 0
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
