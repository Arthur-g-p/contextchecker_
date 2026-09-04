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
    """_validate drops items missing a response or the GT key; an empty GT
    list is data (nothing to extract — the annotated no-answer)."""

    def test_valid_items_pass(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1

    def test_absent_gt_key_is_missing_data(self):
        """Absent = missing (dropped), same rule as ragcheck's gt_answer."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=None),
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1
        assert ev._dropped_missing_gt == 1

    def test_empty_gt_list_is_the_annotated_no_answer(self):
        """Explicit [] = nothing to extract — kept, it is what makes
        unwarranted answers detectable."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[]),
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


class TestCorpusGtGuard:
    """No item carrying the GT key at all = wrong file or wrong --gt-key.

    Presence of the KEY is what counts. An empty list is a deliberate
    abstention trap and must never trip the guard or the warning.
    """

    def test_no_item_has_gt_key_raises(self):
        ev = _evaluator()
        data = [_make_item(gt_triplets=None), _make_item(gt_triplets=None)]
        with pytest.raises(InvalidInputError, match="No ground truth found"):
            ev._validate(data)

    def test_wrong_gt_key_raises(self):
        """GT present under another key — the real-world --gt-key mistake."""
        ev = _evaluator(gt_key="some_other_key")
        data = [_make_item(gt_triplets=[_make_triplet("a", "b", "c")])]
        with pytest.raises(InvalidInputError, match="some_other_key"):
            ev._validate(data)

    def test_all_empty_gt_lists_do_NOT_raise(self):
        """An all-abstention corpus is a legitimate eval, not a broken input."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=[]), _make_item(gt_triplets=[])]
        valid = ev._validate(data)
        assert len(valid) == 2

    def test_empty_gt_not_counted_as_missing(self):
        """Abstentions carry the key (an empty list), so nothing is dropped."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=[]), _make_item(gt_triplets=[])]
        ev._validate(data)
        assert ev._dropped_missing_gt == 0

    def test_all_gt_keys_absent_is_a_wrong_file(self):
        """Every response-bearing item lacks the key → wrong file / wrong
        --gt-key, caught before any LLM call."""
        ev = _evaluator()
        with pytest.raises(InvalidInputError, match="No ground truth found"):
            ev._validate([_make_item(gt_triplets=None), _make_item(gt_triplets=None)])


class TestDataTreeRendering:
    """_log_data_pre reports each drop reason only when it happened."""

    def test_reports_missing_gt_drops(self):
        ev = _evaluator()
        ev._dropped_missing_response = 0
        ev._dropped_missing_gt = 1
        with patch("contextchecker.eval.extractoreval.logger") as mock_log:
            ev._log_data_pre(6, 1, 5)
        lines = [str(c) for c in mock_log.info.call_args_list]
        assert any("missing GT key" in l and "1" in l for l in lines)
        assert not any("missing response" in l for l in lines)

    def test_silent_when_nothing_dropped(self):
        ev = _evaluator()
        ev._dropped_missing_response = 0
        ev._dropped_missing_gt = 0
        with patch("contextchecker.eval.extractoreval.logger") as mock_log:
            ev._log_data_pre(5, 0, 5)
        lines = [str(c) for c in mock_log.info.call_args_list]
        assert not any("dropped" in l for l in lines)


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
        assert len(buckets.unwarranted_answer) == 0
        assert len(buckets.unjustified_abstention) == 0
        assert len(buckets.justified_abstention) == 0

    def test_unwarranted_answer(self):
        """Item with predictions but no GT → unwarranted answer."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=None,
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.unwarranted_answer) == 1

    def test_unjustified_abstention(self):
        """Item with GT but no predictions → unwarranted abstention."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=None,
        )]
        buckets = ev._classify(data)
        assert len(buckets.unjustified_abstention) == 1

    def test_skipped(self):
        """Item with neither GT nor predictions → justified_abstention."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=None, pred_triplets=None)]
        buckets = ev._classify(data)
        assert len(buckets.justified_abstention) == 1

    def test_empty_gt_treated_as_no_gt(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[],
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.unwarranted_answer) == 1

    def test_empty_pred_treated_as_no_pred(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=[],
        )]
        buckets = ev._classify(data)
        assert len(buckets.unjustified_abstention) == 1

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
        assert len(buckets.unwarranted_answer) == 1
        assert len(buckets.unjustified_abstention) == 1
        assert len(buckets.justified_abstention) == 1


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


# ── Test _count_pass (pure verdict counting, one pass) ───────────────────────

V_KEY = "test-checker_checker_verdict"
E_KEY = "test-checker_checker_explanation"


def _checked(*verdicts, explanation=None):
    """Build checked triplets carrying the given verdicts (None = no verdict key)."""
    out = []
    for v in verdicts:
        t = {}
        if v is not None:
            t[V_KEY] = v
        if explanation is not None:
            t[E_KEY] = explanation
        out.append(t)
    return out


def _originals(n):
    return [_make_canonical_triplet(f"s{i}", f"p{i}", f"o{i}") for i in range(n)]


class TestCountPass:
    """Three-way verdict split: Entailment / judged miss / unjudged (None)."""

    def _run(self, checked, originals, field="gt_triplet"):
        return ExtractorEvaluator._count_pass(checked, originals, V_KEY, E_KEY, field)

    def test_all_entailment(self):
        entailed, misses, unjudged = self._run(
            _checked("Entailment", "Entailment"), _originals(2))
        assert entailed == 2
        assert misses == []
        assert unjudged == []

    def test_real_verdicts_are_misses(self):
        entailed, misses, unjudged = self._run(
            _checked("Entailment", "Neutral", "Contradiction"), _originals(3))
        assert entailed == 1
        assert [m["verdict"] for m in misses] == ["Neutral", "Contradiction"]
        assert unjudged == []

    def test_miss_carries_triplet_string_and_field_name(self):
        _, misses, _ = self._run(_checked("Neutral"), _originals(1), field="pred_triplet")
        assert misses[0]["pred_triplet"] == "s0 p0 o0"

    def test_explanation_used_as_reason(self):
        _, misses, _ = self._run(
            _checked("Neutral", explanation="not in reference"), _originals(1))
        assert misses[0]["reason"] == "not in reference"

    def test_verdict_falls_back_as_reason(self):
        _, misses, _ = self._run(_checked("Neutral"), _originals(1))
        assert misses[0]["reason"] == "Neutral"

    def test_none_verdict_is_unjudged_never_a_miss(self):
        """The fix: a None verdict (checker failure — nobody judged the claim)
        is no evidence about the extractor. It leaves numerator AND
        denominator instead of being charged as a miss."""
        entailed, misses, unjudged = self._run(
            _checked("Entailment", None), _originals(2))
        assert entailed == 1
        assert misses == []
        assert len(unjudged) == 1
        assert unjudged[0]["gt_triplet"] == "s1 p1 o1"
        assert unjudged[0]["cause"] == "checker_failure"

    def test_missing_verdict_key_is_unjudged(self):
        """A triplet the checker never touched (no verdict key at all) is
        indistinguishable from an explicit None — both are unjudged."""
        entailed, misses, unjudged = self._run(_checked(None), _originals(1))
        assert (entailed, misses) == (0, [])
        assert len(unjudged) == 1

    def test_all_unjudged_zero_denominator(self):
        """Checker failed everything: nothing entailed, nothing missed —
        the pass contributes an empty (0/0) measurement, not a zero score."""
        entailed, misses, unjudged = self._run(_checked(None, None), _originals(2))
        assert (entailed, len(misses), len(unjudged)) == (0, 0, 2)

    def test_three_way_partition_is_exhaustive(self):
        """Every claim lands in exactly one bucket — the counts must always
        re-add to the number of issued claims."""
        checked = _checked("Entailment", "Neutral", None, "Contradiction", None)
        entailed, misses, unjudged = self._run(checked, _originals(5))
        assert entailed + len(misses) + len(unjudged) == 5


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
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.recall_counts["covered"] == 1
        assert result.recall_counts["denominator"] == 1
        assert result.precision_counts["supported"] == 1
        assert result.precision_counts["denominator"] == 1
        assert result.checker_failures["count"] == 0
        assert result.to_compare_items == 1
        assert result.abstentions["justified"] == 0

    def test_unjustified_abstention_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            unwarranted_answer=[],
            unjustified_abstention=[
                _make_item(gt_triplets=[_make_triplet("d", "e", "f"),
                                        _make_triplet("g", "h", "i")]),
            ],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        rc = result.recall_counts
        assert rc["covered"] == 1
        assert rc["missed"] == 0
        assert rc["unjustified_abstention_penalty"] == 2  # penalty IS in the denominator
        assert rc["denominator"] == 3
        assert result.recall == round(1 / 3, 4)
        assert result.abstentions["unjustified"] == 1

    def test_unwarranted_answer_fp_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            unwarranted_answer=[
                _make_item(gt_triplets=None,
                           pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            ],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        pc = result.precision_counts
        assert pc["supported"] == 1
        assert pc["unsupported"] == 0
        assert pc["unwarranted_answer_penalty"] == 1  # penalty IS in the denominator
        assert pc["denominator"] == 2
        assert result.precision == 0.5
        assert result.abstentions["unwarranted_answer"] == 1

    def test_zero_items(self):
        """Empty denominator → None, never 0.0: nothing judged is not a score."""
        ev = _evaluator()
        buckets = _ItemBucket(to_compare=[], unwarranted_answer=[], unjustified_abstention=[], justified_abstention=[])
        result = ev._build_result([], buckets, total_items=0)
        assert result.precision is None
        assert result.recall is None
        assert result.f1 is None

    def test_all_unjudged_yields_null_not_zero(self):
        """Checker failed everything: 0/0 on both sides must be None — a 0.0
        would give the worst possible score for zero information."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=0, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "a b c", "cause": "checker_failure"}],
            unjudged_pred=[{"pred_triplet": "a b c", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision is None
        assert result.recall is None
        assert result.f1 is None
        assert result.checker_failures == {
            "count": 2, "issued_verdicts": 2, "rate": 1.0,
            "items_affected": 1, "unjudged_gt": 1, "unjudged_pred": 1,
        }

    def test_unjudged_excluded_from_denominator(self):
        """1 covered + 1 unjudged GT claim → recall 1/1, not 1/2; the unjudged
        claim shows up in checker_failures instead."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c"),
                                        _make_triplet("d", "e", "f")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=1, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "d e f", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.recall == 1.0
        assert result.recall_counts["denominator"] == 1
        assert result.recall_counts["unjudged"] == 1
        assert result.recall_counts["total_gt_claims"] == 2
        assert result.checker_failures["count"] == 1

    def test_counts_are_exhaustive_partitions(self):
        """total = judged buckets + penalty + unjudged, denominator = total - unjudged."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet(c, c, c) for c in "abcd"],
                           pred_triplets=[_make_canonical_triplet(c, c, c) for c in "abc"]),
            ],
            unwarranted_answer=[
                _make_item(gt_triplets=None,
                           pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            ],
            unjustified_abstention=[
                _make_item(gt_triplets=[_make_triplet("d", "e", "f")]),
            ],
            justified_abstention=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=2, tp_precision=1, fp=1, fn=1,
            false_positives=[{"pred_triplet": "b b b", "verdict": "Neutral", "reason": "r"}],
            false_negatives=[{"gt_triplet": "c c c", "verdict": "Contradiction", "reason": "r"}],
            unjudged_gt=[{"gt_triplet": "d d d", "cause": "checker_failure"}],
            unjudged_pred=[{"pred_triplet": "c c c", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=3)
        rc, pc = result.recall_counts, result.precision_counts
        assert (rc["covered"] + rc["missed"] + rc["unjustified_abstention_penalty"]
                + rc["unjudged"]) == rc["total_gt_claims"] == 5
        assert rc["denominator"] == rc["total_gt_claims"] - rc["unjudged"] == 4
        assert (pc["supported"] + pc["unsupported"] + pc["unwarranted_answer_penalty"]
                + pc["unjudged"]) == pc["total_pred_claims"] == 4
        assert pc["denominator"] == pc["total_pred_claims"] - pc["unjudged"] == 3
        # Totals must equal what the extraction stats report
        assert rc["total_gt_claims"] == result.gt_stats["total_triplets"]
        assert pc["total_pred_claims"] == result.pred_stats["total_triplets"]
        # Issued verdicts: judged + unjudged, penalties never sent to the checker
        assert result.checker_failures["issued_verdicts"] == 7


class TestUnjudgedInDisagreements:
    """Checker failures are not disagreements, but affected items must stay
    identifiable in the disagreements file with their unjudged claims."""

    def test_item_with_only_unjudged_appears(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c")],
                          item_id="u1")
        buckets = _ItemBucket(
            to_compare=[item], unwarranted_answer=[],
            unjustified_abstention=[], justified_abstention=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "a b c", "cause": "checker_failure"}],
        )]
        disagreements = ev._build_disagreements(buckets, item_results)
        assert len(disagreements) == 1
        assert disagreements[0]["id"] == "u1"
        assert disagreements[0]["false_negatives"] == []
        assert disagreements[0]["unjudged"] == [
            {"gt_triplet": "a b c", "cause": "checker_failure"}]

    def test_perfect_match_still_excluded(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c")])
        buckets = _ItemBucket(
            to_compare=[item], unwarranted_answer=[],
            unjustified_abstention=[], justified_abstention=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=1, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
        )]
        assert ev._build_disagreements(buckets, item_results) == []


# ── Test extraction-error handling (tooling failures, never abstentions) ─────

ERROR_KEY = "test-model_extraction_error"


class TestExtractionErrors:
    """[] caused by a tooling failure must be excluded from ALL metrics and
    reported as an error rate — not scored as (unwarranted/correct) abstention."""

    def test_errored_items_get_own_bucket(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        normal = _make_item(gt_triplets=[_make_triplet("d", "e", "f")],
                            pred_triplets=[_make_canonical_triplet("d", "e", "f")])

        buckets = ev._classify([errored, normal])
        assert buckets.extraction_error == [errored]
        # Crucially NOT scored as unwarranted abstention despite GT + empty pred
        assert buckets.unjustified_abstention == []
        assert buckets.to_compare == [normal]

    def test_error_rate_computed_and_no_fn_penalty(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c"),
                                          _make_triplet("d", "e", "f")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        buckets = _ItemBucket(
            to_compare=[],
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[_make_item()],
            extraction_error=[errored],
        )
        result = ev._build_result([], buckets, total_items=2)
        # The errored item's 2 GT triplets added NO FN penalty
        assert result.recall_counts["unjustified_abstention_penalty"] == 0
        assert result.recall_counts["total_gt_claims"] == 0
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
            to_compare=[], unwarranted_answer=[], unjustified_abstention=[],
            justified_abstention=[], extraction_error=[errored],
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
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[],
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
            unwarranted_answer=[],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        results = [_ItemMatchResult(tp_recall=0, tp_precision=0, fp=1, fn=1,
                                     false_positives=[fp_detail],
                                     false_negatives=[])]
        disagreements = ev._build_disagreements(buckets, results)
        assert len(disagreements) == 1
        assert "error_type" not in disagreements[0]
        assert disagreements[0]["fp"] == 1

    def test_unwarranted_answer_in_disagreements(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[],
            unwarranted_answer=[_make_item(
                gt_triplets=None,
                pred_triplets=[_make_canonical_triplet("a", "b", "c")],
            )],
            unjustified_abstention=[],
            justified_abstention=[],
        )
        disagreements = ev._build_disagreements(buckets, [])
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "unwarranted_answer"
        assert disagreements[0]["false_positives"][0]["verdict"] == "no comparison made."

    def test_unjustified_abstention_in_disagreements(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[],
            unwarranted_answer=[],
            unjustified_abstention=[_make_item(
                gt_triplets=[_make_canonical_triplet("a", "b", "c"), _make_canonical_triplet("d", "e", "f")],
                pred_triplets=None,
            )],
            justified_abstention=[],
        )
        disagreements = ev._build_disagreements(buckets, [])
        assert len(disagreements) == 1
        assert disagreements[0]["error_type"] == "unjustified_abstention"
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
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            summary_doc, disagreements_doc = ev.run_sync(data)

        assert summary_doc["precision"] == 1.0
        assert summary_doc["recall"] == 1.0
        assert summary_doc["f1"] == 1.0
        assert summary_doc["_meta"]["report_type"] == "extractor_eval"
        assert disagreements_doc["total_disagreements"] == 0

    def test_unjustified_abstention_from_extraction(self):
        """Extraction produces empty results → unjustified abstention."""
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
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            summary_doc, disagreements_doc = ev.run_sync(data)

        assert summary_doc["recall"] == 0.0  # measured: 0 of 2 GT claims covered
        assert summary_doc["recall_counts"]["covered"] == 0
        assert summary_doc["recall_counts"]["unjustified_abstention_penalty"] == 2
        assert summary_doc["recall_counts"]["denominator"] == 2
        assert summary_doc["precision"] is None  # no predictions → nothing to judge
        assert summary_doc["abstentions"]["unjustified"] == 1
        assert disagreements_doc["total_disagreements"] == 1
        assert disagreements_doc["items"][0]["error_type"] == "unjustified_abstention"

    def test_unwarranted_answer_from_extraction(self):
        """GT is an explicit empty list (nothing to extract) but extraction
        produces triplets → unwarranted answer."""
        ev = _evaluator()

        data = [
            _make_item(
                gt_triplets=[],
                item_id="1",
            ),
            # Second no-answer item with no predictions: a justified
            # abstention, so the counts below are unchanged.
            _make_item(
                gt_triplets=[],
                item_id="2",
            ),
        ]

        # Mock extraction: service.run() produces triplets on the no-GT item
        async def mock_run(items):
            for item in items:
                if item["id"] == "1":
                    item[PRED_KEY] = [_make_canonical_triplet("x", "y", "z")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            summary_doc, disagreements_doc = ev.run_sync(data)

        assert summary_doc["abstentions"]["unwarranted_answer"] == 1
        assert summary_doc["precision_counts"]["unwarranted_answer_penalty"] == 1
        assert summary_doc["precision"] == 0.0  # measured: 1 penalty claim, 0 supported
        assert disagreements_doc["total_disagreements"] == 1
        assert disagreements_doc["items"][0]["error_type"] == "unwarranted_answer"
        assert disagreements_doc["items"][0]["false_positives"][0]["verdict"] == "no comparison made."
