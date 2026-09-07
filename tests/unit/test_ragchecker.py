"""
Unit tests for RagCheckerPipeline: validation, chunk normalization, and the
report projection. Workers are patched out at construction — no LLM.
"""

import pytest
from unittest.mock import patch

from claimlens.exceptions import InvalidInputError
from claimlens.pipelines.ragchecker import (
    METRIC_NAMES,
    RagCheckerPipeline,
    compute_item_metrics,
    compute_overall_counts,
    compute_overall_metrics,
)


FAKE_API_KEY = "test-key-12345"
EXT = "ext-model"
CHK = "chk-model"

RESPONSE_KG = f"{EXT}_response_kg"
GT_KG = f"{EXT}_gt_answer_kg"


@pytest.fixture(autouse=True)
def _patch_api_keys(monkeypatch):
    monkeypatch.setattr("claimlens.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("claimlens.settings.CHECKER_API_KEY", FAKE_API_KEY)


@pytest.fixture
def pipeline():
    with patch("claimlens.services.extraction.Extractor"), \
         patch("claimlens.services.checking.Checker"):
        return RagCheckerPipeline(extractor_model=EXT, checker_model=CHK)


def _full_item(**overrides):
    item = {
        "query_id": "0",
        "question": "longest river?",
        "response": "The Nile is the longest river.",
        "gt_answer": "The Nile is the longest river in the world.",
        "retrieved_context": [
            {"doc_id": "000", "text": "chunk zero"},
            {"doc_id": "001", "text": "chunk one"},
        ],
    }
    item.update(overrides)
    return item


# ── Validation ───────────────────────────────────────────────────────────────

class TestValidate:

    def test_full_item_passes(self, pipeline):
        assert len(pipeline._validate([_full_item()])) == 1

    @pytest.mark.parametrize("key", ["response", "gt_answer", "retrieved_context"])
    def test_missing_key_drops_item(self, pipeline, key):
        item = _full_item()
        del item[key]
        valid = pipeline._validate([item, _full_item()])
        assert len(valid) == 1

    def test_empty_chunk_list_drops_item(self, pipeline):
        """No context = nothing to evaluate against."""
        valid = pipeline._validate([_full_item(retrieved_context=[]), _full_item()])
        assert len(valid) == 1

    def test_empty_strings_are_data_not_missing(self, pipeline):
        """docs/abstention.md §2: an empty response is a full abstention; an
        explicit "" gt_answer is the annotated no-answer. Both are kept, and
        only the latter is marked."""
        valid = pipeline._validate(
            [_full_item(response=""), _full_item(gt_answer="  "), _full_item()])
        assert len(valid) == 3
        assert valid[1]["gt_no_answer"] is True
        assert "gt_no_answer" not in valid[0]
        assert "gt_no_answer" not in valid[2]

    def test_null_gt_answer_is_missing(self, pipeline):
        valid = pipeline._validate([_full_item(gt_answer=None), _full_item()])
        assert len(valid) == 1

    def test_nothing_valid_raises(self, pipeline):
        with pytest.raises(InvalidInputError):
            pipeline._validate([{"response": "only response"}])

    def test_bare_string_chunks_normalized(self, pipeline):
        item = _full_item(retrieved_context=["plain chunk A", "plain chunk B"])
        pipeline._validate([item])
        assert item["retrieved_context"] == [
            {"doc_id": "000", "text": "plain chunk A"},
            {"doc_id": "001", "text": "plain chunk B"},
        ]

    def test_dict_chunks_preserved(self, pipeline):
        item = _full_item()
        pipeline._validate([item])
        assert item["retrieved_context"][0] == {"doc_id": "000", "text": "chunk zero"}

    def test_paper_results_envelope_unwrapped(self):
        """The original RAGChecker input format wraps items in {'results': []}."""
        from claimlens.pipelines.directions import unwrap_items
        items = [_full_item()]
        assert unwrap_items({"results": items}) == items
        assert unwrap_items(items) == items

    def test_garbage_input_raises(self):
        from claimlens.pipelines.directions import unwrap_items
        with pytest.raises(InvalidInputError):
            unwrap_items("just a string")
        with pytest.raises(InvalidInputError):
            unwrap_items({"no_results_key": []})

    def test_non_dict_items_dropped(self, pipeline):
        """String items (e.g. from a malformed file) are dropped, not crashed on."""
        valid = pipeline._validate(["results", _full_item()])
        assert len(valid) == 1


# ── Report projection ────────────────────────────────────────────────────────

def _checked_item():
    """A hand-mutated item as it looks after both extractions + 4 directions."""
    return _full_item(**{
        RESPONSE_KG: [{
            "subject": "Nile", "predicate": "is", "object": "longest river",
            f"{CHK}_answer2response_verdict": "Entailment",
            f"{CHK}_answer2response_explanation": "stated in gt",
            f"{CHK}_retrieved2response_verdicts": {0: "Entailment", 1: "Neutral"},
        }],
        GT_KG: [{
            "subject": "Nile", "predicate": "is", "object": "longest river in the world",
            f"{CHK}_response2answer_verdict": "Entailment",
            f"{CHK}_response2answer_explanation": "stated in response",
            f"{CHK}_retrieved2answer_verdicts": {0: "Neutral", 1: "Entailment"},
        }],
    })


class TestBuildReport:

    def test_structure_and_meta(self, pipeline):
        report = pipeline._build_run([_checked_item()])
        meta = report["_meta"]
        assert meta["schema_version"] == 5
        assert meta["report_type"] == "ragcheck"
        assert meta["evaluated_items"] == 1
        assert meta["dropped_items"] == 0
        # Models are arguments, not discovered facts — they live in _args now.
        assert "extractor_model" not in meta
        assert list(report) == ["_meta", "metrics", "counts", "items"]
        assert len(report["items"]) == 1

    def test_claims_projected_clean(self, pipeline):
        """Report claims are pure s/p/o — verdict keys stay out of them."""
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["response_claims"] == [
            {"subject": "Nile", "predicate": "is", "object": "longest river"}
        ]

    def test_flat_arrays_parallel_to_claims(self, pipeline):
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["answer2response"] == [
            {"verdict": "Entailment", "explanation": "stated in gt"}
        ]
        assert entry["response2answer"] == [
            {"verdict": "Entailment", "explanation": "stated in response"}
        ]

    def test_matrix_rows_in_chunk_order(self, pipeline):
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["retrieved2response"] == [
            [{"verdict": "Entailment", "explanation": None},
             {"verdict": "Neutral", "explanation": None}]
        ]
        assert entry["retrieved2answer"] == [
            [{"verdict": "Neutral", "explanation": None},
             {"verdict": "Entailment", "explanation": None}]
        ]

    def test_is_abstention_explicit_false(self, pipeline):
        """Sparse in the working data, explicit in the report view."""
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["is_abstention"] is False

    def test_abstained_item(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [], GT_KG: [],
            "is_abstention": True, "abstention_source": "heuristic",
        })
        entry = pipeline._build_run([item])["items"][0]
        assert entry["is_abstention"] is True
        assert entry["response_claims"] == []
        assert entry["answer2response"] == []

    def test_extraction_errors_surface_sparse(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [],
            f"{EXT}_extraction_error": "parse_failure",
            GT_KG: [{"subject": "a", "predicate": "b", "object": "c"}],
        })
        entry = pipeline._build_run([item])["items"][0]
        assert entry["extraction_errors"] == {"response": "parse_failure"}
        # Healthy items carry no error key at all
        clean = pipeline._build_run([_checked_item()])["items"][0]
        assert "extraction_errors" not in clean

    def test_dropped_items_counted_not_listed(self, pipeline):
        report = pipeline._build_run([_checked_item(), {"response": "no gt"}])
        assert report["_meta"]["dropped_items"] == 1
        assert len(report["items"]) == 1

    def test_query_id_falls_back_to_id(self, pipeline):
        item = _checked_item()
        del item["query_id"]
        item["id"] = "42"
        entry = pipeline._build_run([item])["items"][0]
        assert entry["query_id"] == "42"

    def test_missing_matrix_verdict_is_null_cell(self, pipeline):
        """A doc_id absent from the fold (e.g. failed chunk) → verdict None."""
        item = _checked_item()
        del item[RESPONSE_KG][0][f"{CHK}_retrieved2response_verdicts"]
        entry = pipeline._build_run([item])["items"][0]
        assert entry["retrieved2response"] == [
            [{"verdict": None, "explanation": None},
             {"verdict": None, "explanation": None}]
        ]

    def test_relevant_chunks_exposed(self, pipeline):
        """Chunks entailing >=1 gt claim are listed by doc_id."""
        entry = pipeline._build_run([_checked_item()])["items"][0]
        # gt claim's retrieved2answer: {0: "Neutral", 1: "Entailment"}
        assert entry["relevant_chunks"] == ["001"]

    def test_flat_cell_carries_error_cause(self, pipeline):
        """A null verdict in the report explains itself — sparse 'error' key."""
        item = _checked_item()
        triplet = item[RESPONSE_KG][0]
        triplet[f"{CHK}_answer2response_verdict"] = None
        triplet[f"{CHK}_answer2response_error"] = "parse_failure"
        entry = pipeline._build_run([item])["items"][0]
        assert entry["answer2response"][0]["verdict"] is None
        assert entry["answer2response"][0]["error"] == "parse_failure"
        # Healthy cells carry no error key
        assert "error" not in entry["response2answer"][0]

    def test_matrix_cell_carries_error_cause(self, pipeline):
        item = _checked_item()
        triplet = item[RESPONSE_KG][0]
        triplet[f"{CHK}_retrieved2response_verdicts"] = {0: "Entailment", 1: None}
        triplet[f"{CHK}_retrieved2response_errors"] = {1: "context_too_long"}
        entry = pipeline._build_run([item])["items"][0]
        row = entry["retrieved2response"][0]
        assert row[0] == {"verdict": "Entailment", "explanation": None}
        assert row[1] == {"verdict": None, "explanation": None,
                          "error": "context_too_long"}


# ── Metrics ──────────────────────────────────────────────────────────────────

N, E, C = "Neutral", "Entailment", "Contradiction"


def _cells(verdicts):
    return [{"verdict": v} for v in verdicts]


def _matrix(rows):
    return [[{"verdict": v} for v in row] for row in rows]


def _metrics_entry(a2r, r2a, ret2r, ret2a, n_chunks,
                   is_abstention=False, errors=None, gt_no_answer=False):
    entry = {
        "is_abstention": is_abstention,
        "gt_no_answer": gt_no_answer,
        "retrieved_context": [
            {"doc_id": f"{i:03d}", "text": "t"} for i in range(n_chunks)
        ],
        "answer2response": _cells(a2r),
        "response2answer": _cells(r2a),
        "retrieved2response": _matrix(ret2r),
        "retrieved2answer": _matrix(ret2a),
    }
    if errors:
        entry["extraction_errors"] = errors
    return entry


# The Nile example from the ORIGINAL RAGChecker implementation's output —
# the permanent reference anchor: all 11 metrics must reproduce it.
NILE_A2R = [N, E, E, E, E, E, E, E, E, N, C]
NILE_R2A = [E, E, E, E, E, E, C, N, E, E, N, N, N, N, N, E, N, N, N, N, E, E]
NILE_RET2R = [
    [N, E, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N],
    [N, N, N, N], [N, E, N, N], [N, N, N, N], [E, N, N, E], [N, N, N, E],
    [N, N, N, N],
]
NILE_RET2A = [
    [N, E, N, N], [N, N, N, N], [N, N, N, N], [E, N, N, N], [E, E, N, N],
    [E, N, N, E], [C, C, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N],
    [N, N, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N],
    [N, N, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N], [N, N, N, N],
    [N, N, N, N], [N, E, N, N],
]


class TestMetricsReference:
    """Reference anchor: the original RAGChecker output values."""

    def test_nile_example_reproduces_paper_output(self):
        entry = _metrics_entry(NILE_A2R, NILE_R2A, NILE_RET2R, NILE_RET2A, 4)
        m = compute_item_metrics(entry)
        expected = {
            "precision": 0.7272727272727273,
            "recall": 0.5,
            "claim_recall": 0.22727272727272727,
            "context_precision": 0.75,
            "faithfulness": 0.36363636363636365,
            "noise_sensitivity_in_relevant": 0.18181818181818182,
            "noise_sensitivity_in_irrelevant": 0.0,
            "f1": 0.5925925925925926,
            "hallucination": 0.09090909090909091,
            "self_knowledge": 0.5454545454545454,
            "context_utilization": 1.0,
        }
        for name, value in expected.items():
            assert m[name] == pytest.approx(value, abs=1e-3), name

    def test_faithfulness_identity_holds(self):
        entry = _metrics_entry(NILE_A2R, NILE_R2A, NILE_RET2R, NILE_RET2A, 4)
        m = compute_item_metrics(entry)
        assert m["faithfulness"] + m["hallucination"] + m["self_knowledge"] \
            == pytest.approx(1.0, abs=1e-3)


class TestMetricsNonePropagation:
    """None verdict = unknown, never 'not entailed'."""

    def test_none_a2r_leaves_precision_denominator(self):
        entry = _metrics_entry([None, E], [E], [[N], [N]], [[N]], 1)
        m = compute_item_metrics(entry)
        assert m["precision"] == 1.0  # 1/1 known, not 1/2

    def test_entailment_beats_unknown_cells(self):
        """One known Entailment in a row decides it, None cells or not."""
        entry = _metrics_entry([E], [E], [[E, None]], [[E, N]], 2)
        m = compute_item_metrics(entry)
        assert m["faithfulness"] == 1.0

    def test_unknown_row_excluded_from_family(self):
        """No Entailment + a None cell → claim unknowable → excluded."""
        entry = _metrics_entry([E, E], [E], [[E, N], [N, None]], [[E, N]], 2)
        m = compute_item_metrics(entry)
        # Only claim 1 is decidable: faithful. Claim 2 must not count as
        # self_knowledge (that would require knowing it is NOT chunk-entailed).
        assert m["faithfulness"] == 1.0
        assert m["self_knowledge"] == 0.0

    def test_unknown_claim_recall_row_excluded(self):
        entry = _metrics_entry([E], [E, E], [[N, N]], [[None, N], [E, N]], 2)
        m = compute_item_metrics(entry)
        assert m["claim_recall"] == 1.0  # 1/1 decidable, not 1/2


class TestMetricsGating:

    def test_extraction_error_nulls_everything(self):
        entry = _metrics_entry(NILE_A2R, NILE_R2A, NILE_RET2R, NILE_RET2A, 4,
                               errors={"response": "parse_failure"})
        m = compute_item_metrics(entry)
        assert all(m[name] is None for name in METRIC_NAMES)

    def test_abstention_recall_is_judged_from_the_text(self):
        """docs/abstention.md §4: a refusal entails nothing → recall 0, F1 0.
        Precision and the faithfulness family have no claims → None."""
        entry = _metrics_entry([], [N], [], [[E, N]], 2, is_abstention=True)
        m = compute_item_metrics(entry)
        assert m["claim_recall"] == 1.0
        assert m["context_precision"] == 0.5
        assert m["precision"] is None
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["faithfulness"] is None
        # relevant chunk existed and it refused: the worst case, on record
        assert m["answers_with_relevant_context"] == 0.0
        assert m["abstains_without_relevant_context"] is None

    def test_false_abstention_surfaces_through_recall(self):
        """Zero extracted claims but the text entails a GT claim: the extractor
        under-extracted. Recall says so instead of hiding it."""
        entry = _metrics_entry([], [E], [], [[E, N]], 2, is_abstention=True)
        m = compute_item_metrics(entry)
        assert m["recall"] == 1.0
        assert m["f1"] is None  # precision undefined, recall > 0

    def test_refusal_calibration_without_relevant_context(self):
        answered = _metrics_entry([E], [N], [[N, N]], [[N, N]], 2)
        refused = _metrics_entry([], [N], [], [[N, N]], 2, is_abstention=True)
        assert compute_item_metrics(answered)["abstains_without_relevant_context"] == 0.0
        assert compute_item_metrics(refused)["abstains_without_relevant_context"] == 1.0
        assert compute_item_metrics(answered)["answers_with_relevant_context"] is None


class TestOverallMetrics:

    def _results(self):
        full = _metrics_entry(NILE_A2R, NILE_R2A, NILE_RET2R, NILE_RET2A, 4)
        full["metrics"] = compute_item_metrics(full)
        # Abstention with zero retrieval evidence → justified
        abstained = _metrics_entry([], [N], [], [[N]], 1, is_abstention=True)
        abstained["metrics"] = compute_item_metrics(abstained)
        errored = _metrics_entry([], [], [], [], 1,
                                 errors={"response": "parse_failure"})
        errored["metrics"] = compute_item_metrics(errored)
        return [full, abstained, errored]

    def test_macro_average_with_support(self):
        overall = compute_overall_metrics(self._results())
        support = compute_overall_counts(self._results())["support"]
        # Only the full item contributes to precision
        assert support["precision"] == 1
        assert overall["precision"] == pytest.approx(8 / 11, abs=1e-3)
        # claim_recall averages the full item and the abstained item
        assert support["claim_recall"] == 2
        assert overall["claim_recall"] == pytest.approx(
            (0.2273 + 0.0) / 2, abs=1e-3)

    def test_abstention_rates(self):
        """The errored item leaves every denominator (charged once, in
        extraction_error_rate). GT is present on both live items, so the
        abstention is unjustified — its cause: no chunk entailed a GT
        claim. No unanswerable items → the NoAns rates are n/a, never 0."""
        overall = compute_overall_metrics(self._results())
        counts = compute_overall_counts(self._results())["abstention"]
        # the plain abstention rate is not a printed metric — its numbers are counts
        assert "abstention_rate" not in overall
        assert counts["abstained"] == 1 and counts["evaluated"] - counts["errored"] == 2
        assert overall["unjustified_abstention_rate"] == pytest.approx(1 / 2, abs=1e-3)
        assert overall["justified_abstention_rate"] is None
        assert overall["unwarranted_answer_rate"] is None
        assert counts["all_chunks_irrelevant"] == 1

    def test_unjustified_abstention_with_evidence_detected(self):
        abstained = _metrics_entry([], [N], [], [[E]], 1, is_abstention=True)
        abstained["metrics"] = compute_item_metrics(abstained)
        overall = compute_overall_metrics([abstained])
        assert overall["unjustified_abstention_rate"] == 1.0
        assert compute_overall_counts([abstained])["abstention"]["relevant_chunk_present"] == 1
        assert overall["justified_abstention_rate"] is None

    def test_no_answer_items_split_justified_and_unwarranted(self):
        """The blank-GT convention: GT "" marks the unanswerable items;
        silence there is justified, an answer there is unwarranted."""
        silent = _metrics_entry([], [], [], [], 1, is_abstention=True,
                                gt_no_answer=True)
        spoke = _metrics_entry([N, N], [], [[N], [N]], [], 1, gt_no_answer=True)
        for e in (silent, spoke):
            e["metrics"] = compute_item_metrics(e)
        overall = compute_overall_metrics([silent, spoke])
        assert overall["justified_abstention_rate"] == 0.5
        assert overall["unwarranted_answer_rate"] == 0.5
        assert overall["unjustified_abstention_rate"] is None  # no answerable items
        assert spoke["metrics"]["precision"] == 0.0            # unwarranted: precision 0
        assert spoke["metrics"]["recall"] is None              # nothing to deliver

    def test_error_rate_and_counts(self):
        overall = compute_overall_metrics(self._results())
        assert overall["extraction_error_rate"] == pytest.approx(1 / 3, abs=1e-3)
        rel = compute_overall_counts(self._results())["reliability"]["extraction"]
        assert rel == {"failed": 1, "items": 3, "by_cause": {"response": 1, "gt_answer": 0}}

    def test_checker_failure_rate(self):
        entry = _metrics_entry([E, None], [E], [[E], [None]], [[E]], 1)
        entry["metrics"] = compute_item_metrics(entry)
        overall = compute_overall_metrics([entry])
        # 6 cells total (2 + 1 + 2 + 1), 2 of them None
        assert overall["checker_failure_rate"] == pytest.approx(2 / 6, abs=1e-3)


# ── Refusal calibration: the right action given the retrieval situation ─────

class TestRefusalCalibration:
    """Two per-item binaries, 1 = the generator did the right thing in its
    situation. claim_recall > 0 → answering is right; claim_recall == 0 →
    abstaining is right; claim_recall None → situation unknown → both None."""

    def test_all_four_cells(self):
        relevant_answered = _metrics_entry([E], [E], [[E, N]], [[E, N]], 2)
        relevant_refused = _metrics_entry([], [N], [], [[E, N]], 2, is_abstention=True)
        irrelevant_answered = _metrics_entry([E], [N], [[N, N]], [[N, N]], 2)
        irrelevant_refused = _metrics_entry([], [N], [], [[N, N]], 2, is_abstention=True)

        m = compute_item_metrics(relevant_answered)
        assert (m["answers_with_relevant_context"], m["abstains_without_relevant_context"]) == (1.0, None)
        m = compute_item_metrics(relevant_refused)
        assert (m["answers_with_relevant_context"], m["abstains_without_relevant_context"]) == (0.0, None)
        m = compute_item_metrics(irrelevant_answered)
        assert (m["answers_with_relevant_context"], m["abstains_without_relevant_context"]) == (None, 0.0)
        m = compute_item_metrics(irrelevant_refused)
        assert (m["answers_with_relevant_context"], m["abstains_without_relevant_context"]) == (None, 1.0)

    def test_unknown_situation_leaves_both_none(self):
        # GT extracted to zero claims: no retrieved2answer rows at all
        no_gt_claims = _metrics_entry([E], [], [[N]], [], 1, is_abstention=True)
        # GT claims exist but every retrieval cell is unjudged
        unjudged = _metrics_entry([], [N], [], [[None, None]], 2, is_abstention=True)
        for entry in (no_gt_claims, unjudged):
            m = compute_item_metrics(entry)
            assert m["claim_recall"] is None
            assert m["answers_with_relevant_context"] is None
            assert m["abstains_without_relevant_context"] is None

    def test_blank_gt_items_never_enter_the_rows(self):
        """Judge A gates judge B: an unanswerable item has no GT claims, so
        the retrieval situation is undefined and neither row counts it —
        whether it stayed silent (justified) or spoke (unwarranted)."""
        silent = _metrics_entry([], [], [], [], 1, is_abstention=True, gt_no_answer=True)
        spoke = _metrics_entry([N], [], [[N]], [], 1, gt_no_answer=True)
        for entry in (silent, spoke):
            m = compute_item_metrics(entry)
            assert m["answers_with_relevant_context"] is None
            assert m["abstains_without_relevant_context"] is None

    def test_extraction_error_nulls_both(self):
        entry = _metrics_entry([E], [E], [[E]], [[E]], 1, errors={"gt_answer": "timeout"})
        m = compute_item_metrics(entry)
        assert m["answers_with_relevant_context"] is None
        assert m["abstains_without_relevant_context"] is None

    def test_macro_rows_reconcile_with_the_tree(self):
        """The Generator row and the ⚪ tree read the same items: the refused
        share of the 'answers when context is relevant' support equals the
        tree's 'refused with relevant chunks' count, and likewise for the
        irrelevant side."""
        entries = [
            _metrics_entry([E], [E], [[E, N]], [[E, N]], 2),                       # relevant, answered
            _metrics_entry([E], [E], [[E, N]], [[N, E]], 2),                       # relevant, answered
            _metrics_entry([], [N], [], [[E, N]], 2, is_abstention=True),          # relevant, refused
            _metrics_entry([E], [N], [[N, N]], [[N, N]], 2),                       # irrelevant, answered
            _metrics_entry([], [N], [], [[N, N]], 2, is_abstention=True),          # irrelevant, refused
            _metrics_entry([], [N], [], [[None, None]], 2, is_abstention=True),    # unknown, refused
        ]
        for e in entries:
            e["metrics"] = compute_item_metrics(e)
        overall = compute_overall_metrics(entries)
        all_counts = compute_overall_counts(entries)
        counts = all_counts["abstention"]

        assert all_counts["support"]["answers_with_relevant_context"] == 3
        assert overall["answers_with_relevant_context"] == pytest.approx(2 / 3, abs=1e-3)
        refused_with_evidence = round((1 - overall["answers_with_relevant_context"]) * 3)
        assert refused_with_evidence == counts["relevant_chunk_present"] == 1

        assert all_counts["support"]["abstains_without_relevant_context"] == 2
        assert overall["abstains_without_relevant_context"] == 0.5
        assert counts["all_chunks_irrelevant"] == 1
        assert counts["relevance_unknown"] == 1
        assert counts["unjustified"] == 3

    def test_context_utilization_is_zero_for_a_refusal_not_null(self):
        """A refusal delivers none of the GT claims the chunks carried, so
        utilization is a real 0 (the row prints without an item bracket);
        noise sensitivity needs response claims and stays null."""
        refused = _metrics_entry([], [N, N], [], [[E, N], [N, N]], 2, is_abstention=True)
        m = compute_item_metrics(refused)
        assert m["context_utilization"] == 0.0
        assert m["noise_sensitivity_in_relevant"] is None


# ── Known-before-request verdicts ────────────────────────────────────────────

class TestPrefillKnownVerdicts:

    def _items(self, pipeline):
        blank_gt = _full_item(gt_answer="", **{
            RESPONSE_KG: [{"subject": "a", "predicate": "b", "object": "c"}],
            GT_KG: [],
        })
        empty_response = _full_item(response="", **{
            RESPONSE_KG: [],
            GT_KG: [{"subject": "x", "predicate": "y", "object": "z"}],
        })
        ordinary = _checked_item()
        pipeline._validate([blank_gt, empty_response, ordinary])
        pipeline._prefill_known_verdicts([blank_gt, empty_response, ordinary])
        return blank_gt, empty_response, ordinary

    def test_blank_gt_prefills_answer2response_neutral(self, pipeline):
        blank_gt, _, _ = self._items(pipeline)
        triplet = blank_gt[RESPONSE_KG][0]
        assert triplet[f"{CHK}_answer2response_verdict"] == "Neutral"
        assert "not sent to the checker" in triplet[f"{CHK}_answer2response_explanation"]
        # precision 0, never null: the unwarranted answer is charged
        entry = pipeline._build_run([blank_gt])["items"][0]
        assert entry["metrics"]["precision"] == 0.0

    def test_empty_response_prefills_response2answer_neutral(self, pipeline):
        _, empty_response, _ = self._items(pipeline)
        triplet = empty_response[GT_KG][0]
        assert triplet[f"{CHK}_response2answer_verdict"] == "Neutral"
        # recall 0 (non-delivery), not null (unjudged): docs/abstention.md §4
        entry = pipeline._build_run([empty_response])["items"][0]
        assert entry["metrics"]["recall"] == 0.0
        assert entry["metrics"]["f1"] == 0.0

    def test_ordinary_items_untouched(self, pipeline):
        _, _, ordinary = self._items(pipeline)
        assert ordinary[RESPONSE_KG][0][f"{CHK}_answer2response_verdict"] == "Entailment"
        assert ordinary[GT_KG][0][f"{CHK}_response2answer_verdict"] == "Entailment"

    def test_refusal_text_is_not_prefilled(self, pipeline):
        """A refusal *sentence* still goes to the checker: the extractor's
        abstention call may be wrong and the text may entail a GT claim."""
        refusal = _full_item(response="I do not know.", **{
            RESPONSE_KG: [], "is_abstention": True,
            GT_KG: [{"subject": "x", "predicate": "y", "object": "z"}],
        })
        pipeline._validate([refusal])
        pipeline._prefill_known_verdicts([refusal])
        assert f"{CHK}_response2answer_verdict" not in refusal[GT_KG][0]


# ── Console output: ⚪ tree and 📊 rows ───────────────────────────────────────

class TestConsoleBlocks:

    def _report(self, pipeline):
        entries = [
            _metrics_entry([E], [E], [[E, N]], [[E, N]], 2),                      # answered
            _metrics_entry([], [N], [], [[E, N]], 2, is_abstention=True),         # refused, relevant chunks present
            _metrics_entry([], [N], [], [[N, N]], 2, is_abstention=True),         # refused, no relevant chunks
            _metrics_entry([], [N], [], [[None, None]], 2, is_abstention=True),   # refused, relevant chunks unknown
            _metrics_entry([], [], [], [], 1, is_abstention=True, gt_no_answer=True),  # justified
            _metrics_entry([N], [], [[N]], [], 1, gt_no_answer=True),             # unwarranted
        ]
        for e in entries:
            e["metrics"] = compute_item_metrics(e)
        pipeline.last_run = {
            "_meta": {"evaluated_items": len(entries)},
            "metrics": compute_overall_metrics(entries),
            "counts": compute_overall_counts(entries),
            "items": entries,
        }

    def test_abstention_tree(self, pipeline, caplog):
        import logging
        self._report(pipeline)
        with caplog.at_level(logging.INFO):
            pipeline._log_abstention()
        text = caplog.text
        assert "⚪ Abstention Behavior — 6 evaluated items" in text
        assert "1 answered" in text and "(GT present — scored)" in text
        assert "1 justified abstention " in text and "correct silence" in text
        assert "3 unjustified abstentions" in text and "charged in recall" in text
        assert "1 refused with relevant chunks" in text and "generator fault" in text
        assert "1 refused without relevant chunks" in text and "retriever fault" in text
        assert "1 refused, relevant chunks unknown" in text
        assert "1 unwarranted answer " in text and "charged in precision" in text
        assert ("→ justified abstention rate 0.500 (1 / 2 unanswerable)"
                " · unjustified abstention rate 0.750 (3 / 4 answerable)"
                " · unwarranted answer rate 0.500 (1 / 2 unanswerable)") in text
        assert "MECE violation" not in text

    def test_generator_rows(self, pipeline, caplog):
        import logging
        self._report(pipeline)
        with caplog.at_level(logging.INFO):
            pipeline._log_metrics()
        text = caplog.text
        # 2 items had a relevant chunk (1 answered, 1 refused); 1 had none
        # and refused; the unknown and blank-GT items enter neither row.
        assert "answers when context is relevant:     0.500  (2 of 6 items · higher is better)" in text
        assert "abstains when context is irrelevant:  1.000  (1 of 6 items · depends on goals)" in text
        assert "answers without" not in text and "abstains despite" not in text

    def test_variance_roster_matches_printed_rates(self):
        """Rule set 4.1: only footer rates of the ⚪ tree enter Behavior;
        the Generator rows carry the two calibration keys."""
        sections = RagCheckerPipeline._VARIANCE_SECTIONS
        assert "abstention_rate" not in sections["behavior"]
        assert sections["behavior"] == ["justified_abstention_rate",
                                        "unjustified_abstention_rate",
                                        "unwarranted_answer_rate"]
        generator = dict(sections["metrics"])["Generator"]
        assert "answers_with_relevant_context" in generator
        assert "abstains_without_relevant_context" in generator
        labels = RagCheckerPipeline._VARIANCE_LABELS
        assert labels["answers_with_relevant_context"] == "answers when context is relevant"
        assert labels["abstains_without_relevant_context"] == "abstains when context is irrelevant"


# ── Findings: the review queue over the run's items ──────────────────────────

class TestFindings:
    """Branch-keyed lists mirroring the console; every entry names its item."""

    @staticmethod
    def _cell(v, ex=None):
        return {"verdict": v, "explanation": ex}

    @staticmethod
    def _spo(s, p, o):
        return {"subject": s, "predicate": p, "object": o}

    def _items(self):
        c, spo = self._cell, self._spo
        return [
            {"query_id": "k6", "query": "composition?", "response": "rocky, two moons",
             "gt_answer": "unknown", "is_abstention": False, "gt_no_answer": False,
             "retrieved_context": [{"doc_id": "d0"}, {"doc_id": "d1"}],
             "response_claims": [spo("k", "has", "rocky surface"), spo("k", "has", "two moons"),
                                 spo("k", "is", "an exoplanet")],
             "gt_answer_claims": [spo("k", "may be", "ocean-covered")],
             "answer2response": [c("Contradiction", "gt says unknown"), c("Neutral", "no moons in gt"),
                                 c("Entailment", "both say exoplanet")],
             "response2answer": [c("Neutral", "no ocean in response")],
             "retrieved2response": [[c("Entailment"), c("Neutral")], [c("Neutral"), c("Neutral")],
                                    [c("Neutral"), c("Neutral")]],
             "retrieved2answer": [[c("Neutral"), c("Entailment")]],
             "relevant_chunks": ["d1"], "metrics": {}},
            {"query_id": "k8", "query": "star?", "response": "I cannot say.", "gt_answer": "G-type",
             "is_abstention": True, "gt_no_answer": False,
             "retrieved_context": [{"doc_id": "d0"}], "response_claims": [],
             "gt_answer_claims": [spo("k", "orbits", "G-type")],
             "answer2response": [], "response2answer": [c("Neutral")],
             "retrieved2response": [], "retrieved2answer": [[c("Entailment")]],
             "relevant_chunks": ["d0"], "metrics": {}},
            {"query_id": "k9", "query": "?", "response": "?", "gt_answer": "?", "is_abstention": False,
             "gt_no_answer": False, "retrieved_context": [], "response_claims": [], "gt_answer_claims": [],
             "answer2response": [], "response2answer": [], "retrieved2response": [], "retrieved2answer": [],
             "relevant_chunks": [], "metrics": {}, "extraction_errors": {"gt_answer": "parse_failure"}},
        ]

    def test_branches_and_attribution(self):
        f = RagCheckerPipeline._build_findings(self._items())
        assert list(f) == ["hallucination", "noise_sensitivity_in_relevant",
                           "noise_sensitivity_in_irrelevant", "self_knowledge", "recall_misses",
                           "unjustified_abstention", "unwarranted_answer", "extraction_failed",
                           "unjudged"]
        assert [e["claim"] for e in f["hallucination"]] == ["k has two moons"]
        # wrong but grounded by an irrelevant chunk → noise, chunk named
        assert f["noise_sensitivity_in_irrelevant"][0]["grounded_by"] == ["d0"]
        assert f["noise_sensitivity_in_relevant"] == []
        assert [e["claim"] for e in f["self_knowledge"]] == ["k is an exoplanet"]
        # the GT claim the response never states — and the retriever had it
        assert f["recall_misses"] == [{"query_id": "k6", "query": "composition?",
                                       "gt_claim": "k may be ocean-covered",
                                       "explanation": "no ocean in response", "retrieved_in": ["d1"]}]
        # abstained items: one entry, no per-claim recall misses
        assert f["unjustified_abstention"] == [{"query_id": "k8", "query": "star?",
                                                "response": "I cannot say.", "relevant_chunks": ["d0"]}]
        assert f["extraction_failed"] == [{"query_id": "k9", "query": "?", "side": "gt_answer",
                                           "cause": "parse_failure"}]
        assert f["unwarranted_answer"] == [] and f["unjudged"] == []

    def test_unwarranted_answer_lists_its_claims(self):
        item = self._items()[0]
        item.update(gt_no_answer=True, gt_answer_claims=[], response2answer=[], retrieved2answer=[])
        f = RagCheckerPipeline._build_findings([item])
        assert f["unwarranted_answer"][0]["claims"] == [
            "k has rocky surface", "k has two moons", "k is an exoplanet"]
        assert f["hallucination"] == [] and f["recall_misses"] == []
