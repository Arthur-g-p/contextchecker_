"""
Unit tests for RagCheckerPipeline: validation, chunk normalization, and the
report projection. Workers are patched out at construction — no LLM.
"""

import pytest
from unittest.mock import patch

from contextchecker.exceptions import InvalidInputError
from contextchecker.pipelines.ragchecker import (
    METRIC_NAMES,
    RagCheckerPipeline,
    compute_item_metrics,
    compute_overall_metrics,
)


FAKE_API_KEY = "test-key-12345"
EXT = "ext-model"
CHK = "chk-model"

RESPONSE_KG = f"{EXT}_response_kg"
GT_KG = f"{EXT}_gt_answer_kg"


@pytest.fixture(autouse=True)
def _patch_api_keys(monkeypatch):
    monkeypatch.setattr("contextchecker.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)


@pytest.fixture
def pipeline():
    with patch("contextchecker.services.extraction.Extractor"), \
         patch("contextchecker.services.checking.Checker"):
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

    @pytest.mark.parametrize("key,empty", [
        ("response", ""),
        ("gt_answer", ""),
        ("retrieved_context", []),
    ])
    def test_empty_value_drops_item(self, pipeline, key, empty):
        """Falsy counts as missing — empty GT or chunk list means garbage metrics."""
        valid = pipeline._validate([_full_item(**{key: empty}), _full_item()])
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
        from contextchecker.pipelines.directions import unwrap_items
        items = [_full_item()]
        assert unwrap_items({"results": items}) == items
        assert unwrap_items(items) == items

    def test_garbage_input_raises(self):
        from contextchecker.pipelines.directions import unwrap_items
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
            f"{CHK}_retrieved2response_verdicts": {"000": "Entailment", "001": "Neutral"},
        }],
        GT_KG: [{
            "subject": "Nile", "predicate": "is", "object": "longest river in the world",
            f"{CHK}_response2answer_verdict": "Entailment",
            f"{CHK}_response2answer_explanation": "stated in response",
            f"{CHK}_retrieved2answer_verdicts": {"000": "Neutral", "001": "Entailment"},
        }],
    })


class TestBuildReport:

    def test_structure_and_meta(self, pipeline):
        report = pipeline.build_report([_checked_item()])
        assert report["_meta"]["schema_version"] == 2
        assert report["_meta"]["extractor_model"] == EXT
        assert report["_meta"]["evaluated_items"] == 1
        assert report["_meta"]["dropped_items"] == 0
        assert "overall_metrics" in report
        assert len(report["results"]) == 1

    def test_claims_projected_clean(self, pipeline):
        """Report claims are pure s/p/o — verdict keys stay out of them."""
        entry = pipeline.build_report([_checked_item()])["results"][0]
        assert entry["response_claims"] == [
            {"subject": "Nile", "predicate": "is", "object": "longest river"}
        ]

    def test_flat_arrays_parallel_to_claims(self, pipeline):
        entry = pipeline.build_report([_checked_item()])["results"][0]
        assert entry["answer2response"] == [
            {"verdict": "Entailment", "explanation": "stated in gt"}
        ]
        assert entry["response2answer"] == [
            {"verdict": "Entailment", "explanation": "stated in response"}
        ]

    def test_matrix_rows_in_chunk_order(self, pipeline):
        entry = pipeline.build_report([_checked_item()])["results"][0]
        assert entry["retrieved2response"] == [
            [{"verdict": "Entailment"}, {"verdict": "Neutral"}]
        ]
        assert entry["retrieved2answer"] == [
            [{"verdict": "Neutral"}, {"verdict": "Entailment"}]
        ]

    def test_is_abstention_explicit_false(self, pipeline):
        """Sparse in the working data, explicit in the report view."""
        entry = pipeline.build_report([_checked_item()])["results"][0]
        assert entry["is_abstention"] is False

    def test_abstained_item(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [], GT_KG: [],
            "is_abstention": True, "abstention_source": "heuristic",
        })
        entry = pipeline.build_report([item])["results"][0]
        assert entry["is_abstention"] is True
        assert entry["response_claims"] == []
        assert entry["answer2response"] == []

    def test_extraction_errors_surface_sparse(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [],
            f"{EXT}_extraction_error": "parse_failure",
            GT_KG: [{"subject": "a", "predicate": "b", "object": "c"}],
        })
        entry = pipeline.build_report([item])["results"][0]
        assert entry["extraction_errors"] == {"response": "parse_failure"}
        # Healthy items carry no error key at all
        clean = pipeline.build_report([_checked_item()])["results"][0]
        assert "extraction_errors" not in clean

    def test_dropped_items_counted_not_listed(self, pipeline):
        report = pipeline.build_report([_checked_item(), {"response": "no gt"}])
        assert report["_meta"]["dropped_items"] == 1
        assert len(report["results"]) == 1

    def test_query_id_falls_back_to_id(self, pipeline):
        item = _checked_item()
        del item["query_id"]
        item["id"] = "42"
        entry = pipeline.build_report([item])["results"][0]
        assert entry["query_id"] == "42"

    def test_missing_matrix_verdict_is_null_cell(self, pipeline):
        """A doc_id absent from the fold (e.g. failed chunk) → verdict None."""
        item = _checked_item()
        del item[RESPONSE_KG][0][f"{CHK}_retrieved2response_verdicts"]
        entry = pipeline.build_report([item])["results"][0]
        assert entry["retrieved2response"] == [
            [{"verdict": None}, {"verdict": None}]
        ]

    def test_relevant_chunks_exposed(self, pipeline):
        """Chunks entailing >=1 gt claim are listed by doc_id."""
        entry = pipeline.build_report([_checked_item()])["results"][0]
        # gt claim's retrieved2answer: {"000": "Neutral", "001": "Entailment"}
        assert entry["relevant_chunks"] == ["001"]

    def test_flat_cell_carries_error_cause(self, pipeline):
        """A null verdict in the report explains itself — sparse 'error' key."""
        item = _checked_item()
        triplet = item[RESPONSE_KG][0]
        triplet[f"{CHK}_answer2response_verdict"] = None
        triplet[f"{CHK}_answer2response_error"] = "parse_failure"
        entry = pipeline.build_report([item])["results"][0]
        assert entry["answer2response"][0]["verdict"] is None
        assert entry["answer2response"][0]["error"] == "parse_failure"
        # Healthy cells carry no error key
        assert "error" not in entry["response2answer"][0]

    def test_matrix_cell_carries_error_cause(self, pipeline):
        item = _checked_item()
        triplet = item[RESPONSE_KG][0]
        triplet[f"{CHK}_retrieved2response_verdicts"] = {"000": "Entailment", "001": None}
        triplet[f"{CHK}_retrieved2response_errors"] = {"001": "context_too_long"}
        entry = pipeline.build_report([item])["results"][0]
        row = entry["retrieved2response"][0]
        assert row[0] == {"verdict": "Entailment"}
        assert row[1] == {"verdict": None, "error": "context_too_long"}


# ── Metrics ──────────────────────────────────────────────────────────────────

N, E, C = "Neutral", "Entailment", "Contradiction"


def _cells(verdicts):
    return [{"verdict": v} for v in verdicts]


def _matrix(rows):
    return [[{"verdict": v} for v in row] for row in rows]


def _metrics_entry(a2r, r2a, ret2r, ret2a, n_chunks,
                   is_abstention=False, errors=None):
    entry = {
        "is_abstention": is_abstention,
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

    def test_abstention_keeps_retrieval_metrics_only(self):
        entry = _metrics_entry([], [E], [], [[E, N]], 2, is_abstention=True)
        m = compute_item_metrics(entry)
        assert m["claim_recall"] == 1.0
        assert m["context_precision"] == 0.5
        assert m["precision"] is None
        assert m["recall"] is None
        assert m["faithfulness"] is None


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
        # Only the full item contributes to precision
        assert overall["support"]["precision"] == 1
        assert overall["precision"] == pytest.approx(8 / 11, abs=1e-3)
        # claim_recall averages the full item and the abstained item
        assert overall["support"]["claim_recall"] == 2
        assert overall["claim_recall"] == pytest.approx(
            (0.2273 + 0.0) / 2, abs=1e-3)

    def test_abstention_rates(self):
        overall = compute_overall_metrics(self._results())
        assert overall["abstention_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert overall["justified_abstention_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert overall["unjustified_abstention_rate"] == 0.0

    def test_unjustified_abstention_detected(self):
        abstained = _metrics_entry([], [E], [], [[E]], 1, is_abstention=True)
        abstained["metrics"] = compute_item_metrics(abstained)
        overall = compute_overall_metrics([abstained])
        assert overall["unjustified_abstention_rate"] == 1.0
        assert overall["justified_abstention_rate"] == 0.0

    def test_error_rate_and_counts(self):
        overall = compute_overall_metrics(self._results())
        assert overall["extraction_error_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert overall["extraction_errors"] == {"response": 1, "gt_answer": 0}

    def test_judge_failure_rate(self):
        entry = _metrics_entry([E, None], [E], [[E], [None]], [[E]], 1)
        entry["metrics"] = compute_item_metrics(entry)
        overall = compute_overall_metrics([entry])
        # 6 cells total (2 + 1 + 2 + 1), 2 of them None
        assert overall["judge_failure_rate"] == pytest.approx(2 / 6, abs=1e-3)
