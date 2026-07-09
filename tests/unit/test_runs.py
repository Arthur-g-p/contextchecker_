"""
Unit tests for repeated-run variance: the stats aggregation helpers and the
pipeline-internal --runs mode. No LLM — _run_once is stubbed.
"""

import pytest
from unittest.mock import patch

from contextchecker.utils import aggregate_values, build_variance
from contextchecker.pipelines.ragchecker import RagCheckerPipeline


FAKE_API_KEY = "test-key-12345"
EXT = "ext-model"
CHK = "chk-model"


@pytest.fixture(autouse=True)
def _patch_api_keys(monkeypatch):
    monkeypatch.setattr("contextchecker.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)


def _pipeline(runs):
    with patch("contextchecker.services.extraction.Extractor"), \
         patch("contextchecker.services.checking.Checker"):
        return RagCheckerPipeline(extractor_model=EXT, checker_model=CHK,
                                  runs=runs)


# ── Aggregation helpers (stats.py) ───────────────────────────────────────────

class TestAggregation:

    def test_aggregate_values(self):
        agg = aggregate_values([0.5, 0.7, 0.6])
        assert agg["min"] == 0.5
        assert agg["max"] == 0.7
        assert agg["values"] == [0.5, 0.7, 0.6]
        assert agg["std"] == pytest.approx(0.1, abs=1e-4)

    def test_nulls_are_skipped_not_zeroed(self):
        agg = aggregate_values([0.5, None, 0.7])
        assert agg["values"] == [0.5, 0.7]

    def test_all_null_is_none(self):
        assert aggregate_values([None, None]) is None

    def test_single_value_std_zero(self):
        assert aggregate_values([0.5])["std"] == 0.0

    def test_build_variance_means_and_nested_skipped(self):
        means, variance = build_variance([
            {"precision": 0.5, "support": {"precision": 2}},
            {"precision": 0.7, "support": {"precision": 2}},
        ])
        assert means == {"precision": 0.6}
        assert variance["precision"]["min"] == 0.5
        # dict-valued keys never enter the aggregate
        assert "support" not in means


# ── Pipeline-internal runs mode ──────────────────────────────────────────────

class TestPipelineRuns:

    def test_children_silent_in_variance_mode(self):
        pipeline = _pipeline(runs=3)
        assert pipeline._extract_response.verbosity == "silent"
        assert pipeline._directions[0][1].verbosity == "silent"

    def test_children_compact_for_single_run(self):
        pipeline = _pipeline(runs=1)
        assert pipeline._extract_response.verbosity == "compact"

    def test_multi_run_document_shape(self, monkeypatch):
        pipeline = _pipeline(runs=3)
        seen = []

        async def fake_run_once(data, announce=True, report=True):
            seen.append(data)
            data[0]["mutated"] = True
            pipeline.last_report = {
                "_meta": {"schema_version": 2, "extractor_model": EXT},
                "overall_metrics": {"precision": 0.5 + 0.1 * len(seen)},
                "results": [{"query_id": "0"}],
            }
            return data

        monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
        source = [{"response": "x"}]
        pipeline.run_sync(source)
        doc = pipeline.last_report

        # A report is a report is a report: runs holds N normal documents
        assert doc["_meta"]["runs"] == 3
        assert "duration_seconds" in doc["_meta"]
        assert "run" not in doc["_meta"]           # outer meta has no run number
        assert len(doc["runs"]) == 3
        assert doc["runs"][0]["_meta"]["run"] == 1
        assert doc["runs"][2]["_meta"]["run"] == 3
        assert all("duration_seconds" in r["_meta"] for r in doc["runs"])
        assert doc["runs"][0]["results"] == [{"query_id": "0"}]
        # No top-level results in a multi-run file — runs IS the content
        assert "results" not in doc

        # Aggregates: flat means + variance sibling (0.6, 0.7, 0.8)
        assert doc["overall_metrics"]["precision"] == pytest.approx(0.7, abs=1e-4)
        assert doc["variance"]["precision"]["min"] == 0.6
        assert doc["variance"]["precision"]["max"] == 0.8

    def test_run_one_in_place_then_deep_copies(self, monkeypatch):
        """Run 1 keeps the run() contract (mutates the input); runs 2..N
        operate on isolated deep copies."""
        pipeline = _pipeline(runs=3)
        seen = []
        mutated_at_entry = []

        async def fake_run_once(data, announce=True, report=True):
            seen.append(data)
            mutated_at_entry.append("mutated" in data[0])
            data[0]["mutated"] = True
            pipeline.last_report = {
                "_meta": {}, "overall_metrics": {}, "results": [],
            }
            return data

        monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
        source = [{"response": "x"}]
        pipeline.run_sync(source)

        assert seen[0] is source                    # run 1: in place
        assert seen[1] is not source
        assert seen[2] is not source
        # Pristine snapshot: runs 2..N must NOT see run 1's mutations at
        # entry — otherwise skip logic no-ops them and fakes zero variance.
        assert mutated_at_entry == [False, False, False]

    def test_single_run_keeps_normal_report(self, monkeypatch):
        pipeline = _pipeline(runs=1)

        async def fake_run_once(data, announce=True, report=True):
            pipeline.last_report = {"_meta": {}, "overall_metrics": {},
                                    "results": [{"query_id": "0"}]}
            return data

        monkeypatch.setattr(pipeline, "_run_once", fake_run_once)
        pipeline.run_sync([{"response": "x"}])
        # No variance/runs keys with the default — byte-compatible document
        assert "variance" not in pipeline.last_report
        assert "runs" not in pipeline.last_report
        assert pipeline.last_report["results"] == [{"query_id": "0"}]