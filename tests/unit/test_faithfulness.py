"""
Unit tests for FaithfulnessPipeline: validation, report projection, the
faithfulness metric, and the check_faithfulness facade. Workers patched out
at construction — no LLM.
"""

import pytest
from unittest.mock import patch

from claimlens.exceptions import InvalidInputError
from claimlens.pipelines.faithfulness import (
    FaithfulnessPipeline,
    check_faithfulness,
)


FAKE_API_KEY = "test-key-12345"
EXT = "ext-model"
CHK = "chk-model"

RESPONSE_KG = f"{EXT}_response_kg"
NAMESPACE = f"{CHK}_retrieved2response"


@pytest.fixture(autouse=True)
def _patch_api_keys(monkeypatch):
    monkeypatch.setattr("claimlens.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("claimlens.settings.CHECKER_API_KEY", FAKE_API_KEY)


@pytest.fixture
def pipeline():
    with patch("claimlens.services.extraction.Extractor"), \
         patch("claimlens.services.checking.Checker"):
        return FaithfulnessPipeline(extractor_model=EXT, checker_model=CHK)


def _full_item(**overrides):
    item = {
        "query_id": "0",
        "response": "The Nile is the longest river.",
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

    def test_no_gt_answer_required(self, pipeline):
        """The whole point: works without ground truth."""
        item = _full_item()
        assert "gt_answer" not in item
        assert len(pipeline._validate([item])) == 1

    def test_empty_chunks_drop_item(self, pipeline):
        valid = pipeline._validate([_full_item(retrieved_context=[]), _full_item()])
        assert len(valid) == 1

    def test_empty_response_is_an_abstention_not_missing(self, pipeline):
        """docs/abstention.md §1: "" is a full abstention — kept, flagged later."""
        valid = pipeline._validate([_full_item(response=""), _full_item()])
        assert len(valid) == 2

    def test_null_response_is_missing(self, pipeline):
        valid = pipeline._validate([_full_item(response=None), _full_item()])
        assert len(valid) == 1

    def test_nothing_valid_raises(self, pipeline):
        with pytest.raises(InvalidInputError):
            pipeline._validate([{"response": "no chunks"}])

    def test_bare_string_chunks_normalized(self, pipeline):
        item = _full_item(retrieved_context=["plain chunk"])
        pipeline._validate([item])
        assert item["retrieved_context"] == [{"doc_id": "000", "text": "plain chunk"}]


# ── Report projection + metric ───────────────────────────────────────────────

def _checked_item():
    """Item as it looks after extraction + the retrieved2response direction."""
    return _full_item(**{
        RESPONSE_KG: [
            {"subject": "Nile", "predicate": "is", "object": "longest river",
             f"{NAMESPACE}_verdicts": {0: "Entailment", 1: "Neutral"}},
            {"subject": "Nile", "predicate": "has", "object": "5M inhabitants",
             f"{NAMESPACE}_verdicts": {0: "Neutral", 1: "Neutral"}},
        ],
    })


class TestBuildRun:

    def test_entry_shape_and_metric(self, pipeline):
        report = pipeline._build_run([_checked_item()])
        assert report["_meta"]["report_type"] == "faithcheck"
        entry = report["items"][0]
        assert entry["response_claims"] == [
            {"subject": "Nile", "predicate": "is", "object": "longest river"},
            {"subject": "Nile", "predicate": "has", "object": "5M inhabitants"},
        ]
        assert entry["retrieved2response"] == [
            [{"verdict": "Entailment", "explanation": None},
             {"verdict": "Neutral", "explanation": None}],
            [{"verdict": "Neutral", "explanation": None},
             {"verdict": "Neutral", "explanation": None}],
        ]
        # 1 of 2 claims grounded
        assert entry["metrics"]["faithfulness"] == 0.5

    def test_claim_support_attribution(self, pipeline):
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["claim_support"] == [["000"], []]

    def test_abstention_is_null(self, pipeline):
        item = _full_item(**{RESPONSE_KG: [], "is_abstention": True})
        entry = pipeline._build_run([item])["items"][0]
        assert entry["is_abstention"] is True
        assert entry["metrics"]["faithfulness"] is None

    def test_extraction_error_is_null(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [],
            f"{EXT}_extraction_error": "parse_failure",
        })
        entry = pipeline._build_run([item])["items"][0]
        assert entry["extraction_errors"] == {"response": "parse_failure"}
        assert entry["metrics"]["faithfulness"] is None

    def test_unknown_cells_propagate(self, pipeline):
        """No Entailment + a None cell → claim unknowable → excluded."""
        item = _full_item(**{
            RESPONSE_KG: [
                {"subject": "a", "predicate": "b", "object": "c",
                 f"{NAMESPACE}_verdicts": {0: "Entailment", 1: None}},
                {"subject": "d", "predicate": "e", "object": "f",
                 f"{NAMESPACE}_verdicts": {0: "Neutral", 1: None}},
            ],
        })
        entry = pipeline._build_run([item])["items"][0]
        # Claim 1 decided (Entailment beats unknown); claim 2 excluded.
        assert entry["metrics"]["faithfulness"] == 1.0

    def test_run_entry_metrics_and_counts(self, pipeline):
        abstained = _full_item(**{RESPONSE_KG: [], "is_abstention": True})
        run = pipeline._build_run([_checked_item(), abstained])
        assert list(run) == ["_meta", "metrics", "counts", "items"]
        assert run["metrics"] == {"faithfulness": 0.5, "abstention_rate": 0.5,
                                  "extraction_error_rate": 0.0, "checker_failure_rate": 0.0}
        counts = run["counts"]
        assert counts["support"] == {"faithfulness": 1}
        assert counts["abstention"] == {"evaluated": 2, "errored": 0, "abstained": 1, "answered": 1}
        assert counts["reliability"]["checking"] == {"unjudged": 0, "issued": 4}
        assert counts["pipeline"]["retrieved2response"]["verdicts"] == 4

    def test_findings_open_the_branches(self, pipeline):
        abstained = _full_item(**{RESPONSE_KG: [], "is_abstention": True})
        run = pipeline._build_run([_checked_item(), abstained])
        findings = pipeline._build_findings(run["items"])
        assert list(findings) == ["ungrounded", "contradicted", "undecidable",
                                  "abstained", "extraction_failed"]
        # claim 2 of the checked item has no Entailment anywhere
        assert findings["ungrounded"] == [
            {"query_id": "0", "query": "", "claim": "Nile has 5M inhabitants",
             "chunks_checked": 2}]
        assert findings["contradicted"] == [] and findings["undecidable"] == []
        assert findings["abstained"][0]["query_id"] == "0"
        assert findings["extraction_failed"] == []


# ── Facade ───────────────────────────────────────────────────────────────────

class TestFacade:

    def test_check_faithfulness_returns_single_entry(self):
        fake_report = {"runs": [{"items": [{"metrics": {"faithfulness": 0.75}}]}]}

        with patch("claimlens.pipelines.faithfulness.FaithfulnessPipeline") as cls:
            instance = cls.return_value
            instance.last_report = fake_report
            entry = check_faithfulness(
                "some response", ["chunk"],
                extractor_model=EXT, checker_model=CHK,
            )

        assert entry == fake_report["runs"][0]["items"][0]
        cls.assert_called_once()
        # Facade forces silence — it is a library call, not a CLI run
        assert cls.call_args.kwargs["verbosity"] == "silent"
        instance.run_sync.assert_called_once_with(
            [{"response": "some response", "retrieved_context": ["chunk"]}]
        )