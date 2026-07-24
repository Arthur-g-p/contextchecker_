"""
Unit tests for FaithfulnessPipeline: validation, report projection, the
faithfulness metric, and the check_faithfulness facade. Workers patched out
at construction — no LLM.
"""

import pytest
from unittest.mock import patch

from contextchecker.exceptions import InvalidInputError
from contextchecker.pipelines.faithfulness import (
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
    monkeypatch.setattr("contextchecker.settings.EXTRACTOR_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr("contextchecker.settings.CHECKER_API_KEY", FAKE_API_KEY)


@pytest.fixture
def pipeline():
    with patch("contextchecker.services.extraction.Extractor"), \
         patch("contextchecker.services.checking.Checker"):
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

    @pytest.mark.parametrize("key", ["response", "retrieved_context"])
    def test_missing_or_empty_key_drops_item(self, pipeline, key):
        valid = pipeline._validate([_full_item(**{key: ""}), _full_item()])
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
             f"{NAMESPACE}_verdicts": {"000": "Entailment", "001": "Neutral"}},
            {"subject": "Nile", "predicate": "has", "object": "5M inhabitants",
             f"{NAMESPACE}_verdicts": {"000": "Neutral", "001": "Neutral"}},
        ],
    })


class TestBuildReport:

    def test_entry_shape_and_metric(self, pipeline):
        report = pipeline.build_report([_checked_item()])
        assert report["_meta"]["report_type"] == "faithfulness"
        entry = report["results"][0]
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
        entry = pipeline.build_report([_checked_item()])["results"][0]
        assert entry["claim_support"] == [["000"], []]

    def test_abstention_is_null(self, pipeline):
        item = _full_item(**{RESPONSE_KG: [], "is_abstention": True})
        entry = pipeline.build_report([item])["results"][0]
        assert entry["is_abstention"] is True
        assert entry["metrics"]["faithfulness"] is None

    def test_extraction_error_is_null(self, pipeline):
        item = _full_item(**{
            RESPONSE_KG: [],
            f"{EXT}_extraction_error": "parse_failure",
        })
        entry = pipeline.build_report([item])["results"][0]
        assert entry["extraction_errors"] == {"response": "parse_failure"}
        assert entry["metrics"]["faithfulness"] is None

    def test_unknown_cells_propagate(self, pipeline):
        """No Entailment + a None cell → claim unknowable → excluded."""
        item = _full_item(**{
            RESPONSE_KG: [
                {"subject": "a", "predicate": "b", "object": "c",
                 f"{NAMESPACE}_verdicts": {"000": "Entailment", "001": None}},
                {"subject": "d", "predicate": "e", "object": "f",
                 f"{NAMESPACE}_verdicts": {"000": "Neutral", "001": None}},
            ],
        })
        entry = pipeline.build_report([item])["results"][0]
        # Claim 1 decided (Entailment beats unknown); claim 2 excluded.
        assert entry["metrics"]["faithfulness"] == 1.0

    def test_overall_metrics(self, pipeline):
        abstained = _full_item(**{RESPONSE_KG: [], "is_abstention": True})
        report = pipeline.build_report([_checked_item(), abstained])
        om = report["overall_metrics"]
        assert om["faithfulness"] == 0.5
        assert om["support"]["faithfulness"] == 1
        assert om["abstention_rate"] == 0.5
        assert om["extraction_error_rate"] == 0.0


# ── Facade ───────────────────────────────────────────────────────────────────

class TestFacade:

    def test_check_faithfulness_returns_single_entry(self):
        fake_report = {"results": [{"metrics": {"faithfulness": 0.75}}]}

        with patch("contextchecker.pipelines.faithfulness.FaithfulnessPipeline") as cls:
            instance = cls.return_value
            instance.last_report = fake_report
            entry = check_faithfulness(
                "some response", ["chunk"],
                extractor_model=EXT, checker_model=CHK,
            )

        assert entry == fake_report["results"][0]
        cls.assert_called_once()
        # Facade forces silence — it is a library call, not a CLI run
        assert cls.call_args.kwargs["verbosity"] == "silent"
        instance.run_sync.assert_called_once_with(
            [{"response": "some response", "retrieved_context": ["chunk"]}]
        )