"""
Unit tests for RefCheckerPipeline — validation gate + run() composition.

RefChecker is a service that composes ExtractionService + CheckingService.
Both child services are mocked at construction, so these tests never touch the
LLM or need API keys.

The focus is validation. RefChecker requires BOTH keys on every item:
    - 'response'  (extraction needs it)
    - 'reference' (checking needs it)
so the check is stricter than a single service's. Canonicalization
(context -> reference) runs in run() before validation, so a context-only
item becomes valid there.
"""

import pytest
from unittest.mock import patch

from claimlens.pipelines.refchecker import RefCheckerPipeline
from claimlens.exceptions import InvalidInputError


EXTRACTOR = "ext-model"
CHECKER = "chk-model"
KG_KEY = f"{EXTRACTOR}_response_kg"
VERDICT_KEY = f"{CHECKER}_checker_verdict"

_MISSING = object()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pipeline():
    """Build a RefCheckerPipeline with both child services mocked out."""
    with patch("claimlens.pipelines.refchecker.ExtractionService"), \
         patch("claimlens.pipelines.refchecker.CheckingService"):
        return RefCheckerPipeline(
            extractor_model=EXTRACTOR, checker_model=CHECKER, verbosity="silent"
        )


def _running_pipeline():
    """Pipeline whose child services are async passthroughs that mutate data
    in place (extraction writes a triplet, checking stamps a verdict)."""
    p = _pipeline()

    async def ext_run(data):
        for it in data:
            it[KG_KEY] = [{"subject": "s", "predicate": "p", "object": "o"}]
        return data

    async def chk_run(data):
        for it in data:
            for t in it.get(KG_KEY, []):
                t[VERDICT_KEY] = "Entailment"
        return data

    p._extraction.run = ext_run
    p._checking.run = chk_run
    return p


def _item(item_id="1", response="a response", reference=("passage 1",)):
    """Build an item; pass response=_MISSING or reference=_MISSING to omit a key."""
    it = {"id": item_id}
    if response is not _MISSING:
        it["response"] = response
    if reference is not _MISSING:
        it["reference"] = list(reference) if isinstance(reference, tuple) else reference
    return it


# ── _validate: requires BOTH 'response' and 'reference' ──────────────────────

class TestValidate:

    def test_item_with_both_keys_passes(self):
        p = _pipeline()
        valid = p._validate([_item()])
        assert len(valid) == 1

    def test_missing_response_dropped(self):
        p = _pipeline()
        with pytest.raises(InvalidInputError):
            p._validate([_item(response=_MISSING)])

    def test_missing_reference_dropped(self):
        """The pipeline-specific requirement: no reference -> dropped."""
        p = _pipeline()
        with pytest.raises(InvalidInputError):
            p._validate([_item(reference=_MISSING)])

    def test_missing_both_dropped(self):
        p = _pipeline()
        with pytest.raises(InvalidInputError):
            p._validate([_item(response=_MISSING, reference=_MISSING)])

    def test_mixed_keeps_only_fully_valid_and_preserves_order(self):
        p = _pipeline()
        data = [
            _item(item_id="good"),                       # both keys -> valid
            _item(item_id="no_resp", response=_MISSING), # dropped
            _item(item_id="no_ref", reference=_MISSING), # dropped
            _item(item_id="good2"),                      # both keys -> valid
        ]
        valid = p._validate(data)
        assert [it["id"] for it in valid] == ["good", "good2"]

    def test_all_invalid_raises(self):
        p = _pipeline()
        data = [_item(response=_MISSING), _item(reference=_MISSING)]
        with pytest.raises(InvalidInputError, match="No items contain both"):
            p._validate(data)

    def test_returns_same_object_references(self):
        """Valid items are the same dicts (mutation must reach the caller)."""
        p = _pipeline()
        item = _item()
        valid = p._validate([item])
        assert valid[0] is item

    def test_empty_reference_currently_kept_presence_only(self):
        """_validate checks key PRESENCE, not emptiness: reference=[] passes.
        If validation is later tightened to require a non-empty reference,
        this test should flip."""
        p = _pipeline()
        valid = p._validate([_item(reference=[])])
        assert len(valid) == 1


# ── run(): canonicalize -> validate -> compose -> return ─────────────────────

class TestRun:

    def test_context_alias_is_canonicalized_then_valid(self):
        """An item with 'context' (not 'reference') is invalid to _validate
        alone, but run() canonicalizes context->reference first, so it survives
        and gets processed."""
        p = _running_pipeline()
        data = [{"id": "1", "response": "r", "context": ["the ref"]}]
        out = p.run_sync(data)
        assert "reference" in out[0]
        assert out[0][KG_KEY][0][VERDICT_KEY] == "Entailment"

    def test_composes_both_stages_and_returns_data(self):
        p = _running_pipeline()
        data = [_item()]
        out = p.run_sync(data)
        assert out is data                       # same list, mutated
        assert out[0][KG_KEY][0][VERDICT_KEY] == "Entailment"

    def test_invalid_items_pass_through_untouched(self):
        """Dropped items stay in the returned document, just unprocessed."""
        p = _running_pipeline()
        data = [_item(item_id="good"), {"id": "bad"}]
        out = p.run_sync(data)
        assert KG_KEY in out[0]
        assert KG_KEY not in out[1]

    def test_all_invalid_raises_before_execute(self):
        p = _running_pipeline()
        with pytest.raises(InvalidInputError):
            p.run_sync([{"id": "bad"}])


# ── Record + findings: the same skeleton as every report producer ────────────

class TestRecord:

    def test_run_populates_the_record(self):
        p = _running_pipeline()
        p.run_sync([_item()])
        doc = p.last_report
        assert list(doc) == ["_meta", "metrics", "variance", "runs"]
        assert doc["_meta"]["report_type"] == "refcheck"
        assert doc["_meta"]["runs"] == 1 and len(doc["runs"]) == 1
        # no aggregate metric → the roster is empty, the skeleton holds
        assert doc["metrics"] == {} and doc["variance"] == {}
        run = doc["runs"][0]
        assert list(run) == ["_meta", "metrics", "counts", "items"]

    def test_items_carry_the_checked_claims(self):
        p = _running_pipeline()
        p.run_sync([_item()])
        items = p.last_report["runs"][0]["items"]
        assert items[0]["id"] == "1"
        assert items[0]["claims"] == [
            {"claim": "s p o", "verdict": "Entailment", "explanation": None}]
        assert items[0]["is_abstention"] is False

    def test_counts_mirror_the_console(self):
        p = _running_pipeline()
        p.run_sync([_item()])
        counts = p.last_report["runs"][0]["counts"]
        assert counts["extraction"] == {"with_claims": 1, "abstained": 0, "failed": 0}
        assert counts["checking"] == {"Entailment": 1, "Contradiction": 0, "Neutral": 0, "unjudged": 0}
        assert counts["pipeline"]["check reference"]["verdicts"] == 1

    def test_meta_counts_dropped_items(self):
        """Invalid items are counted, not evaluated, and never in items."""
        p = _running_pipeline()
        p.run_sync([_item(item_id="good"), {"id": "bad"}])
        meta = p.last_report["_meta"]
        assert (meta["total_items"], meta["evaluated_items"],
                meta["dropped_items"]) == (2, 1, 1)
        assert [it["id"] for it in p.last_report["runs"][0]["items"]] == ["good"]

    def test_no_arguments_leak_into_meta(self):
        """Models are given, not discovered — they belong in _args."""
        p = _running_pipeline()
        p.run_sync([_item()])
        assert "extractor_model" not in p.last_report["_meta"]
        assert "checker_model" not in p.last_report["_meta"]

    def test_last_report_is_none_before_a_run(self):
        assert _pipeline().last_report is None
        assert _pipeline().last_findings is None

    def test_build_run_is_pure_projection(self):
        """Rebuildable anytime, no LLM calls, same content."""
        p = _running_pipeline()
        data = [_item()]
        p.run_sync(data)
        assert p._build_run(data)["items"] == p.last_report["runs"][0]["items"]


class TestFindings:

    def _items(self):
        return [
            {"id": "a", "question": "q", "response": "r", "reference": ["x"], "is_abstention": False,
             "claims": [{"claim": "s p o", "verdict": "Entailment", "explanation": "e1"},
                        {"claim": "s q o", "verdict": "Neutral", "explanation": "not covered"},
                        {"claim": "s r o", "verdict": "Contradiction", "explanation": "opposite"},
                        {"claim": "s t o", "verdict": None, "explanation": None, "error": "timeout"}]},
            {"id": "b", "question": "q", "response": "I don't know.", "reference": ["x"],
             "is_abstention": True, "claims": []},
            {"id": "c", "question": "q", "response": "r", "reference": ["x"], "is_abstention": False,
             "claims": [], "extraction_error": "parse_failure"},
        ]

    def test_branches_open_the_checking_tree(self):
        f = RefCheckerPipeline._build_findings(self._items())
        assert list(f) == ["unsupported", "contradicted", "unjudged", "abstained", "extraction_failed"]
        assert f["unsupported"] == [{"id": "a", "question": "q", "claim": "s q o", "explanation": "not covered"}]
        assert f["contradicted"] == [{"id": "a", "question": "q", "claim": "s r o", "explanation": "opposite"}]
        assert f["unjudged"] == [{"id": "a", "question": "q", "claim": "s t o", "cause": "timeout"}]
        assert f["abstained"] == [{"id": "b", "question": "q", "response": "I don't know."}]
        assert f["extraction_failed"] == [{"id": "c", "question": "q", "cause": "parse_failure"}]

    def test_run_produces_the_findings_document(self):
        p = _running_pipeline()
        p.run_sync([_item()])
        f = p.last_findings
        assert list(f) == ["_meta", "runs"]
        assert all(v == [] for v in f["runs"][0]["findings"].values())


# ── Envelope input: {"results": [...]} accepted, like ragcheck/faithcheck ────

class TestEnvelopeInput:

    def test_results_envelope_is_unwrapped(self):
        p = _running_pipeline()
        p.run_sync({"results": [_item()]})
        assert p.last_report["runs"][0]["items"][0]["claims"][0]["verdict"] == "Entailment"

    def test_bare_list_still_accepted(self):
        p = _running_pipeline()
        p.run_sync([_item()])
        assert p.last_report["_meta"]["total_items"] == 1

    def test_neither_list_nor_envelope_raises(self):
        p = _running_pipeline()
        with pytest.raises(InvalidInputError):
            p.run_sync({"items": [_item()]})
