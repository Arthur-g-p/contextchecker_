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

from contextchecker.pipelines.refchecker import RefCheckerPipeline
from contextchecker.exceptions import InvalidInputError


EXTRACTOR = "ext-model"
CHECKER = "chk-model"
KG_KEY = f"{EXTRACTOR}_response_kg"
VERDICT_KEY = f"{CHECKER}_checker_verdict"

_MISSING = object()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pipeline():
    """Build a RefCheckerPipeline with both child services mocked out."""
    with patch("contextchecker.pipelines.refchecker.ExtractionService"), \
         patch("contextchecker.pipelines.refchecker.CheckingService"):
        return RefCheckerPipeline(
            extractor_model=EXTRACTOR, checker_model=CHECKER, quiet=True
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
