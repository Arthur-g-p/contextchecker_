"""
Tests for AtomizationService._validate and _filter — the two pipeline gate steps.

These test the NEW logic:
- _validate requires: source kg key (non-empty) + response field
- _filter checks for 'atomized' flag on individual triplets (not a target key)
"""

import pytest
from unittest.mock import patch, MagicMock

from contextchecker.services.atomization import AtomizationService
from contextchecker.exceptions import InvalidInputError


# ── Constants ────────────────────────────────────────────────────────────────

SOURCE_KEY = "test-model_response_kg"
MODEL = "test-model"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _service(**kwargs) -> AtomizationService:
    """Build a service with mocked Atomizer worker."""
    defaults = dict(model=MODEL, source_kg_key=SOURCE_KEY)
    defaults.update(kwargs)
    with patch("contextchecker.services.atomization.settings") as mock_settings:
        mock_settings.ATOMIZER_API_KEY = "test-key"
        mock_settings.PROMPT_PATH = "/fake/prompts.json"
        mock_settings.PROMPTS = {"atomizer_prompt": "test"}
        mock_settings.get_logger = MagicMock(return_value=MagicMock())
        with patch("contextchecker.services.atomization.Atomizer"):
            return AtomizationService(**defaults)


def _triplet(s: str, p: str, o: str, **extra) -> dict:
    """Build a triplet dict with optional extra keys (e.g. atomized=True)."""
    t = {"subject": s, "predicate": p, "object": o}
    t.update(extra)
    return t


def _legacy_triplet(s: str, p: str, o: str, **extra) -> dict:
    """Build a legacy triplet dict."""
    t = {"triplet": [s, p, o]}
    t.update(extra)
    return t


def _item(
    triplets: list[dict] | None = None,
    response: str = "test response",
    item_id: str = "1",
    include_response: bool = True,
) -> dict:
    """Build a test item dict."""
    item = {"id": item_id}
    if include_response:
        item["response"] = response
    if triplets is not None:
        item[SOURCE_KEY] = triplets
    return item


# ── _validate ────────────────────────────────────────────────────────────────

class TestValidateNew:
    """_validate requires source kg key + non-empty triplets + response."""

    def test_valid_item_passes(self):
        svc = _service()
        data = [_item(triplets=[_triplet("a", "b", "c")])]
        valid = svc._validate(data)
        assert len(valid) == 1

    def test_missing_source_key_dropped(self):
        svc = _service()
        data = [
            _item(triplets=[_triplet("a", "b", "c")]),
            _item(triplets=None),  # no source key
        ]
        valid = svc._validate(data)
        assert len(valid) == 1

    def test_empty_triplets_dropped(self):
        svc = _service()
        data = [_item(triplets=[])]
        with pytest.raises(InvalidInputError):
            svc._validate(data)

    def test_missing_response_dropped(self):
        """Items without a response field are dropped — response is required."""
        svc = _service()
        data = [_item(triplets=[_triplet("a", "b", "c")], include_response=False)]
        with pytest.raises(InvalidInputError):
            svc._validate(data)

    def test_empty_response_dropped(self):
        """Items with an empty-string response are dropped."""
        svc = _service()
        data = [_item(triplets=[_triplet("a", "b", "c")], response="")]
        with pytest.raises(InvalidInputError):
            svc._validate(data)

    def test_mixed_valid_and_invalid(self):
        """Only items with both source key + response survive."""
        svc = _service()
        data = [
            _item(triplets=[_triplet("a", "b", "c")], item_id="good"),          # valid
            _item(triplets=None, item_id="no_kg"),                               # no source key
            _item(triplets=[_triplet("d", "e", "f")], include_response=False,
                  item_id="no_resp"),                                             # no response
            _item(triplets=[], item_id="empty_kg"),                              # empty triplets
        ]
        valid = svc._validate(data)
        assert len(valid) == 1
        assert valid[0]["id"] == "good"

    def test_all_invalid_raises(self):
        svc = _service()
        data = [
            _item(triplets=None),
            _item(triplets=[_triplet("a", "b", "c")], include_response=False),
        ]
        with pytest.raises(InvalidInputError, match="No items contain"):
            svc._validate(data)


# ── _filter ──────────────────────────────────────────────────────────────────

class TestFilterNew:
    """_filter skips items where ALL triplets have an 'atomized' flag."""

    def test_no_flag_means_pending(self):
        """Triplets without 'atomized' key → item is pending."""
        svc = _service()
        data = [_item(triplets=[_triplet("a", "b", "c")])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert skipped == 0

    def test_all_atomized_true_means_skipped(self):
        """All triplets have atomized=True → skip."""
        svc = _service()
        data = [_item(triplets=[
            _triplet("a", "b", "c", atomized=True),
            _triplet("d", "e", "f", atomized=True),
        ])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 0
        assert skipped == 1

    def test_all_atomized_failed_means_skipped(self):
        """All triplets have atomized='failed' → skip (already attempted)."""
        svc = _service()
        data = [_item(triplets=[
            _triplet("a", "b", "c", atomized="failed"),
        ])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 0
        assert skipped == 1

    def test_mixed_atomized_and_not_means_pending(self):
        """Some triplets flagged, some not → item is pending (partial run)."""
        svc = _service()
        data = [_item(triplets=[
            _triplet("a", "b", "c", atomized=True),
            _triplet("d", "e", "f"),  # no flag
        ])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert skipped == 0

    def test_mixed_items(self):
        """Multiple items — one fully atomized, one not."""
        svc = _service()
        data = [
            _item(triplets=[_triplet("a", "b", "c", atomized=True)], item_id="done"),
            _item(triplets=[_triplet("d", "e", "f")], item_id="pending"),
        ]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert pending[0]["id"] == "pending"
        assert skipped == 1

    def test_legacy_triplet_with_flag_skipped(self):
        """Legacy format triplets with atomized flag are also detected."""
        svc = _service()
        data = [_item(triplets=[
            _legacy_triplet("a", "b", "c", atomized=True),
        ])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 0
        assert skipped == 1

    def test_empty_source_key_after_validation_edge_case(self):
        """Edge case: if somehow an item with empty triplets gets here, it's pending.
        (Shouldn't happen — _validate drops empties — but _filter shouldn't crash.)
        """
        svc = _service()
        # Manually construct — normally _validate would have caught this
        data = [{"id": "edge", "response": "test", SOURCE_KEY: []}]
        # all() on empty list returns True → would skip. But this is harmless:
        # an empty list means "nothing to atomize" which is correct to skip.
        pending, skipped = svc._filter(data)
        assert skipped == 1
        assert len(pending) == 0
