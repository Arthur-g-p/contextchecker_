"""Unit tests for the AtomizationService private pipeline methods: _validate, _filter, _build_payloads, and _serialize."""

import pytest
from unittest.mock import patch, MagicMock

from claimlens.services.atomization import AtomizationService
from claimlens.models import AtomizationPayload
from claimlens.workers.atomizer import AtomicTriplet, AtomizationDecision
from claimlens.stats import PhaseStats
from claimlens.exceptions import InvalidInputError


# ── Constants ────────────────────────────────────────────────────────────────

SOURCE_KEY = "test-model_response_kg"
ARCHIVE_KEY = "test-model_response_kg_not_atomized"
MODEL = "test-model"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _service(**kwargs) -> AtomizationService:
    """Build a service with mocked Atomizer worker."""
    defaults = dict(
        model=MODEL,
        source_kg_key=SOURCE_KEY,
    )
    defaults.update(kwargs)
    with patch("claimlens.services.atomization.settings") as mock_settings:
        mock_settings.ATOMIZER_API_KEY = "test-key"
        mock_settings.PROMPT_PATH = "/fake/prompts.json"
        mock_settings.get_logger = MagicMock(return_value=MagicMock())
        with patch("claimlens.services.atomization.Atomizer"):
            return AtomizationService(**defaults)


def _make_item(
    triplets: list[dict] | None = None,
    item_id: str = "1",
    response: str | None = "test response",
) -> dict:
    """Build a test item dict."""
    item = {"id": item_id}
    if response is not None:
        item["response"] = response
    if triplets is not None:
        item[SOURCE_KEY] = triplets
    return item


def _triplet(s: str, p: str, o: str, **meta) -> dict:
    t = {"subject": s, "predicate": p, "object": o}
    t.update(meta)
    return t


def _legacy_triplet(s: str, p: str, o: str, **meta) -> dict:
    t = {"triplet": [s, p, o]}
    t.update(meta)
    return t


# ── Test _validate ───────────────────────────────────────────────────────────

class TestValidate:
    """_validate drops invalid items and canonicalizes valid ones."""

    def test_valid_items_pass(self):
        svc = _service()
        data = [_make_item(triplets=[_triplet("a", "b", "c")])]
        valid = svc._validate(data)
        assert len(valid) == 1

    def test_missing_source_key_dropped(self):
        svc = _service()
        data = [
            _make_item(triplets=[_triplet("a", "b", "c")]),
            _make_item(triplets=None),  # no source key
        ]
        valid = svc._validate(data)
        assert len(valid) == 1
        assert valid[0]["id"] == "1"

    def test_empty_triplets_dropped(self):
        svc = _service()
        data = [_make_item(triplets=[])]
        with pytest.raises(InvalidInputError):
            svc._validate(data)

    def test_missing_response_dropped(self):
        svc = _service()
        data = [
            _make_item(triplets=[_triplet("a", "b", "c")], response=None),
            _make_item(triplets=[_triplet("x", "y", "z")], response=""),
            _make_item(triplets=[_triplet("d", "e", "f")], response="valid"),
        ]
        valid = svc._validate(data)
        assert len(valid) == 1
        assert valid[0]["response"] == "valid"

    def test_all_missing_raises(self):
        svc = _service()
        data = [_make_item(triplets=None)]
        with pytest.raises(InvalidInputError, match="No items contain"):
            svc._validate(data)

    def test_canonicalizes_triplets_on_validate(self):
        svc = _service()
        data = [_make_item(triplets=[_legacy_triplet("france", "has capital", "paris")])]
        valid = svc._validate(data)
        assert "triplet" not in valid[0][SOURCE_KEY][0]
        assert valid[0][SOURCE_KEY][0]["subject"] == "france"
        assert valid[0][SOURCE_KEY][0]["predicate"] == "has capital"
        assert valid[0][SOURCE_KEY][0]["object"] == "paris"


# ── Test _filter ─────────────────────────────────────────────────────────────

class TestFilter:
    """_filter skips items whose triplets are already atomized."""

    def test_no_skip(self):
        svc = _service()
        data = [_make_item(triplets=[_triplet("a", "b", "c")])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert skipped == 0

    def test_already_atomized_skipped(self):
        svc = _service()
        data = [
            _make_item(triplets=[
                _triplet("a", "b", "c", atomized=True),
                _triplet("d", "e", "f", atomized="failed"),
            ])
        ]
        pending, skipped = svc._filter(data)
        assert len(pending) == 0
        assert skipped == 1

    def test_partial_atomized_not_skipped(self):
        svc = _service()
        data = [
            _make_item(triplets=[
                _triplet("a", "b", "c", atomized=True),
                _triplet("d", "e", "f"),  # missing atomized flag
            ])
        ]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert skipped == 0

    def test_mixed(self):
        svc = _service()
        data = [
            _make_item(triplets=[_triplet("a", "b", "c")], item_id="1"),
            _make_item(triplets=[_triplet("d", "e", "f", atomized=True)], item_id="2"),
        ]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert pending[0]["id"] == "1"
        assert skipped == 1


# ── Test _build_payloads ─────────────────────────────────────────────────────

class TestBuildPayloads:
    def test_payload_structure(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("cats", "are", "animals")], response="cats response")]
        payloads = svc._build_payloads(items)
        assert len(payloads) == 1
        p = payloads[0]
        assert isinstance(p, AtomizationPayload)
        assert p.subject == "cats"
        assert p.predicate == "are"
        assert p.object == "animals"
        assert p.response == "cats response"
        assert p.item_index == 0
        assert p.triplet_index == 0

    def test_multiple_items_and_triplets(self):
        svc = _service()
        items = [
            _make_item(triplets=[_triplet("a", "b", "c"), _triplet("d", "e", "f")], item_id="1"),
            _make_item(triplets=[_triplet("x", "y", "z")], item_id="2"),
        ]
        payloads = svc._build_payloads(items)
        assert len(payloads) == 3
        assert payloads[0].item_index == 0
        assert payloads[0].triplet_index == 0
        assert payloads[1].item_index == 0
        assert payloads[1].triplet_index == 1
        assert payloads[2].item_index == 1
        assert payloads[2].triplet_index == 0


# ── Test _serialize ──────────────────────────────────────────────────────────

class TestSerialize:
    def test_keep_decision_preserves_original(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("a", "b", "c", custom="meta")])]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(reasoning="is atomic", is_atomic=True, split=[])
        ]
        phase_stats = PhaseStats(failed_indices=[])

        stats = svc._serialize(items, payloads, results, phase_stats)

        # Output mutated in place, original keys (custom="meta") preserved, atomized=True flag added
        output = items[0][SOURCE_KEY]
        assert len(output) == 1
        assert output[0]["subject"] == "a"
        assert output[0]["custom"] == "meta"
        assert output[0]["atomized"] is True
        assert ARCHIVE_KEY not in items[0]  # No split occurred

        assert stats["keep"] == 1
        assert stats["split"] == 0
        assert stats["failed"] == 0

    def test_failed_decision_preserves_original(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("a", "b", "c", custom="meta")])]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(reasoning="error", is_atomic=True, split=[])
        ]
        phase_stats = PhaseStats(failed_indices=[0])

        stats = svc._serialize(items, payloads, results, phase_stats)

        output = items[0][SOURCE_KEY]
        assert len(output) == 1
        assert output[0]["subject"] == "a"
        assert output[0]["custom"] == "meta"
        assert output[0]["atomized"] == "failed"
        assert ARCHIVE_KEY not in items[0]

        assert stats["keep"] == 0
        assert stats["split"] == 0
        assert stats["failed"] == 1

    def test_split_decision_emits_children(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("a", "b", "c", custom="meta")])]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(
                reasoning="needs split",
                is_atomic=False,
                split=[
                    AtomicTriplet(subject="s1", predicate="p1", object="o1"),
                    AtomicTriplet(subject="s2", predicate="p2", object="o2"),
                ]
            )
        ]
        phase_stats = PhaseStats(failed_indices=[])

        stats = svc._serialize(items, payloads, results, phase_stats)

        # Output mutated to children, metadata (custom="meta") preserved on both, archive original created
        output = items[0][SOURCE_KEY]
        assert len(output) == 2
        assert output[0] == {"subject": "s1", "predicate": "p1", "object": "o1", "custom": "meta", "atomized": True}
        assert output[1] == {"subject": "s2", "predicate": "p2", "object": "o2", "custom": "meta", "atomized": True}

        assert items[0][ARCHIVE_KEY] == [_triplet("a", "b", "c", custom="meta")]

        assert stats["keep"] == 0
        assert stats["split"] == 1
        assert stats["children"] == 2
        assert stats["output"] == 2

    def test_deduplication_enabled(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("a", "b", "c")])]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(
                reasoning="split to dups",
                is_atomic=False,
                split=[
                    AtomicTriplet(subject="dup", predicate="is", object="dup"),
                    AtomicTriplet(subject="dup", predicate="is", object="dup"),
                ]
            )
        ]
        phase_stats = PhaseStats(failed_indices=[])

        stats = svc._serialize(items, payloads, results, phase_stats)

        output = items[0][SOURCE_KEY]
        assert len(output) == 1  # Deduplicated down to 1
        assert output[0]["subject"] == "dup"

        assert stats["dups"] == 1
        assert stats["output"] == 1

    def test_deduplication_disabled(self):
        svc = _service(dedup=False)
        items = [_make_item(triplets=[_triplet("a", "b", "c")])]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(
                reasoning="split to dups",
                is_atomic=False,
                split=[
                    AtomicTriplet(subject="dup", predicate="is", object="dup"),
                    AtomicTriplet(subject="dup", predicate="is", object="dup"),
                ]
            )
        ]
        phase_stats = PhaseStats(failed_indices=[])

        stats = svc._serialize(items, payloads, results, phase_stats)

        output = items[0][SOURCE_KEY]
        assert len(output) == 2  # Duplicates preserved

        assert stats["dups"] == 1
        assert stats["output"] == 2

    def test_trace_populated(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("a", "b", "c")], item_id="trace_id")]
        payloads = [
            AtomizationPayload(subject="a", predicate="b", object="c", response="t", item_index=0, triplet_index=0)
        ]
        results = [
            AtomizationDecision(reasoning="reason", is_atomic=True, split=[])
        ]
        phase_stats = PhaseStats(failed_indices=[])

        svc._serialize(items, payloads, results, phase_stats)

        assert len(svc.last_trace) == 1
        t = svc.last_trace[0]
        assert t["id"] == "trace_id"
        assert t["duplicates_removed"] == 0
        assert len(t["decisions"]) == 1
        assert t["decisions"][0]["decision"] == "keep"
        assert t["decisions"][0]["reasoning"] == "reason"
