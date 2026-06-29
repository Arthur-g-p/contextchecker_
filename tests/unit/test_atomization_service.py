"""
Tests for the AtomizationService — validation, filtering, serialization, and pipeline.

Mirrors test patterns from test_extraction_service.py.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from contextchecker.services.atomization import AtomizationService
from contextchecker.models import AtomizationPayload
from contextchecker.workers.atomizer import AtomicTriplet
from contextchecker.exceptions import InvalidInputError


# ── Constants ────────────────────────────────────────────────────────────────

SOURCE_KEY = "test-model_response_kg"
TARGET_KEY = "test-model_response_kg_atomized"
MODEL = "test-model"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _service(**kwargs) -> AtomizationService:
    """Build a service with mocked Atomizer worker."""
    defaults = dict(
        model=MODEL,
        source_kg_key=SOURCE_KEY,
    )
    defaults.update(kwargs)
    with patch("contextchecker.services.atomization.settings") as mock_settings:
        mock_settings.ATOMIZER_API_KEY = "test-key"
        mock_settings.PROMPT_PATH = "/fake/prompts.json"
        mock_settings.PROMPTS = {"atomizer_prompt": "test {{triplet}}"}
        mock_settings.get_logger = MagicMock(return_value=MagicMock())
        with patch("contextchecker.services.atomization.Atomizer"):
            return AtomizationService(**defaults)


def _make_item(
    triplets: list[dict] | None = None,
    atomized: list[dict] | None = None,
    item_id: str = "1",
) -> dict:
    """Build a test item dict."""
    item = {"id": item_id, "response": "test response"}
    if triplets is not None:
        item[SOURCE_KEY] = triplets
    if atomized is not None:
        item[TARGET_KEY] = atomized
    return item


def _triplet(s: str, p: str, o: str) -> dict:
    return {"subject": s, "predicate": p, "object": o}


# ── Test _validate ───────────────────────────────────────────────────────────

class TestValidate:
    """_validate drops items without the source kg key."""

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

    def test_empty_triplets_dropped(self):
        svc = _service()
        data = [_make_item(triplets=[])]
        with pytest.raises(InvalidInputError):
            svc._validate(data)

    def test_all_missing_raises(self):
        svc = _service()
        data = [_make_item(triplets=None)]
        with pytest.raises(InvalidInputError, match="No items contain"):
            svc._validate(data)


# ── Test _filter ─────────────────────────────────────────────────────────────

class TestFilter:
    """_filter skips items that already have the target atomized key."""

    def test_no_skip(self):
        svc = _service()
        data = [_make_item(triplets=[_triplet("a", "b", "c")])]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert skipped == 0

    def test_already_atomized_skipped(self):
        svc = _service()
        data = [_make_item(
            triplets=[_triplet("a", "b", "c")],
            atomized=[_triplet("a", "b", "c")],
        )]
        pending, skipped = svc._filter(data)
        assert len(pending) == 0
        assert skipped == 1

    def test_mixed(self):
        svc = _service()
        data = [
            _make_item(triplets=[_triplet("a", "b", "c")], item_id="1"),
            _make_item(
                triplets=[_triplet("d", "e", "f")],
                atomized=[_triplet("d", "e", "f")],
                item_id="2",
            ),
        ]
        pending, skipped = svc._filter(data)
        assert len(pending) == 1
        assert pending[0]["id"] == "1"
        assert skipped == 1


# ── Test _build_payloads ─────────────────────────────────────────────────────

class TestBuildPayloads:
    def test_single_item_single_triplet(self):
        svc = _service()
        items = [_make_item(triplets=[_triplet("cats", "are", "animals")])]
        payloads = svc._build_payloads(items)
        assert len(payloads) == 1
        assert payloads[0].triplet == "cats are animals"
        assert payloads[0].item_index == 0
        assert payloads[0].triplet_index == 0

    def test_multiple_triplets_per_item(self):
        svc = _service()
        items = [_make_item(triplets=[
            _triplet("cats", "are", "animals"),
            _triplet("dogs", "are", "animals"),
        ])]
        payloads = svc._build_payloads(items)
        assert len(payloads) == 2
        assert payloads[1].triplet_index == 1

    def test_multiple_items(self):
        svc = _service()
        items = [
            _make_item(triplets=[_triplet("a", "b", "c")], item_id="1"),
            _make_item(triplets=[_triplet("d", "e", "f")], item_id="2"),
        ]
        payloads = svc._build_payloads(items)
        assert len(payloads) == 2
        assert payloads[0].item_index == 0
        assert payloads[1].item_index == 1

    def test_legacy_triplet_format(self):
        svc = _service()
        items = [_make_item(triplets=[{"triplet": ["cats", "are", "animals"]}])]
        payloads = svc._build_payloads(items)
        assert payloads[0].triplet == "cats are animals"


# ── Test _triplet_to_str ─────────────────────────────────────────────────────

class TestTripletToStr:
    def test_canonical(self):
        result = AtomizationService._triplet_to_str(
            {"subject": "cats", "predicate": "are", "object": "animals"}
        )
        assert result == "cats are animals"

    def test_legacy(self):
        result = AtomizationService._triplet_to_str(
            {"triplet": ["cats", "are", "animals"]}
        )
        assert result == "cats are animals"

    def test_fallback(self):
        result = AtomizationService._triplet_to_str({"something": "else"})
        assert "something" in result


# ── Test _serialize ──────────────────────────────────────────────────────────

class TestSerialize:
    def test_groups_by_item(self):
        svc = _service()
        items = [
            _make_item(triplets=[_triplet("a", "b", "c"), _triplet("d", "e", "f")]),
        ]

        payloads = [
            AtomizationPayload(triplet="a b c", item_index=0, triplet_index=0),
            AtomizationPayload(triplet="d e f", item_index=0, triplet_index=1),
        ]
        results = [
            [AtomicTriplet(subject="a", predicate="b", object="c")],
            [AtomicTriplet(subject="d1", predicate="e1", object="f1"),
             AtomicTriplet(subject="d2", predicate="e2", object="f2")],
        ]

        svc._serialize(items, payloads, results)

        assert TARGET_KEY in items[0]
        assert len(items[0][TARGET_KEY]) == 3  # 1 + 2 split

    def test_multi_item_serialize(self):
        svc = _service()
        items = [
            _make_item(triplets=[_triplet("a", "b", "c")], item_id="1"),
            _make_item(triplets=[_triplet("x", "y", "z")], item_id="2"),
        ]

        payloads = [
            AtomizationPayload(triplet="a b c", item_index=0, triplet_index=0),
            AtomizationPayload(triplet="x y z", item_index=1, triplet_index=0),
        ]
        results = [
            [AtomicTriplet(subject="a", predicate="b", object="c")],
            [AtomicTriplet(subject="x", predicate="y", object="z")],
        ]

        svc._serialize(items, payloads, results)
        assert len(items[0][TARGET_KEY]) == 1
        assert len(items[1][TARGET_KEY]) == 1


# ── Integration-lite tests ───────────────────────────────────────────────────

class TestRunIntegration:
    """Test run() flow with mocked atomizer worker."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        svc = _service()

        data = [
            _make_item(
                triplets=[_triplet("cats and dogs", "are", "animals")],
                item_id="1",
            ),
        ]

        # Mock atomizer: splits compound triplet
        async def mock_atomize(payloads):
            return [
                [
                    AtomicTriplet(subject="cats", predicate="are", object="animals"),
                    AtomicTriplet(subject="dogs", predicate="are", object="animals"),
                ]
            ]

        svc._atomizer.atomize_batch = AsyncMock(side_effect=mock_atomize)
        svc._atomizer.last_stats = MagicMock(
            total_items=2, success=1, empty=0, total_errors=0,
            http_requests=1, first_pass_count=1, first_pass_ok=1,
            parse_error=0, context_too_long=0, content_policy=0,
            rounds=[], failed_indices=[],
        )

        with patch.object(svc, "_log_validation"), \
             patch.object(svc, "_log_skip"), \
             patch.object(svc, "_log_config"), \
             patch.object(svc, "_log_results"):
            result = await svc.run(data)

        assert TARGET_KEY in result[0]
        assert len(result[0][TARGET_KEY]) == 2
        assert result[0][TARGET_KEY][0]["subject"] == "cats"
        assert result[0][TARGET_KEY][1]["subject"] == "dogs"

    @pytest.mark.asyncio
    async def test_already_atomized_skipped(self):
        svc = _service()

        data = [
            _make_item(
                triplets=[_triplet("cats", "are", "animals")],
                atomized=[_triplet("cats", "are", "animals")],
                item_id="1",
            ),
        ]

        with patch.object(svc, "_log_validation"), \
             patch.object(svc, "_log_skip"):
            result = await svc.run(data)

        # Should not have called atomizer at all
        svc._atomizer.atomize_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_atomic_passthrough(self):
        """Already-atomic triplet → returned unchanged."""
        svc = _service()

        data = [
            _make_item(
                triplets=[_triplet("cats", "are", "animals")],
                item_id="1",
            ),
        ]

        async def mock_atomize(payloads):
            return [
                [AtomicTriplet(subject="cats", predicate="are", object="animals")]
            ]

        svc._atomizer.atomize_batch = AsyncMock(side_effect=mock_atomize)
        svc._atomizer.last_stats = MagicMock(
            total_items=1, success=1, empty=0, total_errors=0,
            http_requests=1, first_pass_count=1, first_pass_ok=1,
            parse_error=0, context_too_long=0, content_policy=0,
            rounds=[], failed_indices=[],
        )

        with patch.object(svc, "_log_validation"), \
             patch.object(svc, "_log_skip"), \
             patch.object(svc, "_log_config"), \
             patch.object(svc, "_log_results"):
            result = await svc.run(data)

        assert len(result[0][TARGET_KEY]) == 1
        assert result[0][TARGET_KEY][0]["subject"] == "cats"
