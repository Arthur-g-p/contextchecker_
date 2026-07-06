"""
Unit tests for the direction runner (pipelines/directions.py).

The CheckingService is replaced by a fake that mimics its output contract:
it stamps verdict keys onto every triplet of the shadow items it receives.
No LLM, no network.
"""

import asyncio

import pytest

from contextchecker.exceptions import FilterError
from contextchecker.models import Direction
from contextchecker.pipelines.directions import run_direction


KG_KEY = "ext_response_kg"
NAMESPACE = "chk_answer2response"


class FakeCheckingService:
    """Mimics CheckingService's output contract: writes a verdict on every
    triplet of every item it runs over. Records the shadow items it saw."""

    def __init__(self, verdict="Entailment", raise_filter_error=False):
        self.kg_key = KG_KEY
        self.verdict_key = f"{NAMESPACE}_verdict"
        self.explanation_key = f"{NAMESPACE}_explanation"
        self.checker_error_key = f"{NAMESPACE}_error"
        self.extraction_error_key = "ext_extraction_error"
        self._verdict = verdict
        self._raise = raise_filter_error
        self.seen_items: list[dict] = []

    async def run(self, data):
        if self._raise:
            raise FilterError("all skipped")
        self.seen_items = data
        for item in data:
            for triplet in item.get(self.kg_key, []):
                triplet[self.verdict_key] = self._verdict
        return data


def _item(triplets, **extra):
    return {KG_KEY: triplets, "response": "resp text", **extra}


def _trip(s="a", p="b", o="c"):
    return {"subject": s, "predicate": p, "object": o}


# ── Flat mode ────────────────────────────────────────────────────────────────

class TestFlatMode:

    def test_verdicts_land_on_original_triplets(self):
        """Shadow items share triplet objects — no fold-back needed."""
        fake = FakeCheckingService()
        items = [_item([_trip()], gt_answer="The GT text.")]
        direction = Direction(name="answer2response", kg_key=KG_KEY,
                              reference_key="gt_answer")
        asyncio.run(run_direction(fake, items, direction))
        assert items[0][KG_KEY][0][f"{NAMESPACE}_verdict"] == "Entailment"

    def test_string_reference_normalized_to_list(self):
        fake = FakeCheckingService()
        items = [_item([_trip()], gt_answer="plain string")]
        direction = Direction(name="answer2response", kg_key=KG_KEY,
                              reference_key="gt_answer")
        asyncio.run(run_direction(fake, items, direction))
        assert fake.seen_items[0]["reference"] == ["plain string"]

    def test_items_missing_reference_are_skipped(self):
        fake = FakeCheckingService()
        items = [_item([_trip()]), _item([_trip()], gt_answer="has ref")]
        direction = Direction(name="answer2response", kg_key=KG_KEY,
                              reference_key="gt_answer")
        asyncio.run(run_direction(fake, items, direction))
        assert len(fake.seen_items) == 1
        # The skipped item's triplets carry no verdict
        assert f"{NAMESPACE}_verdict" not in items[0][KG_KEY][0]

    def test_extraction_error_marker_carried_into_shadow(self):
        fake = FakeCheckingService()
        items = [_item([], gt_answer="gt", ext_extraction_error="parse_failure")]
        direction = Direction(name="answer2response", kg_key=KG_KEY,
                              reference_key="gt_answer")
        asyncio.run(run_direction(fake, items, direction))
        assert fake.seen_items[0]["ext_extraction_error"] == "parse_failure"

    def test_flat_without_reference_key_raises(self):
        fake = FakeCheckingService()
        direction = Direction(name="broken", kg_key=KG_KEY)
        with pytest.raises(ValueError):
            asyncio.run(run_direction(fake, [_item([_trip()])], direction))

    def test_filter_error_is_swallowed(self):
        """'Nothing to check' is a normal per-direction outcome."""
        fake = FakeCheckingService(raise_filter_error=True)
        items = [_item([_trip()], gt_answer="gt")]
        direction = Direction(name="answer2response", kg_key=KG_KEY,
                              reference_key="gt_answer")
        asyncio.run(run_direction(fake, items, direction))  # must not raise


# ── Matrix mode ──────────────────────────────────────────────────────────────

class TestMatrixMode:

    def test_verdicts_folded_as_doc_id_dicts(self):
        fake = FakeCheckingService()
        items = [_item(
            [_trip()],
            retrieved_context=[
                {"doc_id": "000", "text": "chunk zero"},
                {"doc_id": "001", "text": "chunk one"},
            ],
        )]
        direction = Direction(name="retrieved2response", kg_key=KG_KEY,
                              per_chunk=True)
        asyncio.run(run_direction(fake, items, direction))
        triplet = items[0][KG_KEY][0]
        assert triplet[f"{NAMESPACE}_verdicts"] == {
            "000": "Entailment", "001": "Entailment",
        }
        # Matrix mode must NOT leave a flat verdict on the originals
        assert f"{NAMESPACE}_verdict" not in triplet

    def test_bare_string_chunks_get_synthesized_ids(self):
        fake = FakeCheckingService()
        items = [_item([_trip()], retrieved_context=["chunk A", "chunk B"])]
        direction = Direction(name="retrieved2response", kg_key=KG_KEY,
                              per_chunk=True)
        asyncio.run(run_direction(fake, items, direction))
        assert set(items[0][KG_KEY][0][f"{NAMESPACE}_verdicts"]) == {"000", "001"}

    def test_one_shadow_item_per_item_chunk_pair(self):
        fake = FakeCheckingService()
        items = [
            _item([_trip()], retrieved_context=["c1", "c2"]),
            _item([_trip("x", "y", "z")], retrieved_context=["c3"]),
        ]
        direction = Direction(name="retrieved2response", kg_key=KG_KEY,
                              per_chunk=True)
        asyncio.run(run_direction(fake, items, direction))
        assert len(fake.seen_items) == 3

    def test_error_causes_folded_separately(self):
        """Null-verdict causes land in the {namespace}_errors dict."""
        class ErrorService(FakeCheckingService):
            async def run(self, data):
                self.seen_items = data
                for item in data:
                    for triplet in item.get(self.kg_key, []):
                        triplet[self.verdict_key] = None
                        triplet[self.checker_error_key] = "parse_failure"
                return data

        fake = ErrorService()
        items = [_item([_trip()], retrieved_context=["c1"])]
        direction = Direction(name="retrieved2response", kg_key=KG_KEY,
                              per_chunk=True)
        asyncio.run(run_direction(fake, items, direction))
        triplet = items[0][KG_KEY][0]
        assert triplet[f"{NAMESPACE}_verdicts"] == {"000": None}
        assert triplet[f"{NAMESPACE}_errors"] == {"000": "parse_failure"}

    def test_items_without_chunks_are_skipped(self):
        fake = FakeCheckingService()
        items = [_item([_trip()])]  # no retrieved_context
        direction = Direction(name="retrieved2response", kg_key=KG_KEY,
                              per_chunk=True)
        asyncio.run(run_direction(fake, items, direction))
        assert fake.seen_items == []
        assert f"{NAMESPACE}_verdicts" not in items[0][KG_KEY][0]
