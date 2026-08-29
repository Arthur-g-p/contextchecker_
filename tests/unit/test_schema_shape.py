"""Unit tests for build_schema_shape.

Run with:
    pytest tests/unit/test_schema_shape.py -v
"""

import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from contextchecker.utils import build_schema_shape
from contextchecker.workers.checker import JointCheckResult
from contextchecker.workers.extractor import ExtractionResult


def shape(model) -> dict:
    return json.loads(build_schema_shape(model))


class Inner(BaseModel):
    a: str
    b: int


class TestShape:
    def test_scalars(self):
        class M(BaseModel):
            name: str
            n: int
            ok: bool

        assert shape(M) == {"name": "<string>", "n": "<integer>", "ok": "<boolean>"}

    def test_field_order_is_preserved(self):
        """Chain-of-thought schemas depend on it: reasoning before the verdict."""
        class M(BaseModel):
            reasoning: str
            verdict: str

        assert list(shape(M)) == ["reasoning", "verdict"]

    def test_nested_model_is_expanded(self):
        class M(BaseModel):
            inner: Inner

        assert shape(M) == {"inner": {"a": "<string>", "b": "<integer>"}}

    def test_list_of_models(self):
        class M(BaseModel):
            items: list[Inner]

        assert shape(M) == {"items": [{"a": "<string>", "b": "<integer>"}]}

    def test_enum_lists_its_values(self):
        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        class M(BaseModel):
            color: Color

        assert shape(M) == {"color": "red | blue"}

    def test_single_value_literal(self):
        """Pydantic emits const, not enum, for a one-value Literal."""
        class M(BaseModel):
            kind: Literal["only"]

        assert shape(M) == {"kind": "only"}

    def test_typed_dict(self):
        class M(BaseModel):
            mapping: dict[str, int]

        assert shape(M) == {"mapping": {"<key>": "<integer>"}}

    def test_tuple_enumerates_its_members(self):
        """Not a homogeneous array — arity and per-slot types both matter."""
        class M(BaseModel):
            pair: tuple[int, str]

        assert shape(M) == {"pair": ["<integer>", "<string>"]}

    def test_optional_shows_the_real_branch(self):
        class M(BaseModel):
            maybe: Optional[str]

        assert shape(M) == {"maybe": "<string>"}

    def test_self_reference_terminates(self):
        class Node(BaseModel):
            value: str
            children: list["Node"] = Field(default_factory=list)

        Node.model_rebuild()
        assert shape(Node) == {"value": "<string>", "children": ["<recursive>"]}

    def test_a_broken_schema_never_raises(self):
        class NotAModel:
            pass

        assert shape(NotAModel) == {"result": "<see prompt for format>"}


class TestRealSchemas:
    def test_joint_check_result(self):
        assert shape(JointCheckResult) == {
            "verdicts": [{
                "claim_id": "<integer>",
                "explanation": "<string>",
                "verdict": "Entailment | Contradiction | Neutral",
            }]
        }

    def test_extraction_result(self):
        assert shape(ExtractionResult) == {
            "triplets": [{
                "subject": "<string>",
                "predicate": "<string>",
                "object": "<string>",
            }]
        }
