"""
Extractor worker — the dumb execution unit.

Takes text in, sends it to an LLM, parses triplets out.
Owns no validation, filtering, or orchestration logic.
"""

from dataclasses import dataclass

from contextchecker.models import ExtractionPayload
from contextchecker import settings

logger = settings.get_logger(__name__)


@dataclass
class Triplet:
    """A single (subject, predicate, object) fact extracted from text."""
    subject: str
    predicate: str
    object: str


class Extractor:
    """
    Stateless extractor. Receives text, calls LLM, returns triplets.

    Async-native because it makes network calls (rule #12).
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def extract(self, payload: ExtractionPayload) -> list[Triplet]:
        """Extract triplets from a single text payload."""
        # TODO: wire to llmclient
        raise NotImplementedError

    async def extract_batch(self, payloads: list[ExtractionPayload]) -> list[list[Triplet]]:
        """Extract triplets from multiple payloads."""
        # TODO: gather concurrent calls
        raise NotImplementedError
