"""
Extractor worker — async execution unit for knowledge graph extraction.

Takes text in, sends it to an LLM, parses triplets out.
Owns no validation, filtering, or orchestration logic.
The service layer handles all of that before calling us.
"""

from pydantic import BaseModel

from contextchecker.llmclient import LLMClient
from contextchecker.models import ExtractionPayload
from contextchecker.exceptions import LLMClientError, ParsingError
from contextchecker.utils import format_prompt
from contextchecker import settings

logger = settings.get_logger(__name__)


# ── LLM Response Schemas (Pydantic — structured output) ─────────────────────

class Triplet(BaseModel):
    """A single (subject, predicate, object) fact extracted from text."""
    subject: str
    predicate: str
    object: str


class ExtractionResult(BaseModel):
    """LLM response schema for the extraction prompt."""
    triplets: list[Triplet]


# ── Worker ───────────────────────────────────────────────────────────────────

class Extractor:
    """
    Async extractor. Receives text, calls LLM, returns parsed triplets.

    Stateless beyond its LLMClient — all orchestration, validation, and
    filtering live in the extraction service.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        concurrency: int = 10,
    ):
        self.model = model
        self.client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
        )
        self._prompt_template = settings.PROMPTS["extractor_prompt"]

    def _build_messages(self, text: str) -> list[dict]:
        """Build chat messages for a single extraction call."""
        prompt = format_prompt(self._prompt_template, {"text": text})
        return [
            {"role": "system", "content": "Extract knowledge triplets."},
            {"role": "user", "content": prompt},
        ]

    async def extract(self, payload: ExtractionPayload) -> list[Triplet]:
        """Extract triplets from a single text.

        Raises ParsingError if the LLM returns unparseable output.
        Raises LLMClientError on network / API failures.
        """
        messages = self._build_messages(payload.text)
        raw = await self.client.generate(
            messages=messages,
            schema=ExtractionResult,
            task="extract",
        )
        try:
            parsed = ExtractionResult.model_validate_json(raw)
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse extraction result: {exc}"
            ) from exc
        return parsed.triplets

    async def extract_batch(
        self, payloads: list[ExtractionPayload]
    ) -> list[list[Triplet]]:
        """Extract triplets from multiple texts concurrently.

        Returns one list of triplets per payload. Failed items get an
        empty list so the caller always gets len(payloads) results.
        """
        tasks = [
            {
                "messages": self._build_messages(p.text),
                "schema": ExtractionResult,
                "temperature": 0.0,
            }
            for p in payloads
        ]

        raw_responses = await self.client.generate_batch(
            tasks, description="Extracting", task="extract",
        )

        results: list[list[Triplet]] = []
        for raw in raw_responses:
            # generate_batch returns Union[str, LLMError] per item
            if isinstance(raw, Exception):
                logger.warning("Extraction failed for item: %s", raw)
                results.append([])
                continue
            try:
                parsed = ExtractionResult.model_validate_json(raw)
                results.append(parsed.triplets)
            except Exception as exc:
                logger.warning("Failed to parse extraction result: %s", exc)
                results.append([])

        return results

    # TODO: retry pass for parse errors (vanilla prompt fallback)
    # TODO: wire to stats tracking once stats ownership is decided
