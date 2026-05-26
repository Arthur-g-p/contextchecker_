"""
Checker worker — async execution unit for claim entailment checking.

Supports two modes:
- Single: one claim + reference → one verdict  (check_batch)
- Joint:  N claims + reference → N verdicts     (check_joint_batch)

Takes a claim + reference, sends to an LLM, returns a verdict.
Owns no validation, filtering, or orchestration logic.
The service layer handles all of that before calling us.
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from contextchecker.llmclient import LLMClient
from contextchecker.models import CheckingPayload
from contextchecker.exceptions import ParsingError
from contextchecker.utils import format_prompt
from contextchecker import settings

logger = settings.get_logger(__name__)


# ── LLM Response Schemas (Pydantic — structured output) ─────────────────────

class Verdict(str, Enum):
    """Entailment verdict for a single claim against a reference."""
    ENTAILMENT = "Entailment"
    CONTRADICTION = "Contradiction"
    NEUTRAL = "Neutral"


class CheckResult(BaseModel):
    """LLM response schema for single-claim checker prompt.

    Explanation first — forces chain-of-thought reasoning before
    the model commits to a verdict. Improves accuracy.
    """
    explanation: str
    verdict: Verdict


class JointVerdictItem(BaseModel):
    """One claim's verdict in a joint checking response."""
    claim_id: int
    explanation: str
    verdict: Verdict


class JointCheckResult(BaseModel):
    """LLM response schema for joint checker prompt — multiple claims at once."""
    verdicts: list[JointVerdictItem] = Field(default_factory=list)


# ── Worker return type ───────────────────────────────────────────────────────

@dataclass
class ClaimVerdict:
    """Result for a single claim — verdict + explanation.

    This is the typed contract between the worker and the service.
    """
    verdict: Verdict | None
    explanation: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_reference(reference: list[str]) -> str:
    """Join a list of reference passages into a single string for the prompt.

    Each passage is numbered for clarity.
    """
    if len(reference) == 1:
        return reference[0]
    return "\n".join(
        f"[Passage {i+1}] {passage}" for i, passage in enumerate(reference)
    )


def _reference_word_count(reference: list[str]) -> int:
    """Total word count across all reference passages."""
    return sum(len(p.split()) for p in reference)


# ── Worker ───────────────────────────────────────────────────────────────────

class Checker:
    """
    Async checker. Receives claims + reference, calls LLM, returns verdicts.

    Supports single mode (one claim per call) and joint mode (N claims
    per call with ID-based matching).

    Stateless beyond its LLMClient — all orchestration, validation, and
    filtering live in the checking service.
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
        self._prompt_template = settings.PROMPTS["checker_prompt"]
        self._joint_prompt_template = settings.PROMPTS["checker_prompt_joint"]

    # ── Single mode ──────────────────────────────────────────────

    def _build_messages(self, claim: str, reference: list[str]) -> list[dict]:
        """Build chat messages for a single check call."""
        prompt = format_prompt(self._prompt_template, {
            "claim": claim,
            "reference": _format_reference(reference),
        })
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    async def check(self, payload: CheckingPayload) -> ClaimVerdict:
        """Check a single claim against its reference.

        Raises ParsingError if the LLM returns unparseable output.
        """
        messages = self._build_messages(payload.claim, payload.reference)
        raw = await self.client.generate(
            messages=messages,
            schema=CheckResult,
            task="check",
        )
        try:
            parsed = CheckResult.model_validate_json(raw)
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse check result: {exc}"
            ) from exc
        return ClaimVerdict(verdict=parsed.verdict, explanation=parsed.explanation)

    async def check_batch(
        self, payloads: list[CheckingPayload]
    ) -> list[ClaimVerdict]:
        """Check multiple claims concurrently (single mode — 1 call per claim).

        Returns one ClaimVerdict per payload. Failed items get None verdict
        so the caller always gets len(payloads) results.
        """
        tasks = [
            {
                "messages": self._build_messages(p.claim, p.reference),
                "schema": CheckResult,
                "temperature": 0.0,
            }
            for p in payloads
        ]

        raw_responses = await self.client.generate_batch(
            tasks, description="Checking", task="check",
        )

        results: list[ClaimVerdict] = []
        for raw in raw_responses:
            if isinstance(raw, Exception):
                logger.warning("Check failed for claim: %s", raw)
                results.append(ClaimVerdict(verdict=None))
                continue
            try:
                parsed = CheckResult.model_validate_json(raw)
                results.append(ClaimVerdict(
                    verdict=parsed.verdict,
                    explanation=parsed.explanation,
                ))
            except Exception as exc:
                logger.warning("Failed to parse check result: %s", exc)
                results.append(ClaimVerdict(verdict=None))

        return results

    # ── Joint mode ───────────────────────────────────────────────

    def _build_joint_messages(
        self, numbered_claims: list[tuple[int, str]], reference: list[str]
    ) -> list[dict]:
        """Build chat messages for a joint check call.

        Args:
            numbered_claims: list of (claim_id, claim_text) tuples.
            reference: list of reference passages.
        """
        claims_block = "\n".join(
            f"[{cid}] {text}" for cid, text in numbered_claims
        )
        prompt = format_prompt(self._joint_prompt_template, {
            "claims": claims_block,
            "reference": _format_reference(reference),
        })
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    async def check_joint_batch(
        self,
        chunks: list[tuple[list[tuple[int, str]], list[str]]],
    ) -> list[dict[int, ClaimVerdict]]:
        """Check multiple joint chunks concurrently via generate_batch.

        Each chunk is (numbered_claims, reference). All chunks are sent
        as a single batch through generate_batch (→ tqdm progress bar
        + concurrency).

        Args:
            chunks: list of (numbered_claims, reference) tuples.
                    numbered_claims: list of (claim_id, claim_text).
                    reference: list of reference passages.

        Returns:
            List of dicts, one per chunk. Each dict maps
            claim_id → ClaimVerdict. Missing IDs (gaps) get None verdict.
        """
        tasks = [
            {
                "messages": self._build_joint_messages(numbered, ref),
                "schema": JointCheckResult,
                "temperature": 0.0,
            }
            for numbered, ref in chunks
        ]

        raw_responses = await self.client.generate_batch(
            tasks, description="Checking (joint)", task="check",
        )

        results: list[dict[int, ClaimVerdict]] = []
        for raw, (numbered, _) in zip(raw_responses, chunks):
            expected_ids = {cid for cid, _ in numbered}

            if isinstance(raw, Exception):
                logger.warning("Joint check failed for chunk: %s", raw)
                results.append({cid: ClaimVerdict(verdict=None) for cid in expected_ids})
                continue

            try:
                parsed = JointCheckResult.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Failed to parse joint check result: %s", exc)
                results.append({cid: ClaimVerdict(verdict=None) for cid in expected_ids})
                continue

            # Map claim_id → ClaimVerdict, tracking gaps
            chunk_result: dict[int, ClaimVerdict] = {}
            for item in parsed.verdicts:
                if item.claim_id in expected_ids:
                    chunk_result[item.claim_id] = ClaimVerdict(
                        verdict=item.verdict,
                        explanation=item.explanation,
                    )
                else:
                    logger.debug(
                        "Joint check returned unexpected claim_id %d — ignoring.",
                        item.claim_id,
                    )

            # Fill gaps with None
            for cid in expected_ids:
                if cid not in chunk_result:
                    logger.debug("Joint check gap: claim_id %d missing from response.", cid)
                    chunk_result[cid] = ClaimVerdict(verdict=None)

            results.append(chunk_result)

        return results

    # TODO: retry pass for parse errors
    # TODO: wire to stats tracking
