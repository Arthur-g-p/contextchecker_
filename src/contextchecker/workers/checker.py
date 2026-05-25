"""
Checker worker — async execution unit for claim entailment checking.

Supports two modes:
- Single: one claim + reference → one verdict  (check_batch)
- Joint:  N claims + reference → N verdicts     (check_joint)

Takes a claim + reference, sends to an LLM, returns a verdict.
Owns no validation, filtering, or orchestration logic.
The service layer handles all of that before calling us.
"""

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
    """LLM response schema for single-claim checker prompt."""
    verdict: Verdict


class JointVerdictItem(BaseModel):
    """One claim's verdict in a joint checking response."""
    claim_id: int
    verdict: Verdict


class JointCheckResult(BaseModel):
    """LLM response schema for joint checker prompt — multiple claims at once."""
    verdicts: list[JointVerdictItem] = Field(default_factory=list)


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

    def _build_messages(self, claim: str, reference: str) -> list[dict]:
        """Build chat messages for a single check call."""
        prompt = format_prompt(self._prompt_template, {
            "claim": claim,
            "reference": reference,
        })
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    async def check(self, payload: CheckingPayload) -> Verdict:
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
        return parsed.verdict

    async def check_batch(
        self, payloads: list[CheckingPayload]
    ) -> list[Verdict | None]:
        """Check multiple claims concurrently (single mode — 1 call per claim).

        Returns one Verdict per payload. Failed items get None
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

        results: list[Verdict | None] = []
        for raw in raw_responses:
            if isinstance(raw, Exception):
                logger.warning("Check failed for claim: %s", raw)
                results.append(None)
                continue
            try:
                parsed = CheckResult.model_validate_json(raw)
                results.append(parsed.verdict)
            except Exception as exc:
                logger.warning("Failed to parse check result: %s", exc)
                results.append(None)

        return results

    # ── Joint mode ───────────────────────────────────────────────

    def _build_joint_messages(
        self, numbered_claims: list[tuple[int, str]], reference: str
    ) -> list[dict]:
        """Build chat messages for a joint check call.

        Args:
            numbered_claims: list of (claim_id, claim_text) tuples.
            reference: the reference passage.
        """
        claims_block = "\n".join(
            f"[{cid}] {text}" for cid, text in numbered_claims
        )
        prompt = format_prompt(self._joint_prompt_template, {
            "claims": claims_block,
            "reference": reference,
        })
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    async def check_joint(
        self,
        numbered_claims: list[tuple[int, str]],
        reference: str,
    ) -> dict[int, Verdict | None]:
        """Check multiple claims in a single LLM call (joint mode).

        Args:
            numbered_claims: list of (claim_id, claim_text) tuples.
            reference: the reference passage.

        Returns:
            Dict mapping claim_id → Verdict. Missing IDs (gaps) get None.
        """
        expected_ids = {cid for cid, _ in numbered_claims}

        messages = self._build_joint_messages(numbered_claims, reference)
        raw = await self.client.generate(
            messages=messages,
            schema=JointCheckResult,
            task="check",
        )

        # Parse the response
        try:
            parsed = JointCheckResult.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Failed to parse joint check result: %s", exc)
            return {cid: None for cid in expected_ids}

        # Map claim_id → verdict, tracking gaps
        result: dict[int, Verdict | None] = {}
        for item in parsed.verdicts:
            if item.claim_id in expected_ids:
                result[item.claim_id] = item.verdict
            else:
                logger.debug(
                    "Joint check returned unexpected claim_id %d — ignoring.",
                    item.claim_id,
                )

        # Fill gaps with None
        for cid in expected_ids:
            if cid not in result:
                logger.debug("Joint check gap: claim_id %d missing from response.", cid)
                result[cid] = None

        return result

    # TODO: retry pass for parse errors
    # TODO: wire to stats tracking
