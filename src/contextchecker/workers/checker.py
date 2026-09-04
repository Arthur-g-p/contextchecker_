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
from contextchecker.models import (
    DEFAULT_RETRY_ROUNDS,
    CheckingPayload,
    RetryRoundConfig,
)
from contextchecker.exceptions import (
    ParsingError, ContextTooLongError, ContentPolicyError, LLMTimeoutError,
    FinishReasonLengthError,
)
from contextchecker.stats import PhaseStats, RoundResult
from contextchecker.utils import format_prompt, prepare_plain_prompt
from contextchecker import settings

logger = settings.get_logger(__name__)


# ── LLM Response Schemas (Pydantic — structured output) ─────────────────────

class Verdict(str, Enum):
    """Verdict for a single claim against a reference."""
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
    ``error`` carries WHY the verdict is None ("context_too_long" |
    "finish_reason_length" | "content_policy" | "timeout" | "parse_failure")
    so a null verdict is never
    left open to interpretation downstream.
    """
    verdict: Verdict | None
    explanation: str | None = None
    error: str | None = None


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
        retry_rounds: list[RetryRoundConfig] | None = None,
        joint_prompt_key: str | None = None,
    ):
        self.model = model
        self.client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            concurrency=concurrency,
        )
        self._prompt_template = settings.PROMPTS["checker_prompt"]
        joint_key = joint_prompt_key or "checker_prompt_joint"
        self._joint_prompt_template = settings.PROMPTS[joint_key]
        key = "checker_prompt_plain"
        self._plain_prompt = prepare_plain_prompt(
            settings.PROMPTS.get(key), key, CheckResult, logger)
        # Derived from the key in use — a hardcoded one would swap the task when
        # joint_prompt_key is not the default.
        joint_plain_key = f"{joint_key}_plain"
        self._joint_plain_prompt = prepare_plain_prompt(
            settings.PROMPTS.get(joint_plain_key), joint_plain_key, JointCheckResult, logger)
        self.last_stats: PhaseStats | None = None

        self._retry_rounds = retry_rounds or DEFAULT_RETRY_ROUNDS

    # ── Single mode ──────────────────────────────────────────────

    def _build_messages(
        self, claim: str, reference: list[str], prompt_type: str = "standard"
    ) -> list[dict]:
        """Build chat messages for a single check call."""
        template = self._prompt_template
        if prompt_type == "plain" and self._plain_prompt:
            template = self._plain_prompt

        prompt = format_prompt(template, {
            "claim": claim,
            "reference": _format_reference(reference),
        })
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    def _plain_messages(self, claim: str, reference: list[str]) -> list[dict] | None:
        """None when no plain prompt exists and it is needed — the client WILL fail!"""
        if not self._plain_prompt:
            return None
        return self._build_messages(claim, reference, "plain")

    def _plain_joint_messages(
        self, numbered_claims: list[tuple[int, str]], reference: list[str],
        extra_vars: dict | None = None,
    ) -> list[dict] | None:
        """None when no plain joint prompt exists — see _plain_messages."""
        if not self._joint_plain_prompt:
            return None
        return self._build_joint_messages(numbered_claims, reference, "plain", extra_vars)

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
        self, payloads: list[CheckingPayload], description: str | None = None,
    ) -> list[ClaimVerdict]:
        """Check multiple claims concurrently (single mode — 1 call per claim).

        Returns one ClaimVerdict per payload. Failed items get None verdict
        so the caller always gets len(payloads) results. *description*
        labels the progress bar (pipelines pass their section label).
        """
        stats = PhaseStats()
        self.last_stats = stats

        # Initialize results list
        results: list[ClaimVerdict] = [ClaimVerdict(verdict=None) for _ in payloads]
        retry_indices: list[int] = []

        # ── First pass ───────────────────────────────────────
        tasks = []
        for p in payloads:
            tasks.append({
                "messages": self._build_messages(p.claim, p.reference),
                "plain_messages": self._plain_messages(p.claim, p.reference),
                "schema": CheckResult,
                "temperature": 0.0,
            })

        raw_responses = await self.client.generate_batch(
            tasks, description=description or "Checking", task="check",
        )
        stats.http_requests += self.client.last_batch_requests
        stats.first_pass_count = len(tasks)

        # Classify first pass
        for i, raw in enumerate(raw_responses):
            if isinstance(raw, ContextTooLongError):
                stats.context_too_long += 1
                results[i] = ClaimVerdict(verdict=None, error="context_too_long")
                continue
            if isinstance(raw, ContentPolicyError):
                stats.content_policy += 1
                results[i] = ClaimVerdict(verdict=None, error="content_policy")
                continue
            if isinstance(raw, FinishReasonLengthError):
                stats.finish_reason_length += 1
                results[i] = ClaimVerdict(verdict=None, error="finish_reason_length")
                continue
            if isinstance(raw, LLMTimeoutError):
                stats.timeout += 1
                results[i] = ClaimVerdict(verdict=None, error="timeout")
                continue

            if isinstance(raw, Exception):
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)
                continue

            try:
                parsed = CheckResult.model_validate_json(raw)
                results[i] = ClaimVerdict(
                    verdict=parsed.verdict,
                    explanation=parsed.explanation,
                )
                stats.success += 1
                stats.total_items += 1
            except Exception:
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)

        stats.first_pass_ok = stats.success

        # ── Retry loop ───────────────────────────────────────
        for round_num, round_config in enumerate(self._retry_rounds):
            if not retry_indices:
                break

            logger.info("   ♻️  Round %d: retrying %d items (temp=%.1f, prompt=%s)...",
                        round_num + 1, len(retry_indices),
                        round_config.temperature, round_config.prompt)

            retry_tasks = []
            for i in retry_indices:
                p = payloads[i]
                retry_tasks.append({
                    "messages": self._build_messages(p.claim, p.reference, round_config.prompt),
                    "plain_messages": self._plain_messages(p.claim, p.reference),
                    "schema": CheckResult,
                    "temperature": round_config.temperature,
                })

            raw_retries = await self.client.generate_batch(
                retry_tasks, description=f"Retry round {round_num + 1}", task="check",
            )
            stats.http_requests += self.client.last_batch_requests

            # Apply retries
            round_result = RoundResult()
            next_retry_indices: list[int] = []

            for raw, original_idx in zip(raw_retries, retry_indices):
                try:
                    if isinstance(raw, Exception):
                        raise raw
                    parsed = CheckResult.model_validate_json(raw)
                    results[original_idx] = ClaimVerdict(
                        verdict=parsed.verdict,
                        explanation=parsed.explanation,
                    )
                    round_result.recovered += 1
                    stats.success += 1
                    stats.total_items += 1
                except Exception:
                    round_result.still_failed += 1
                    next_retry_indices.append(original_idx)

            stats.rounds.append(round_result)
            retry_indices = next_retry_indices

        # Whatever never produced a valid verdict is a permanent parse failure.
        for i in retry_indices:
            results[i] = ClaimVerdict(verdict=None, error="parse_failure")

        return results

    # ── Joint mode ───────────────────────────────────────────────

    def _build_joint_messages(
        self,
        numbered_claims: list[tuple[int, str]],
        reference: list[str],
        prompt_type: str = "standard",
        extra_vars: dict | None = None,
    ) -> list[dict]:
        """Build chat messages for a joint check call.

        Args:
            numbered_claims: list of (claim_id, claim_text) tuples.
            reference: list of reference passages.
            prompt_type: standard or plain prompt template.
            extra_vars: additional template variables merged with
                        {claims, reference} before formatting.
        """
        template = self._joint_prompt_template
        if prompt_type == "plain" and self._joint_plain_prompt:
            template = self._joint_plain_prompt

        claims_block = "\n".join(
            f"[{cid}] {text}" for cid, text in numbered_claims
        )
        template_vars = {
            "claims": claims_block,
            "reference": _format_reference(reference),
        }
        if extra_vars:
            template_vars.update(extra_vars)

        prompt = format_prompt(template, template_vars)
        return [
            {"role": "system", "content": "You are a factual entailment judge."},
            {"role": "user", "content": prompt},
        ]

    async def check_joint_batch(
        self,
        chunks: list[tuple[list[tuple[int, str]], list[str]]],
        extra_vars_list: list[dict] | None = None,
        description: str | None = None,
    ) -> list[dict[int, ClaimVerdict]]:
        """Check multiple joint chunks concurrently via generate_batch.

        Each chunk is (numbered_claims, reference). All chunks are sent
        as a single batch through generate_batch (→ tqdm progress bar
        + concurrency).

        Args:
            chunks: list of (numbered_claims, reference) tuples.
                    numbered_claims: list of (claim_id, claim_text).
                    reference: list of reference passages.
            extra_vars_list: optional per-chunk extra template variables.
                    Must be same length as chunks if provided.
            description: progress-bar label (pipelines pass their
                    section label).

        Returns:
            List of dicts, one per chunk. Each dict maps
            claim_id → ClaimVerdict. Missing IDs (gaps) get None verdict.
        """
        stats = PhaseStats()
        self.last_stats = stats

        # Initialize results: list of dicts mapping claim_id -> ClaimVerdict
        results: list[dict[int, ClaimVerdict]] = [{} for _ in chunks]

        # Keep track of retryable claims per chunk: index -> set of failed claim_ids
        retryable_claims: dict[int, set[int]] = {}

        # ── First pass ───────────────────────────────────────
        tasks = []
        for i, (numbered, ref) in enumerate(chunks):
            ev = extra_vars_list[i] if extra_vars_list else None
            tasks.append({
                "messages": self._build_joint_messages(numbered, ref, extra_vars=ev),
                "plain_messages": self._plain_joint_messages(numbered, ref, extra_vars=ev),
                "schema": JointCheckResult,
                "temperature": 0.0,
            })

        raw_responses = await self.client.generate_batch(
            tasks, description=description or "Checking (joint)", task="check",
        )
        stats.http_requests += self.client.last_batch_requests
        stats.first_pass_count = len(tasks)

        # Classify first pass
        for i, raw in enumerate(raw_responses):
            numbered, ref = chunks[i]
            expected_ids = {cid for cid, _ in numbered}

            # Permanent per-item failures — fill with None, no retry
            if isinstance(raw, ContextTooLongError):
                stats.context_too_long += 1
                for cid in expected_ids:
                    results[i][cid] = ClaimVerdict(verdict=None, error="context_too_long")
                continue
            if isinstance(raw, ContentPolicyError):
                stats.content_policy += 1
                for cid in expected_ids:
                    results[i][cid] = ClaimVerdict(verdict=None, error="content_policy")
                continue
            if isinstance(raw, FinishReasonLengthError):
                stats.finish_reason_length += 1
                for cid in expected_ids:
                    results[i][cid] = ClaimVerdict(verdict=None, error="finish_reason_length")
                continue
            if isinstance(raw, LLMTimeoutError):
                stats.timeout += 1
                for cid in expected_ids:
                    results[i][cid] = ClaimVerdict(verdict=None, error="timeout")
                continue

            # Any other error type — retryable failure for the whole chunk
            if isinstance(raw, Exception):
                stats.parse_error += 1
                retryable_claims[i] = expected_ids.copy()
                stats.failed_indices.append(i)
                continue

            # Parse JSON response
            try:
                parsed = JointCheckResult.model_validate_json(raw)
                # Map claim_id -> ClaimVerdict
                chunk_result: dict[int, ClaimVerdict] = {}
                for item in parsed.verdicts:
                    if item.claim_id in expected_ids:
                        chunk_result[item.claim_id] = ClaimVerdict(
                            verdict=item.verdict,
                            explanation=item.explanation,
                        )
                        stats.success += 1
                        stats.total_items += 1
                    else:
                        logger.debug("Unexpected claim_id %d in response — ignoring.", item.claim_id)

                # Find any gaps (missing claim IDs from expected)
                gaps = expected_ids - chunk_result.keys()
                if gaps:
                    logger.debug("Chunk %d has gaps: %s", i, gaps)
                    stats.id_gaps += len(gaps)
                    stats.parse_error += 1
                    retryable_claims[i] = gaps
                    stats.failed_indices.append(i)

                # Save the successful ones
                results[i].update(chunk_result)

            except Exception:
                stats.parse_error += 1
                retryable_claims[i] = expected_ids.copy()
                stats.failed_indices.append(i)

        stats.first_pass_ok = (stats.first_pass_count - stats.context_too_long
                               - stats.content_policy - stats.timeout
                               - stats.finish_reason_length - stats.parse_error)

        # ── Retry loop ───────────────────────────────────────
        for round_num, round_config in enumerate(self._retry_rounds):
            if not retryable_claims:
                break

            logger.info("   ♻️  Round %d: retrying %d items (temp=%.1f, prompt=%s)...",
                        round_num + 1, len(retryable_claims),
                        round_config.temperature, round_config.prompt)

            # Build tasks only for the retryable claims
            retry_indices = sorted(retryable_claims.keys())
            retry_tasks = []
            for idx in retry_indices:
                numbered, ref = chunks[idx]
                failed_ids = retryable_claims[idx]
                filtered_numbered = [(cid, text) for cid, text in numbered if cid in failed_ids]

                retry_tasks.append({
                    "messages": self._build_joint_messages(
                        filtered_numbered, ref, round_config.prompt,
                        extra_vars=extra_vars_list[idx] if extra_vars_list else None,
                    ),
                    "plain_messages": self._plain_joint_messages(
                        filtered_numbered, ref,
                        extra_vars=extra_vars_list[idx] if extra_vars_list else None,
                    ),
                    "schema": JointCheckResult,
                    "temperature": round_config.temperature,
                })

            raw_retries = await self.client.generate_batch(
                retry_tasks, description=f"Retry round {round_num + 1}", task="check",
            )
            stats.http_requests += self.client.last_batch_requests

            # Apply retries
            round_result = RoundResult()
            next_retryable: dict[int, set[int]] = {}

            for idx, raw in zip(retry_indices, raw_retries):
                numbered, ref = chunks[idx]
                failed_ids = retryable_claims[idx]

                try:
                    if isinstance(raw, Exception):
                        raise raw
                    parsed = JointCheckResult.model_validate_json(raw)

                    # Map claim_id -> ClaimVerdict
                    chunk_result: dict[int, ClaimVerdict] = {}
                    for item in parsed.verdicts:
                        if item.claim_id in failed_ids:
                            chunk_result[item.claim_id] = ClaimVerdict(
                                verdict=item.verdict,
                                explanation=item.explanation,
                            )
                            stats.success += 1
                            stats.total_items += 1
                        else:
                            logger.debug("Unexpected claim_id %d in retry response — ignoring.", item.claim_id)

                    # Update results
                    results[idx].update(chunk_result)

                    # Find remaining gaps
                    still_missing = failed_ids - chunk_result.keys()
                    if still_missing:
                        logger.debug("Chunk %d still has gaps after retry: %s", idx, still_missing)
                        next_retryable[idx] = still_missing
                        round_result.still_failed += 1
                    else:
                        round_result.recovered += 1

                except Exception:
                    next_retryable[idx] = failed_ids
                    round_result.still_failed += 1

            stats.rounds.append(round_result)
            retryable_claims = next_retryable

        # Fill any remaining failures/gaps with None verdict so we always
        # return results for every claim. CTL/CP chunks were already filled
        # with their cause above, so everything left here never got a valid
        # response — a permanent parse failure.
        for idx, (numbered, _) in enumerate(chunks):
            expected_ids = {cid for cid, _ in numbered}
            for cid in expected_ids:
                if cid not in results[idx]:
                    results[idx][cid] = ClaimVerdict(verdict=None, error="parse_failure")

        return results
