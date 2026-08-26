"""
Atomizer worker — async execution unit for triplet atomization.

Takes a structured triplet (subject, predicate, object + response context),
sends it to an LLM, returns atomic triplets.
Owns no validation, filtering, or orchestration logic.
The service layer handles all of that before calling us.

Error handling:
    generate_batch() returns list[str | WorkerError].
    Per-item errors are VALUES in the list, not raised exceptions.
    Fatal errors (auth, connection) propagate — the worker doesn't catch them.
    This worker classifies per-item errors, retries parse failures in
    configurable rounds, and returns clean results with stats on self.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from contextchecker.llmclient import LLMClient
from contextchecker.models import AtomizationPayload
from contextchecker.exceptions import (
    ContextTooLongError, ContentPolicyError, LLMTimeoutError, FinishReasonLengthError,
)
from contextchecker.stats import PhaseStats, RoundResult
from contextchecker.utils import format_prompt
from contextchecker import settings

logger = settings.get_logger(__name__)


# ── LLM Response Schemas (Pydantic — structured output) ─────────────────────

class AtomicTriplet(BaseModel):
    """A single atomic (subject, predicate, object) fact."""
    subject: str
    predicate: str
    object: str


class AtomizationDecision(BaseModel):
    """LLM response schema — reasoning FIRST (chain-of-thought), then the decision.

    Field order matters: the model must reason BEFORE committing, which
    suppresses spurious splits and context-bleed.
      - is_atomic True  → keep the ORIGINAL triplet unchanged (service ignores `split`).
      - is_atomic False → `split` holds the atomic facts the original decomposes into.
    """
    reasoning: str
    is_atomic: bool
    split: list[AtomicTriplet] = Field(default_factory=list)


# ── Retry configuration ─────────────────────────────────────────────────────

@dataclass
class RetryRoundConfig:
    """Config for one retry round. Prompt can be 'standard' or 'vanilla'."""
    temperature: float = 0.3
    prompt: str = "standard"


# Default: 2 retry rounds. Round 1 same prompt with higher temp, Round 2 vanilla.
DEFAULT_RETRY_ROUNDS = [
    RetryRoundConfig(temperature=0.3, prompt="standard"),
    RetryRoundConfig(temperature=0.5, prompt="vanilla"),
]


# ── Worker ───────────────────────────────────────────────────────────────────

class Atomizer:
    """
    Async atomizer. Receives structured triplets, calls LLM, returns atomic triplets.

    Stateless beyond its LLMClient — all orchestration, validation, and
    filtering live in the atomization service.
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
        self._prompt_template = settings.PROMPTS["atomizer_prompt"]
        self.last_stats: PhaseStats | None = None

        # Vanilla prompt for retry — falls back to the standard template if
        # "atomizer_prompt_vanilla" is not in prompt_map.json
        self._vanilla_prompt = settings.PROMPTS.get("atomizer_prompt_vanilla", None)

        self._retry_rounds = DEFAULT_RETRY_ROUNDS

    # ── Message builders ─────────────────────────────────────────

    def _build_messages(
        self, payload: AtomizationPayload, prompt_type: str = "standard",
    ) -> list[dict]:
        """Build chat messages for a single atomization call.

        Substitutes structured S/P/O + response into the prompt template
        so the LLM sees labeled roles, never an ambiguous joined string.
        """
        template = self._prompt_template
        if prompt_type == "vanilla" and self._vanilla_prompt:
            template = self._vanilla_prompt

        prompt = format_prompt(template, {
            "subject": payload.subject,
            "predicate": payload.predicate,
            "object": payload.object,
            "response": "No context available",
        })
        return [
            {"role": "system", "content": "Split compound triplets into atomic facts. Preserve subject/predicate/object roles exactly."},
            {"role": "user", "content": prompt},
        ]

    def _build_task(
        self, payload: AtomizationPayload, round_config: RetryRoundConfig | None = None,
    ) -> dict:
        """Build a generate_batch task dict for one item.

        Uses the round config to determine prompt and temperature.
        None means first pass (standard prompt, temp 0.0).
        """
        if round_config is None:
            messages = self._build_messages(payload)
            temperature = 0.0
        else:
            messages = self._build_messages(payload, round_config.prompt)
            temperature = round_config.temperature

        return {
            "messages": messages,
            "schema": AtomizationDecision,
            "temperature": temperature,
        }

    # ── Batch atomization with retry ─────────────────────────────

    async def atomize_batch(
        self, payloads: list[AtomizationPayload], description: str | None = None,
    ) -> list[AtomizationDecision]:
        """Atomize multiple triplets concurrently.

        *description* labels the progress bar (pipelines pass their
        section label).

        Returns one AtomizationDecision per payload (reasoning + is_atomic +
        split). Failed items get a keep-original fallback decision — the
        caller always gets len(payloads) results and decides how to apply them.
        Stats are stored on self.last_stats after completion.

        Fatal errors (auth, connection) propagate — not caught here.
        """
        stats = PhaseStats()
        self.last_stats = stats

        # Build fallback decisions (keep original) so failed items stay unchanged
        originals = self._build_fallbacks(payloads)

        # ── First pass ───────────────────────────────────────
        tasks = [self._build_task(p) for p in payloads]

        raw_responses = await self.client.generate_batch(
            tasks, description=description or "Atomizing", task="atomize",
        )
        stats.http_requests += self.client.last_batch_requests
        stats.first_pass_count = len(tasks)

        results, retry_indices = self._classify(raw_responses, originals, stats)
        stats.first_pass_ok = stats.success + stats.empty

        # ── Retry loop ───────────────────────────────────────
        for round_num, round_config in enumerate(self._retry_rounds):
            if not retry_indices:
                break

            logger.info(
                "   ♻️  Round %d: retrying %d items (temp=%.1f, prompt=%s)...",
                round_num + 1, len(retry_indices), round_config.temperature,
                round_config.prompt,
            )

            retry_tasks = [
                self._build_task(payloads[i], round_config) for i in retry_indices
            ]

            raw_retries = await self.client.generate_batch(
                retry_tasks, description=f"Retry round {round_num + 1}", task="atomize",
            )
            stats.http_requests += self.client.last_batch_requests

            round_result, retry_indices = self._apply_retries(
                raw_retries, retry_indices, results, originals, stats,
            )
            stats.rounds.append(round_result)

        # Output triplet count (keep = 1, genuine split = N) for reporting.
        stats.total_items = sum(self._output_count(d) for d in results)
        return results

    @staticmethod
    def _output_count(decision: "AtomizationDecision") -> int:
        """Triplets this decision yields: a genuine split (>=2) keeps N, else 1."""
        if not decision.is_atomic and len(decision.split) >= 2:
            return len(decision.split)
        return 1

    # ── Classification (pure logic, unit-testable) ───────────────

    @staticmethod
    def _build_fallbacks(
        payloads: list[AtomizationPayload],
    ) -> list[AtomizationDecision]:
        """Build keep-original fallback decisions, one per payload.

        If atomization fails for a triplet, this decision keeps it unchanged
        (is_atomic=True, no split). The service flags it 'failed' via stats.
        """
        return [
            AtomizationDecision(
                reasoning="not processed (fallback)",
                is_atomic=True,
                split=[],
            )
            for _ in payloads
        ]

    def _classify(
        self,
        responses: list,
        originals: list[AtomizationDecision],
        stats: PhaseStats,
    ) -> tuple[list[AtomizationDecision], list[int]]:
        """Sort batch results into successes, permanent failures, and retryable.

        Returns (results, retry_indices) where results[i] is the decision
        for input i, and retry_indices lists positions that failed with a
        retryable error (parse errors).
        """
        results: list[AtomizationDecision] = list(originals)  # default: keep original
        retry_indices: list[int] = []

        for i, raw in enumerate(responses):
            # Permanent per-item failures — keep original, no retry
            if isinstance(raw, ContextTooLongError):
                stats.context_too_long += 1
                continue
            if isinstance(raw, FinishReasonLengthError):
                stats.finish_reason_length += 1
                continue
            if isinstance(raw, LLMTimeoutError):
                stats.timeout += 1
                continue
            if isinstance(raw, ContentPolicyError):
                stats.content_policy += 1
                continue

            # Any other error type — retryable
            if isinstance(raw, Exception):
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)
                continue

            # Success — parse the decision
            try:
                decision = AtomizationDecision.model_validate_json(raw)
                results[i] = decision
                stats.success += 1
            except Exception:
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)

        return results, retry_indices

    def _apply_retries(
        self,
        responses: list,
        indices: list[int],
        results: list[AtomizationDecision],
        originals: list[AtomizationDecision],
        stats: PhaseStats,
    ) -> tuple[RoundResult, list[int]]:
        """Merge retry results back into the main results list.

        Returns (round_result, remaining_indices) where remaining_indices
        are the items that still failed — they feed into the next round.
        """
        round_result = RoundResult()
        remaining: list[int] = []

        for raw, original_idx in zip(responses, indices):
            try:
                if isinstance(raw, Exception):
                    raise raw
                decision = AtomizationDecision.model_validate_json(raw)
                results[original_idx] = decision
                round_result.recovered += 1
                stats.success += 1
                # Recovered — no longer a failure, so the service won't flag it.
                if original_idx in stats.failed_indices:
                    stats.failed_indices.remove(original_idx)
            except Exception:
                round_result.still_failed += 1
                remaining.append(original_idx)

        return round_result, remaining
