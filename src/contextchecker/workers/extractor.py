"""
Extractor worker — async execution unit for knowledge graph extraction.

Takes text in, sends it to an LLM, parses triplets out.
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

from pydantic import BaseModel

from contextchecker.llmclient import LLMClient
from contextchecker.models import ExtractionPayload
from contextchecker.exceptions import ContextTooLongError, ContentPolicyError
from contextchecker.stats import PhaseStats, RoundResult
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


# ── Retry configuration ─────────────────────────────────────────────────────

@dataclass
class RetryRoundConfig:
    """Config for one retry round. Prompt can be 'standard' or 'vanilla'."""
    temperature: float = 0.3
    prompt: str = "standard"   # "standard" → _build_messages, "vanilla" → _build_retry_messages


# Default: 2 retry rounds. Round 1 same prompt with higher temp, Round 2 vanilla.
DEFAULT_RETRY_ROUNDS = [
    RetryRoundConfig(temperature=0.3, prompt="standard"),
    RetryRoundConfig(temperature=0.5, prompt="vanilla"),
]


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
        self.last_stats: PhaseStats | None = None

        # Vanilla prompt for retry — falls back to auto-generated schema example
        # if "extractor_prompt_vanilla" is not in prompt_map.json
        self._vanilla_prompt = settings.PROMPTS.get("extractor_prompt_vanilla", None)

        self._retry_rounds = DEFAULT_RETRY_ROUNDS

    # ── Message builders ─────────────────────────────────────────

    def _build_messages(self, text: str) -> list[dict]:
        """Build chat messages for a single extraction call."""
        prompt = format_prompt(self._prompt_template, {"text": text})
        return [
            {"role": "system", "content": "Extract knowledge triplets."},
            {"role": "user", "content": prompt},
        ]

    def _build_retry_messages(self, text: str) -> list[dict]:
        """Build chat messages for the vanilla retry pass.

        Uses a dedicated vanilla prompt if available in prompt_map.json,
        otherwise reuses the standard prompt — the LLMClient will auto-append
        a compact schema example when no response_format is supported.
        """
        template = self._vanilla_prompt or self._prompt_template
        prompt = format_prompt(template, {"text": text})
        return [
            {"role": "system", "content": "Extract knowledge triplets."},
            {"role": "user", "content": prompt},
        ]

    def _build_task(self, payload: ExtractionPayload, round_config: RetryRoundConfig | None = None) -> dict:
        """Build a generate_batch task dict for one item.

        Uses the round config to determine prompt and temperature.
        None means first pass (standard prompt, temp 0.0).
        """
        if round_config is None:
            # First pass — structured, low temperature
            messages = self._build_messages(payload.text)
            temperature = 0.0
        elif round_config.prompt == "vanilla":
            messages = self._build_retry_messages(payload.text)
            temperature = round_config.temperature
        else:
            messages = self._build_messages(payload.text)
            temperature = round_config.temperature

        return {
            "messages": messages,
            "schema": ExtractionResult,
            "temperature": temperature,
        }

    # ── Single-item extraction ───────────────────────────────────

    async def extract(self, payload: ExtractionPayload) -> list[Triplet]:
        """Extract triplets from a single text.

        Raises ContextTooLongError, ContentPolicyError, or ParsingError
        on per-item failures. Raises LLMClientError on fatal failures.
        """
        messages = self._build_messages(payload.text)
        raw = await self.client.generate(
            messages=messages,
            schema=ExtractionResult,
            task="extract",
        )
        parsed = ExtractionResult.model_validate_json(raw)
        return parsed.triplets

    # ── Batch extraction with retry ──────────────────────────────

    async def extract_batch(
        self, payloads: list[ExtractionPayload], description: str | None = None,
    ) -> list[list[Triplet]]:
        """Extract triplets from multiple texts concurrently.

        Returns one list of triplets per payload. Failed items get an
        empty list — the caller always gets len(payloads) results.
        Stats are stored on self.last_stats after completion.
        *description* labels the progress bar (pipelines pass their
        section label); retry rounds keep their own labels.

        Fatal errors (auth, connection) propagate — not caught here.
        """
        stats = PhaseStats()
        self.last_stats = stats

        # ── First pass ───────────────────────────────────────
        tasks = [self._build_task(p) for p in payloads]

        raw_responses = await self.client.generate_batch(
            tasks, description=description or "Extracting", task="extract",
        )
        stats.http_requests += self.client.last_batch_requests
        stats.first_pass_count = len(tasks)

        results, retry_indices = self._classify(raw_responses, stats)
        stats.first_pass_ok = stats.success + stats.empty

        # ── Retry loop ───────────────────────────────────────
        for round_num, round_config in enumerate(self._retry_rounds):
            if not retry_indices:
                break

            logger.info("   ♻️  Round %d: retrying %d items (temp=%.1f, prompt=%s)...",
                        round_num + 1, len(retry_indices),
                        round_config.temperature, round_config.prompt)

            retry_tasks = [
                self._build_task(payloads[i], round_config)
                for i in retry_indices
            ]

            raw_retries = await self.client.generate_batch(
                retry_tasks, description=f"Retry round {round_num + 1}", task="extract",
            )
            stats.http_requests += self.client.last_batch_requests

            round_result, retry_indices = self._apply_retries(
                raw_retries, retry_indices, results, stats
            )
            stats.rounds.append(round_result)

        # Whatever never produced a valid response — after all retry rounds
        # (or with retries disabled) — is a permanent parse failure.
        for i in retry_indices:
            stats.error_causes[i] = "parse_failure"

        return results

    # ── Classification (pure logic, unit-testable) ───────────────

    def _classify(
        self, responses: list, stats: PhaseStats,
    ) -> tuple[list[list[Triplet]], list[int]]:
        """Sort batch results into successes, permanent failures, and retryable.

        Returns (results, retry_indices) where results[i] is the triplet
        list for input i, and retry_indices lists positions that failed
        with a retryable error (parse errors).
        """
        results: list[list[Triplet]] = [[] for _ in responses]
        retry_indices: list[int] = []

        for i, raw in enumerate(responses):
            # Permanent per-item failures — empty result, no retry
            if isinstance(raw, ContextTooLongError):
                stats.context_too_long += 1
                stats.error_causes[i] = "context_too_long"
                continue
            if isinstance(raw, ContentPolicyError):
                stats.content_policy += 1
                stats.error_causes[i] = "content_policy"
                continue

            # Any other error type — retryable
            if isinstance(raw, Exception):
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)
                continue

            # Success — parse the JSON
            try:
                parsed = ExtractionResult.model_validate_json(raw)
                results[i] = parsed.triplets
                if parsed.triplets:
                    stats.success += 1
                    stats.total_items += len(parsed.triplets)
                else:
                    stats.empty += 1
            except Exception:
                stats.parse_error += 1
                retry_indices.append(i)
                stats.failed_indices.append(i)

        return results, retry_indices

    def _apply_retries(
        self,
        responses: list,
        indices: list[int],
        results: list[list[Triplet]],
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
                parsed = ExtractionResult.model_validate_json(raw)
                results[original_idx] = parsed.triplets
                round_result.recovered += 1
                if parsed.triplets:
                    stats.success += 1
                    stats.total_items += len(parsed.triplets)
                else:
                    stats.empty += 1
            except Exception:
                round_result.still_failed += 1
                remaining.append(original_idx)

        return round_result, remaining
