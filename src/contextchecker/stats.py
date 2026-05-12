from dataclasses import dataclass, field
from typing import List
import threading

from contextchecker import settings

logger = settings.get_logger(__name__)


# ───────────────────────────────────────────────────────────────
#  PHASE STATS — per-batch-call outcome tracking
# ───────────────────────────────────────────────────────────────

@dataclass
class PhaseStats:
    """Tracks outcomes for a single LLM batch call. 
    Reusable across extractor, checker, or any batch caller."""
    success: int = 0
    prefilter: int = 0        # skipped before LLM call (e.g. abstention)
    valid_empty: int = 0      # LLM returned valid JSON but empty content
    context_too_long: int = 0 # ContextTooLongError — permanent
    content_policy: int = 0   # ContentPolicyError — permanent
    parse_error: int = 0      # LLMParseError OR model_validate_json fail — retryable
    total_items: int = 0      # domain-specific count (e.g. triplets, verdicts)

    # Retry results (filled after retry pass)
    recovered: int = 0        # items recovered on retry
    still_failed: int = 0     # items that failed again on retry
    id_gaps: int = 0          # claims where LLM skipped the claim_id (joint mode)

    # Which batch indices failed and are retryable
    failed_indices: List[int] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        return self.context_too_long + self.content_policy + self.parse_error


@dataclass
class TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_requests: int = 0
    total_errors: int = 0

    # Per-phase breakdown: phase_name -> {input, output, reasoning, requests, errors}
    _phases: dict = field(default_factory=dict)
    _current_phase: str = "default"

    # Thread-Lock for clean counting with async
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_phase(self, phase: str):
        """Set the current tracking phase (e.g. 'extraction', 'validation', 'checking')."""
        with self._lock:
            self._current_phase = phase
            if phase not in self._phases:
                self._phases[phase] = {
                    "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                    "requests": 0, "errors": 0
                }

    def update(self, usage: dict):
        if not usage:
            return
        with self._lock:
            inp = usage.get('prompt_tokens', 0)
            out = usage.get('completion_tokens', 0)

            # Reasoning tokens (o-series models)
            reasoning = 0
            details = usage.get('completion_tokens_details')
            if details:
                if isinstance(details, dict):
                    reasoning = details.get('reasoning_tokens', 0) or 0
                elif hasattr(details, 'reasoning_tokens'):
                    reasoning = details.reasoning_tokens or 0

            self.input_tokens += inp
            self.output_tokens += out
            self.reasoning_tokens += reasoning
            self.total_requests += 1

            # Update phase stats
            if self._current_phase in self._phases:
                p = self._phases[self._current_phase]
                p["input_tokens"] += inp
                p["output_tokens"] += out
                p["reasoning_tokens"] += reasoning
                p["requests"] += 1

    def log_error(self):
        with self._lock:
            self.total_errors += 1
            if self._current_phase in self._phases:
                self._phases[self._current_phase]["errors"] += 1

    def to_dict(self) -> dict:
        """Crash-safe dict for _meta output. Never raises."""
        try:
            result = {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
            }
            if self._phases:
                result["phases"] = dict(self._phases)
            return result
        except Exception:
            return {"input_tokens": None, "output_tokens": None,
                    "total_requests": None, "total_errors": None}

    def snapshot(self) -> dict:
        """Thread-safe snapshot of current values."""
        with self._lock:
            return self.to_dict()

    def __repr__(self) -> str:
        return f"TokenStats(reqs={self.total_requests}, in={self.input_tokens}, out={self.output_tokens})"


# Global instance — written to by LLMClient, read by services.
GLOBAL_STATS = TokenStats()


# ───────────────────────────────────────────────────────────────
#  SHARED LOGGING — reusable across all services
# ───────────────────────────────────────────────────────────────

def log_api_summary(
    pending: int,
    prefiltered: int,
    successful: int,
    failed: int,
) -> None:
    """Log 🌐 API-Request summary. Same format across all services.

    Uses logger.info() → silent for library users, pretty for CLI.
    """
    logger.info(" 🌐 API-Request summary:")
    logger.info("    %d (pending) input items", pending + prefiltered)
    entries = []
    if prefiltered > 0:
        entries.append(("🔇", f"{prefiltered} prefiltered"))
    entries.append(("✅", f"{successful} successful calls"))
    # TODO: add context_too_long, content_blocked from PhaseStats when wired
    if failed > 0:
        entries.append(("❌", f"{failed} failed"))
    for i, (icon, text) in enumerate(entries):
        is_last = i == len(entries) - 1
        prefix = "└─" if is_last else "├─"
        logger.info("     %s %s %s", prefix, icon, text)
    logger.info("")


def log_token_stats() -> None:
    """Log 📊 Token stats from GLOBAL_STATS. Same format across all services.

    Uses logger.info() → silent for library users, pretty for CLI.
    """
    stats = GLOBAL_STATS.snapshot()
    phases = stats.get("phases", {})

    logger.info("── Execution Stats ────────────────────────────────")
    logger.info("")
    logger.info(" 📊 Tokens")

    active = [(name, p) for name, p in phases.items() if p["requests"] > 0]
    for i, (phase, p) in enumerate(active):
        parts = [
            f"{p['requests']} reqs",
            f"{p['input_tokens']:,} in",
            f"{p['output_tokens']:,} out",
        ]
        if p.get("reasoning_tokens"):
            parts.append(f"🧠 {p['reasoning_tokens']:,}")
        prefix = "└─" if i == len(active) - 1 else "├─"
        logger.info(
            "     %s %s:%s%s",
            prefix, phase, " " * max(1, 9 - len(phase)), " · ".join(parts),
        )

    if not active:
        logger.info("     └─ (no token data)")
    logger.info("")