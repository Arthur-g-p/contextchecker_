from dataclasses import dataclass, field
from typing import List
import threading

from contextchecker import settings

logger = settings.get_logger(__name__)


# ───────────────────────────────────────────────────────────────
#  PHASE STATS — per-batch-call outcome tracking
# ───────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    """Outcome of a single retry round."""
    recovered: int = 0
    still_failed: int = 0


@dataclass
class PhaseStats:
    """Tracks outcomes for a single LLM batch call. 
    Reusable across extractor, checker, or any batch caller."""
    # First pass results
    success: int = 0              # parsed with content on ANY pass (cumulative)
    first_pass_ok: int = 0        # parsed on first attempt (set once, never modified)
    empty: int = 0                # parsed successfully but 0 items (e.g. empty triplets)
    context_too_long: int = 0     # ContextTooLongError — permanent
    content_policy: int = 0       # ContentPolicyError — permanent
    parse_error: int = 0          # initial retryable failure count (set in _classify)
    total_items: int = 0          # domain-specific count (e.g. triplets, verdicts)
    http_requests: int = 0        # total HTTP requests across all rounds

    first_pass_count: int = 0     # total items/tasks sent in the first pass

    # Per-round retry results (index 0 = round 1, etc.)
    rounds: list[RoundResult] = field(default_factory=list)

    # Checker-specific
    id_gaps: int = 0              # claims where LLM skipped the claim_id (joint mode)

    # Which batch indices failed and are retryable (debug aid)
    failed_indices: list[int] = field(default_factory=list)

    # Per-item permanent failure causes: batch index → cause.
    # Causes: "context_too_long" | "content_policy" | "parse_failure".
    # Filled by the worker so services can persist WHY an item failed
    # instead of leaving an uninterpretable empty result.
    error_causes: dict[int, str] = field(default_factory=dict)

    @property
    def permanently_failed(self) -> int:
        """Items that failed ALL retry rounds."""
        if self.rounds:
            return self.rounds[-1].still_failed
        return self.parse_error

    @property
    def total_permanent(self) -> int:
        """All permanently failed items (context + content + exhausted retries)."""
        return self.context_too_long + self.content_policy + self.permanently_failed

    @property
    def total_errors(self) -> int:
        """Backward-compat: total permanent errors."""
        return self.total_permanent


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
            self.total_requests += 1
            if self._current_phase in self._phases:
                p = self._phases[self._current_phase]
                p["errors"] += 1
                p["requests"] += 1

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

def log_api_parsing(
    pending: int,
    stats: PhaseStats,
) -> None:
    """Log 🌐 API & Parsing summary. Same format across all services.

    Shows first-pass results, permanent failures, retryable failures
    with per-round breakdown.
    """
    logger.info(" 🌐 API & Parsing:")
    logger.info("    %d tasks sent to LLM [%d HTTP requests]", pending, stats.http_requests)

    # ── Success on first attempt
    permanent = stats.context_too_long + stats.content_policy
    has_permanent = permanent > 0
    has_retryable = stats.parse_error > 0

    if has_permanent or has_retryable:
        logger.info("     ├─ ✅ %d parsed on first attempt", stats.first_pass_ok)
    else:
        logger.info("     └─ ✅ %d parsed on first attempt", stats.first_pass_ok)

    # ── Permanent failures (context too long / content policy)
    if has_permanent:
        parts = []
        if stats.context_too_long > 0:
            parts.append(f"{stats.context_too_long} context too long")
        if stats.content_policy > 0:
            parts.append(f"{stats.content_policy} content policy")
        prefix = "├─" if has_retryable else "└─"
        logger.info("     %s 🚫 %d permanent failures (%s)", prefix, permanent, ", ".join(parts))

    # ── Retryable failures with per-round breakdown
    if has_retryable:
        logger.info("     └─ ⚠️  %d retryable failures", stats.parse_error)
        total_rounds = len(stats.rounds)
        for i, round_result in enumerate(stats.rounds):
            is_last = (i == total_rounds - 1)
            if is_last and round_result.still_failed > 0:
                # Last round with remaining failures — show round then exhausted
                logger.info("          ├─ ♻️  Round %d: %d recovered | %d failed",
                            i + 1, round_result.recovered, round_result.still_failed)
                logger.info("          └─ ❌ %d exhausted after %d attempts",
                            round_result.still_failed, total_rounds + 1)
            elif is_last:
                # Last round, all recovered
                logger.info("          └─ ♻️  Round %d: %d recovered | %d failed",
                            i + 1, round_result.recovered, round_result.still_failed)
            else:
                logger.info("          ├─ ♻️  Round %d: %d recovered | %d failed",
                            i + 1, round_result.recovered, round_result.still_failed)
    logger.info("")


def log_multi_run_hint(runs: int) -> None:
    """Print the pre-flight hint for variance mode, once, before run 1.

    Cached responses make all runs identical and the variance meaningless —
    warn before the tokens are spent, not after."""
    logger.info(" 💡 %d runs requested — make sure your backend/proxy does not"
                " cache responses, or every run will be identical.", runs)


def log_variance_block(
    runs: int,
    means: dict,
    variance: dict,
    durations: list[float] | None = None,
    total_seconds: float | None = None,
) -> None:
    """Print ══ VARIANCE (N runs) ══: mean ± std [min, max] per metric,
    a time tree (per run + total), and a caching warning when every metric
    has zero spread. The math behind means/variance lives in
    utils.build_variance."""
    logger.info("")
    logger.info(settings.section_rule(f"VARIANCE ({runs} runs)", char="═"))
    logger.info("")
    logger.info(" 📊 Metrics  (mean ± std  [min, max])")
    keys = list(means.keys())
    for i, key in enumerate(keys):
        prefix = "└─" if i == len(keys) - 1 else "├─"
        v = variance[key]
        if means[key] is None:
            # Null in every run: the metric exists but was never computable.
            logger.info(
                "    %s %-24s n/a  (not computable in any of %d runs)",
                prefix, key + ":", runs,
            )
            continue
        partial = ""
        if v.get("n", runs) < runs:
            # Mean over fewer runs than executed — say so, or the mean
            # overstates its support.
            partial = f"  ({v['n']}/{runs} runs)"
        logger.info(
            "    %s %-24s %.3f ± %.3f   [%.3f, %.3f]%s",
            prefix, key + ":", means[key], v["std"], v["min"], v["max"], partial,
        )
    if durations:
        logger.info("")
        total = total_seconds if total_seconds is not None else sum(durations)
        logger.info(" ⏱️  %-10s %6.1fs", "Total:", total)
        for i, d in enumerate(durations, 1):
            prefix = "└─" if i == len(durations) else "├─"
            logger.info("    %s %-10s %6.1fs", prefix, f"run {i}:", d)
    stds = [v["std"] for v in variance.values() if v["std"] is not None]
    if stds and all(std == 0 for std in stds):
        logger.info("")
        logger.info(" ⚠️  Zero variance on every metric — all %d runs returned"
                    " identical results. A caching backend/proxy is the usual"
                    " cause.", runs)
    logger.info("")


def log_token_stats() -> None:
    """Log 📊 Token stats from GLOBAL_STATS. Same format across all services.

    Uses logger.info() → silent for library users, pretty for CLI.
    """
    stats = GLOBAL_STATS.snapshot()
    phases = stats.get("phases", {})

    logger.info(settings.section_rule("Execution Stats"))
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