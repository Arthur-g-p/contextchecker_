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
    first_pass_ok: int = 0        # OK before any retry round (set once, never modified)
    empty: int = 0                # parsed successfully but 0 items (e.g. empty triplets)
    context_too_long: int = 0     # ContextTooLongError — permanent
    content_policy: int = 0       # ContentPolicyError — permanent
    finish_reason_length: int = 0 # FinishReasonLengthError — permanent
    timeout: int = 0              # LLMTimeoutError — permanent
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
    # Causes: "context_too_long" | "finish_reason_length" | "content_policy"
    #       | "timeout" | "parse_failure".
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
        return (self.context_too_long + self.content_policy + self.timeout
                + self.finish_reason_length + self.permanently_failed)

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

            # Update phase stats
            if self._current_phase in self._phases:
                p = self._phases[self._current_phase]
                p["input_tokens"] += inp
                p["output_tokens"] += out
                p["reasoning_tokens"] += reasoning

    def log_request(self):
        """Count one HTTP call, whatever it returns.

        Timeouts and rejected requests carry no usage, so counting on the
        response would report fewer requests than were actually sent.
        """
        with self._lock:
            self.total_requests += 1
            if self._current_phase in self._phases:
                self._phases[self._current_phase]["requests"] += 1

    def log_error(self):
        """Record one permanently failed ITEM.

        Does not touch the request counters — those count HTTP calls, and one
        failed item is not one request.
        """
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
    permanent = (stats.context_too_long + stats.content_policy + stats.timeout
                 + stats.finish_reason_length)
    has_permanent = permanent > 0
    has_retryable = stats.parse_error > 0

    if has_permanent or has_retryable:
        logger.info("     ├─ ✅ %d parsed before any retry round", stats.first_pass_ok)
    else:
        logger.info("     └─ ✅ %d parsed before any retry round", stats.first_pass_ok)

    # ── Permanent failures (context too long / content policy)
    if has_permanent:
        prefix = "├─" if has_retryable else "└─"
        logger.info("     %s 🚫 %d permanent failures", prefix, permanent)
        # A sibling below needs the continuation bar; the last node does not.
        stem = "     │    " if has_retryable else "          "
        causes = [
            ("📏", "context too long", stats.context_too_long),
            ("✂️ ", "finish reason length", stats.finish_reason_length),
            ("🛡️ ", "content policy", stats.content_policy),
            ("⏱️ ", "timed out", stats.timeout),
        ]
        shown = [(icon, label, n) for icon, label, n in causes if n > 0]
        for i, (icon, label, n) in enumerate(shown):
            branch = "└─" if i == len(shown) - 1 else "├─"
            logger.info("%s%s %s %d %s", stem, branch, icon, n, label)

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


def log_run_line(
    run: int,
    runs: int,
    duration_seconds: float,
    metrics: dict,
    keys: tuple[str, ...],
) -> None:
    """Print the ✅ run line closing one run in variance mode.

    One format for all four --runs commands. Presence-filtered: only *keys*
    that exist in *metrics* as numbers are shown, so each command passes its
    own headline keys and missing/null metrics drop out silently.
    """
    parts = [
        f"{key} {metrics[key]:.3f}"
        for key in keys
        if isinstance(metrics.get(key), (int, float))
    ]
    logger.info(" ✅ Run %d/%d done in %.1fs · %s",
                run, runs, duration_seconds, " · ".join(parts) or "done")


def log_mece_tree(
    header: str,
    total: int,
    total_label: str,
    branches: list[tuple],
    footer: list[tuple] | None = None,
    header_note: str | None = None,
) -> None:
    """Render one MECE tree per docs/holy_data.md rule set 1.

    *branches* are (icon, count, plain-English label, note-or-None); their
    counts must partition *total* — a mismatch logs a warning (holy-data
    enforcement, never a crash). *footer* rates are (name, numerator,
    denominator), rendered as the single terminal `→` line with visible
    fractions; only footer rates may be aggregated by the variance block,
    under the same names.
    """
    top = f" {header} — {total} {total_label}"
    if header_note:
        top += f"  ({header_note})"
    logger.info(top)

    branch_sum = sum(count for _, count, _, _ in branches)
    if branch_sum != total:
        logger.warning(
            "⚠️  MECE violation in '%s': branches sum to %d, header says %d",
            header, branch_sum, total,
        )

    texts = [f"{count} {label}" for _, count, label, _ in branches]
    width = max(len(t) for t in texts) if texts else 0
    for i, (icon, _, _, note) in enumerate(branches):
        prefix = "├─" if footer or i < len(branches) - 1 else "└─"
        line = f"     {prefix} {icon} {texts[i]:<{width}}"
        if note:
            line = f"{line}  ({note})"
        logger.info(line.rstrip())

    if footer:
        parts = []
        for entry in footer:
            name, num, den = entry[0], entry[1], entry[2]
            # optional 4th element: a fraction suffix, e.g. "judged" when
            # the footer denominator is a subset of the header total.
            suffix = f" {entry[3]}" if len(entry) > 3 and entry[3] else ""
            parts.append(f"{name} n/a" if not den
                         else f"{name} {num / den:.3f} ({num} / {den}{suffix})")
        logger.info("     └─ → %s", " · ".join(parts))


def log_rate_rows(
    header: str,
    rows: list[tuple],
    header_note: str | None = None,
) -> None:
    """Render one rate-rows block (docs/holy_data.md rule set 2).

    Each row is (icon, label, count, denominator, phrase, rate_key,
    causes) and is its own derivation: ``2 of 8 items failed →
    extraction_error_rate 0.250``. Rates are 0-1 decimals (percent is
    banned); *rate_key* names the exact variance/JSON key the number
    becomes (None for rows whose rate is not exported yet). count None
    renders a "not measured" row with *phrase* as the reason. Rows never
    sum and never pretend to; zero rows always print — a hidden row is
    indistinguishable from an unmeasured one. *causes* ({cause: n})
    render as sub-branches, cause names in plain English.
    """
    top = f" {header}"
    if header_note:
        top += f"  ({header_note})"
    logger.info(top)

    labels = [f"{label}:" for _, label, *_ in rows]
    width = max(len(l) for l in labels) if labels else 0
    for i, (icon, label, count, den, phrase, rate_key, causes) in enumerate(rows):
        last = i == len(rows) - 1
        prefix = "└─" if last else "├─"
        name = f"{label}:".ljust(width)
        if count is None:
            logger.info("     %s %s %s  not measured  (%s)",
                        prefix, icon, name, phrase)
            continue
        line = f"     {prefix} {icon} {name}  {count} of {den} {phrase}"
        if rate_key:
            rate = "n/a" if not den else f"{count / den:.3f}"
            line += f"  → {rate_key} {rate}"
        logger.info(line)
        if causes:
            stem = "          " if last else "     │    "
            items = sorted(causes.items(), key=lambda kv: -kv[1])
            for j, (cause, num) in enumerate(items):
                sub = "└─" if j == len(items) - 1 else "├─"
                logger.info("%s%s %s: %d", stem, sub,
                            str(cause).replace("_", " "), num)


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