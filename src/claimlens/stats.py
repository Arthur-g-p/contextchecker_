import time
from dataclasses import dataclass, field
from typing import List
import threading

from claimlens import settings
from claimlens.utils import build_variance

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

    # Locked request strategy per (endpoint, model), recorded by the client.
    _strategies: dict = field(default_factory=dict)

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

    def record_strategy(self, model: str, endpoint: str | None, strategy: str):
        """Remember which request strategy a model locked, for report _meta."""
        with self._lock:
            self._strategies[(endpoint, model)] = strategy

    def strategies(self) -> dict[str, str]:
        """{model: strategy name}. A model seen on two endpoints keeps both
        entries, the second keyed with its endpoint appended."""
        out: dict[str, str] = {}
        with self._lock:
            for (endpoint, model), name in self._strategies.items():
                key = model if model not in out else f"{model} @ {endpoint or 'litellm'}"
                out[key] = name
        return out

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
                result["phases"] = {k: dict(v) for k, v in self._phases.items()}
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


_USAGE_KEYS = ("requests", "input_tokens", "output_tokens", "reasoning_tokens")


def usage_since(before: dict | None) -> dict:
    """What one report cost: GLOBAL_STATS now minus a snapshot taken at its start.

    The global counters run for the whole process, so under --runs a raw
    read would charge run 3 for runs 1 and 2 as well. `requests` counts every
    HTTP call; tokens exist only for calls that returned usage, so a timeout
    or a rejected request adds a request and no tokens.
    """
    now = GLOBAL_STATS.snapshot()
    if before is None:                      # report built without a run
        before = now

    def _delta(a: dict, b: dict, requests_key: str) -> dict:
        return {
            "requests": a.get(requests_key, 0) - b.get(requests_key, 0),
            "input_tokens": a.get("input_tokens", 0) - b.get("input_tokens", 0),
            "output_tokens": a.get("output_tokens", 0) - b.get("output_tokens", 0),
            "reasoning_tokens": a.get("reasoning_tokens", 0) - b.get("reasoning_tokens", 0),
        }

    usage = _delta(now, before, "total_requests")
    phases = {}
    for name, after in now.get("phases", {}).items():
        d = _delta(after, before.get("phases", {}).get(name, {}), "requests")
        if d["requests"] > 0:
            phases[name] = d
    if phases:
        usage["phases"] = phases
    return usage


def sum_usage(usages: list[dict | None]) -> dict:
    """Add per-run usage blocks into one — for the outer _meta of a --runs report."""
    total = {k: 0 for k in _USAGE_KEYS}
    phases: dict[str, dict] = {}
    for u in usages:
        if not u:
            continue
        for k in _USAGE_KEYS:
            total[k] += u.get(k, 0)
        for name, p in u.get("phases", {}).items():
            acc = phases.setdefault(name, {k: 0 for k in _USAGE_KEYS})
            for k in _USAGE_KEYS:
                acc[k] += p.get(k, 0)
    if phases:
        total["phases"] = phases
    return total


def document_meta(run_docs: list[dict], runs: int, total_seconds: float) -> dict:
    """The document-level ``_meta`` of a record: run 1's core (timestamp,
    counts, strategies), ``runs`` = N, the wall-clock total, usage summed
    over runs. Identical at N = 1, where it mirrors the single run."""
    meta = {k: v for k, v in run_docs[0]["_meta"].items()
            if k not in ("run", "duration_seconds")}
    meta["runs"] = runs
    meta["duration_seconds"] = total_seconds
    meta["usage"] = sum_usage([d["_meta"].get("usage") for d in run_docs])
    return meta


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
    logger.info(" 🌐 API & Parsing")
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
    logger.info(" ✅ Run %d/%d done in %.1fs · %s",
                run, runs, duration_seconds,
                format_headline(metrics, keys) or "done")


def format_headline(metrics: dict, keys: tuple[str, ...]) -> str:
    """``precision 0.657 · recall 0.250 · f1 0.163`` — the headline metrics
    of one run, presence-filtered: keys missing or null drop out. Shared by
    the run line and the single-run Done line, so both speak one grammar."""
    return " · ".join(
        f"{key} {metrics[key]:.3f}"
        for key in keys
        if isinstance(metrics.get(key), (int, float))
    )


def log_mece_tree(
    header: str,
    total: int,
    total_label: str,
    branches: list[tuple],
    footer: list[tuple] | None = None,
    header_note: str | None = None,
) -> None:
    """Render one MECE tree per docs/output_conventions.md rule set 1.

    *branches* are (icon, count, plain-English label, note-or-None[,
    children]); their counts must partition *total* — a mismatch logs a
    warning (holy-data enforcement, never a crash). Optional *children*
    are (label, count, note-or-None) sub-branches that partition their
    parent's count — a cause breakdown, never a second level of the
    header's partition. *footer* rates are (name, numerator,
    denominator), rendered as the single terminal `→` line with visible
    fractions; only footer rates may be aggregated by the variance block,
    under the same names.
    """
    top = f" {header} — {total} {total_label}"
    if header_note:
        top += f"  ({header_note})"
    logger.info(top)

    branch_sum = sum(b[1] for b in branches)
    if branch_sum != total:
        logger.warning(
            "⚠️  MECE violation in '%s': branches sum to %d, header says %d",
            header, branch_sum, total,
        )

    texts = [f"{b[1]} {b[2]}" for b in branches]
    width = max(len(t) for t in texts) if texts else 0
    for i, b in enumerate(branches):
        icon, count, _, note = b[:4]
        children = b[4] if len(b) > 4 else None
        last = not footer and i == len(branches) - 1
        prefix = "└─" if last else "├─"
        line = f"     {prefix} {icon} {texts[i]:<{width}}"
        if note:
            line = f"{line}  {note}" if note.startswith("[") else f"{line}  ({note})"
        logger.info(line.rstrip())
        if children:
            child_sum = sum(c[1] for c in children)
            if child_sum != count:
                logger.warning(
                    "⚠️  MECE violation in '%s' › %s: sub-branches sum to %d,"
                    " branch says %d", header, b[2], child_sum, count,
                )
            stem = "          " if last else "     │    "
            cwidth = max(len(f"{c[1]} {c[0]}") for c in children)
            for j, (clabel, ccount, cnote) in enumerate(children):
                sub = "└─" if j == len(children) - 1 else "├─"
                cline = f"{stem}{sub} {f'{ccount} {clabel}':<{cwidth}}"
                if cnote:
                    cline = (f"{cline}  {cnote}" if cnote.startswith("[")
                             else f"{cline}  ({cnote})")
                logger.info(cline.rstrip())

    if footer:
        parts = []
        for entry in footer:
            name, num, den = entry[0], entry[1], entry[2]
            # optional 4th element: a fraction suffix, e.g. "judged" when
            # the footer denominator is a subset of the header total.
            suffix = f" {entry[3]}" if len(entry) > 3 and entry[3] else ""
            rate = "n/a" if not den else f"{num / den:.3f}"
            parts.append(f"{name} {rate} ({num} / {den}{suffix})")
        logger.info("     └─ → %s", " · ".join(parts))


def log_rate_rows(
    header: str,
    rows: list[tuple],
    header_note: str | None = None,
) -> None:
    """Render one rate-rows block (docs/output_conventions.md rule set 2).

    Each row is (icon, label, count, denominator, phrase, rate_key,
    causes[, warning]) and is its own derivation: ``2 of 8 items failed →
    extraction_error_rate 0.250``. Rates are 0-1 decimals (percent is
    banned); *rate_key* names the exact variance/JSON key the number
    becomes (None for rows whose rate is not exported yet). count None
    renders a "not measured" row with *phrase* as the reason. Rows never
    sum and never pretend to; zero rows always print — a hidden row is
    indistinguishable from an unmeasured one. *causes* ({cause: n})
    render as sub-branches, cause names in plain English. An optional
    8th element is a warning printed under its own row (at warning
    level, so it survives quieter log configurations).
    """
    top = f" {header}"
    if header_note:
        top += f"  ({header_note})"
    logger.info(top)

    labels = [f"{label}:" for _, label, *_ in rows]
    width = max(len(l) for l in labels) if labels else 0
    for i, row in enumerate(rows):
        icon, label, count, den, phrase, rate_key, causes = row[:7]
        warning = row[7] if len(row) > 7 else None
        last = i == len(rows) - 1
        prefix = "└─" if last else "├─"
        stem = "          " if last else "     │    "
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
            items = sorted(causes.items(), key=lambda kv: -kv[1])
            for j, (cause, num) in enumerate(items):
                sub = "└─" if j == len(items) - 1 else "├─"
                logger.info("%s%s %s: %d", stem, sub,
                            str(cause).replace("_", " "), num)
        if warning:
            logger.warning("%s⚠️  %s", stem, warning)


def roster_from_sections(sections: dict) -> list[str]:
    """Flatten a _VARIANCE_SECTIONS spec into the ordered key roster."""
    keys: list[str] = []
    for _, group_keys in sections.get("metrics", []):
        keys.extend(group_keys)
    keys.extend(sections.get("behavior", []))
    keys.extend(sections.get("health", []))
    return keys


class VarianceTracker:
    """Tracks variance across --runs: per-run metrics in, one block out.

    One instance per multi-run invocation. The command's own loop stays
    untouched — it calls ``add(metrics, duration)`` once per run wherever
    it likes, then ``finish(runs)`` computes means/variance over the
    declared roster and prints the VARIANCE block + token table.
    Calculation and printing are central; choreography, documents, and
    _meta stay with the command (docs/holy_data.md rule set 4).
    """

    def __init__(self, sections: dict | None, labels: dict | None = None,
                 unmeasured: dict | None = None, directions: dict | None = None):
        self._sections = sections
        self._labels = labels
        self._directions = directions
        self._unmeasured = unmeasured
        self._metrics: list[dict] = []
        self._durations: list[float] = []
        self._started = time.perf_counter()
        self.total_seconds: float = 0.0

    def add(self, metrics: dict, duration: float) -> None:
        """Record one run's metric dict and duration."""
        self._metrics.append(metrics)
        self._durations.append(duration)

    def finish(self, runs: int, log: bool = True) -> tuple[dict, dict]:
        """Aggregate and (unless silenced) print the closing blocks.

        Returns (means, variance) for document assembly;
        ``total_seconds`` is set for the caller's _meta.
        """
        roster = (roster_from_sections(self._sections)
                  if self._sections else None)
        means, variance = build_variance(self._metrics, roster=roster)
        self.total_seconds = round(time.perf_counter() - self._started, 1)
        if log:
            log_variance_block(runs, means, variance,
                               self._durations, self.total_seconds,
                               sections=self._sections, labels=self._labels,
                               unmeasured=self._unmeasured,
                               directions=self._directions)
            log_token_stats()
        return means, variance


def _variance_label(key: str, labels: dict | None) -> str:
    return (labels or {}).get(key, key.replace("_", " "))


def _log_variance_rows(
    keys: list[str], means: dict, variance: dict, runs: int,
    labels: dict | None, indent: str = "    ", width: int = 30,
    unmeasured: dict | None = None, directions: dict | None = None,
) -> None:
    """One mean ± std row per key: label (same name as the per-run line),
    null / unmeasured / partial-support handling. *width* is the label
    column, shared by every section of the block."""
    present = [k for k in keys if k in means]
    for i, key in enumerate(present):
        prefix = "└─" if i == len(present) - 1 else "├─"
        label = _variance_label(key, labels) + ":"
        v = variance[key]
        if unmeasured and key in unmeasured:
            logger.info("%s%s %-*s n/a  (not measured — %s)",
                        indent, prefix, width, label, unmeasured[key])
            continue
        if means[key] is None:
            # Null in every run: the metric exists but was never computable.
            logger.info("%s%s %-*s n/a  (not computable in any of %d runs)",
                        indent, prefix, width, label, runs)
            continue
        partial = ""
        if v.get("n", runs) < runs:
            # Mean over fewer runs than executed — say so, or the mean
            # overstates its support.
            partial = f"  ({v['n']}/{runs} runs)"
        note = f"  ({directions[key]})" if directions and key in directions else ""
        logger.info("%s%s %-*s %.3f ± %.3f   [%.3f, %.3f]%s%s",
                    indent, prefix, width, label, means[key], v["std"],
                    v["min"], v["max"], partial, note)


def log_variance_block(
    runs: int,
    means: dict,
    variance: dict,
    durations: list[float] | None = None,
    total_seconds: float | None = None,
    sections: dict | None = None,
    labels: dict | None = None,
    unmeasured: dict | None = None,
    directions: dict | None = None,
) -> None:
    """Print ══ VARIANCE (N runs) ══ (docs/output_conventions.md rule set 4).

    With *sections* the block mirrors the per-run structure: 📊 Metrics
    (grouped), ⚪ Behavior, 💥 Health, ⏱ Time — a section renders iff its
    per-run block exists for the command. The zero-variance caching
    warning is computed over Metrics only (all-zero Health is the desired
    state, not caching evidence). Without *sections*: legacy flat render.
    The math lives in utils.build_variance."""
    logger.info("")
    logger.info(settings.section_rule(f"VARIANCE ({runs} runs)", char="═"))
    logger.info("")
    health = [k for k in (sections or {}).get("health", []) if k in means]
    labels = {**(labels or {}), **{k: k for k in health}}
    width = max((len(_variance_label(k, labels)) + 2 for k in means), default=30)
    rows = dict(means=means, variance=variance, runs=runs, labels=labels,
                width=width, unmeasured=unmeasured, directions=directions)
    if sections is None:
        logger.info(" 📊 Metrics  (mean ± std  [min, max])")
        _log_variance_rows(list(means.keys()), **rows)
        warn_keys = list(means.keys())
    else:
        logger.info(" 📊 Metrics  (mean ± std  [min, max])")
        metric_groups = sections.get("metrics", [])
        warn_keys = []
        for group_name, group_keys in metric_groups:
            if group_name:
                logger.info("    %s", group_name)
            _log_variance_rows(group_keys, **rows)
            warn_keys.extend(group_keys)
        behavior = [k for k in sections.get("behavior", []) if k in means]
        if behavior:
            logger.info("")
            logger.info(" ⚪ %s", sections.get("behavior_title", "Abstention Behavior"))
            _log_variance_rows(behavior, **rows)
        if health:
            logger.info("")
            logger.info(" 💥 Reliability  (tooling — should be zero)")
            _log_variance_rows(health, **rows)
    if durations:
        logger.info("")
        total = total_seconds if total_seconds is not None else sum(durations)
        logger.info(" ⏱️  %-10s %6.1fs", "Total:", total)
        for i, d in enumerate(durations, 1):
            prefix = "└─" if i == len(durations) else "├─"
            logger.info("    %s %-10s %6.1fs", prefix, f"run {i}:", d)
    stds = [variance[k]["std"] for k in warn_keys
            if k in variance and variance[k]["std"] is not None]
    if stds and all(std == 0 for std in stds):
        logger.info("")
        logger.info(" ⚠️  Zero variance on every metric — all %d runs returned"
                    " identical results. A caching backend/proxy is the usual"
                    " cause.", runs)


def log_token_stats() -> None:
    """Log 📊 Token stats from GLOBAL_STATS. Same format across all services.

    Uses logger.info() → silent for library users, pretty for CLI.
    """
    stats = GLOBAL_STATS.snapshot()
    phases = stats.get("phases", {})

    logger.info("")  # one blank before the appendix, whatever closed the findings
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