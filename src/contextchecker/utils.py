"""
Shared utility functions — pure helpers with no side effects.

This file is a leaf dependency — it imports nothing from contextchecker.
"""
import json
import statistics


# ── Repeated-run aggregation (pure math; printing lives in stats.py) ─────────

def aggregate_values(values: list) -> dict | None:
    """n/std/min/max/values over the numeric entries; None if nothing numeric.

    Sample standard deviation (n-1); a single value gets std 0.0 so the
    field shape stays constant. Nulls are skipped, never zeroed — and "n"
    reports how many runs actually contributed, so a mean over fewer runs
    than were executed stays visible.
    """
    known = [v for v in values
             if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not known:
        return None
    return {
        "n": len(known),
        "std": round(statistics.stdev(known), 4) if len(known) > 1 else 0.0,
        "min": round(min(known), 4),
        "max": round(max(known), 4),
        "values": [round(v, 4) for v in known],
    }


def build_variance(metric_dicts: list[dict]) -> tuple[dict, dict]:
    """(means, variance) over the top-level scalar keys of per-run dicts.

    Non-scalar keys (support, extraction_errors, nested reports) are
    skipped — they live untouched inside each run's own document.

    A nullable metric keeps its key: a metric that was null in every run
    appears as mean None with n=0 rather than vanishing, so multi-run and
    single-run documents expose the same metric surface. (Null = the run
    could not compute it; dropping the key would hide that a metric was
    supposed to exist.)
    """
    keys: list[str] = []
    for d in metric_dicts:
        for key, value in d.items():
            if key in keys:
                continue
            if value is None or (isinstance(value, (int, float))
                                 and not isinstance(value, bool)):
                keys.append(key)
    # A key that is non-scalar in ANY run is structural, not a metric.
    keys = [k for k in keys if not any(
        isinstance(d.get(k), (dict, list, str, bool)) for d in metric_dicts)]

    means: dict = {}
    variance: dict = {}
    for key in keys:
        agg = aggregate_values([d.get(key) for d in metric_dicts])
        if agg is None:
            means[key] = None
            variance[key] = {"n": 0, "std": None, "min": None, "max": None,
                             "values": []}
        else:
            means[key] = round(sum(agg["values"]) / len(agg["values"]), 4)
            variance[key] = agg
    return means, variance

def build_compact_schema_example(schema) -> str:
    """Build a compact JSON example from a Pydantic model for vanilla LLM prompts.

    Used as a fallback when the model doesn't support structured output
    and no hand-written vanilla prompt exists in prompt_map.json.
    """
    try:
        json_schema = schema.model_json_schema()
        props = json_schema.get("properties", {})
        example = {k: f"<{v.get('type', 'string')}>" for k, v in props.items()}

        return json.dumps(example, indent=2)
    except Exception:
        return '{"result": "<see prompt for format>"}'


def format_prompt(template: str, variables: dict) -> str:
    """Replace ``{{key}}`` placeholders in *template* with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def canonicalize_triplets(triplets: list[dict]) -> None:
    """Normalize triplets IN-PLACE to canonical ``{subject, predicate, object}``.

    Single source of truth — every component calls this ONCE at ingestion, so
    nothing downstream has to juggle both formats. Converts the legacy
    ``{"triplet": [s, p, o], ...meta}`` shape to
    ``{"subject": s, "predicate": p, "object": o, ...meta}``, preserving all
    other keys (e.g. ``human_label``). Already-canonical dicts are left untouched.
    Short/malformed legacy lists are padded with empty strings (never raises).
    """
    for t in triplets:
        if "triplet" in t and "subject" not in t:
            parts = (list(t.pop("triplet")) + ["", "", ""])[:3]
            t["subject"], t["predicate"], t["object"] = parts[0], parts[1], parts[2]


def triplet_key(triplet: dict, *, case_insensitive: bool = False) -> tuple:
    """Normalized (subject, predicate, object) key for exact comparison.

    Pure string comparison, language-independent. Leading/trailing whitespace is
    trimmed; pass ``case_insensitive=True`` to also fold case (Unicode-correct
    casefold).
    """
    parts = (triplet["subject"], triplet["predicate"], triplet["object"])
    key = tuple(str(p).strip() for p in parts)
    if case_insensitive:
        key = tuple(p.casefold() for p in key)
    return key


def deduplicate_triplets(
    triplets: list[dict], *, case_insensitive: bool = False
) -> list[dict]:
    """Return triplets with exact (s, p, o) duplicates removed, keeping the first.

    A repeated triplet carries no information, so dropping it is loss-free.
    Pure string comparison via :func:`triplet_key` — language-independent, no
    semantics, no LLM. Order is preserved; the first occurrence (and its
    metadata) is kept. Does not mutate the input list.
    """
    seen: set = set()
    out: list[dict] = []
    for t in triplets:
        k = triplet_key(t, case_insensitive=case_insensitive)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def find_duplicate_triplets(
    triplets: list[dict], *, case_insensitive: bool = False
) -> list[dict]:
    """Return the duplicate occurrences — every triplet after the first with the
    same (s, p, o) key.

    Complements :func:`deduplicate_triplets`: ``dedup`` returns the survivors,
    this returns the ones ``dedup`` would drop. Used to *report* duplicates
    (e.g. in the evaluator) without removing anything. Order is preserved.
    """
    seen: set = set()
    dups: list[dict] = []
    for t in triplets:
        k = triplet_key(t, case_insensitive=case_insensitive)
        if k in seen:
            dups.append(t)
        else:
            seen.add(k)
    return dups



# ── Report envelope ──────────────────────────────────────────────────────────

REPORT_SCHEMA_VERSION = 4


def build_meta(
    report_type: str,
    *,
    timestamp: str,
    duration_seconds: float,
    total_items: int,
    evaluated_items: int,
    dropped_items: int,
    **extras,
) -> dict:
    """The ``_meta`` block: what the run was, as opposed to what was asked for.

    Everything here is *discovered* — derived keys, counts, timings. Anything
    the caller passed on the command line belongs in ``_args`` instead.

    The eight core keys are identical in every report type and come first;
    report-specific ``extras`` are appended after them. Timestamp and duration
    are supplied by the caller so this stays deterministic under test.
    """
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": report_type,
        "contextchecker_version": _package_version(),
        "timestamp": timestamp,
        "duration_seconds": round(duration_seconds, 1),
        "total_items": total_items,
        "evaluated_items": evaluated_items,
        "dropped_items": dropped_items,
        **extras,
    }


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("contextchecker")
    except Exception:
        return "unknown"
