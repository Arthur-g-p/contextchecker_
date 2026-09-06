"""
Shared utility functions — pure helpers.

This file is a leaf dependency — it imports nothing from contextchecker.
"""
import json
import logging
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


def build_variance(
    metric_dicts: list[dict], roster: list[str] | None = None,
) -> tuple[dict, dict]:
    """(means, variance) over the top-level scalar keys of per-run dicts.
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
    if roster is not None:
        keys = [k for k in roster if k in keys or
                all(d.get(k) is None for d in metric_dicts)]

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

# ── Prompt building ──────────────────────────────────────────────────────────

_SCHEMA_SHAPE_MAX_DEPTH = 6


def build_schema_shape(schema) -> str:
    """Render a Pydantic model as the JSON shape a prompt should ask for."""
    def render(node: dict, defs: dict, depth: int, seen: frozenset):
        if depth > _SCHEMA_SHAPE_MAX_DEPTH:
            return "<...>"

        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                return "<recursive>"
            return render(defs.get(name, {}), defs, depth, seen | {name})

        for union_key in ("anyOf", "oneOf"):
            if union_key in node:
                branches = [b for b in node[union_key] if b.get("type") != "null"]
                return render(branches[0], defs, depth, seen) if branches else None

        # A one-value Literal arrives as const, not enum.
        if "const" in node:
            return str(node["const"])
        if "enum" in node:
            return " | ".join(str(v) for v in node["enum"])

        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            if "properties" in node:
                return {
                    key: render(sub, defs, depth + 1, seen)
                    for key, sub in node["properties"].items()
                }
            # A mapping carries its value type in additionalProperties instead.
            values = node.get("additionalProperties")
            return {"<key>": render(values, defs, depth + 1, seen)
                    if isinstance(values, dict) else "<any>"}
        if node_type == "array":
            # Fixed-length tuples enumerate their members instead of one item type.
            if "prefixItems" in node:
                return [render(i, defs, depth + 1, seen) for i in node["prefixItems"]]
            return [render(node.get("items", {}), defs, depth + 1, seen)]

        return f"<{node_type or 'string'}>"

    try:
        json_schema = schema.model_json_schema()
        shape = render(json_schema, json_schema.get("$defs", {}), 0, frozenset())
        return json.dumps(shape, indent=2)
    except Exception:
        return '{"result": "<see prompt for format>"}'


def prepare_plain_prompt(
    template: str | None, key: str, schema, logger: logging.Logger,
) -> str | None:
    """Substitute {{schema}} into a plain prompt.

    A missing prompt or a missing placeholder only matters if the endpoint
    needs unguided decoding, so both warn here and fail there. Logs through
    *logger* so the warning names the caller, not this module.
    """
    if template is None:
        logger.warning(
            "   ⚠️  No plain prompt '%s' in prompt_map — a model that cannot take "
            "a schema will abort the run.", key,
        )
        return None
    if "{{schema}}" not in template:
        logger.warning(
            "   ⚠️  Plain prompt '%s' has no {{schema}} placeholder — the model "
            "will get no output shape.", key,
        )
        return template
    return format_prompt(template, {"schema": build_schema_shape(schema)})


def findings_view(branches: list[str], items: list[dict], classify) -> dict:
    """The review queue over a record's ``items``, keyed by the console's
    own branch names: ``{branch: [entry, …]}``. *classify(item)* yields
    ``(branch, entry)`` pairs; every entry names its item (``id`` /
    ``query_id``) so it can be found. Every branch is present, an empty
    one as ``[]`` — hidden is not zero. A pure view, never a source."""
    out: dict[str, list[dict]] = {branch: [] for branch in branches}
    for item in items:
        for branch, entry in classify(item):
            out[branch].append(entry)
    return out

def plural(n: int, word: str, plural_form: str | None = None) -> str:
    """The noun for *n*: ``plural(1, "item")`` → ``item``, ``plural(2, "item")``
    → ``items``. Log lines only — JSON keys never change number."""
    return word if n == 1 else (plural_form or word + "s")


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

REPORT_SCHEMA_VERSION = 5


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
