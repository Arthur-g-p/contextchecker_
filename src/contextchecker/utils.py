"""
Shared utility functions — pure helpers with no side effects.

This file is a leaf dependency — it imports nothing from contextchecker.
"""
import json

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

