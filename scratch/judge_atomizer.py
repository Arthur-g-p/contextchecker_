"""
LLM-as-judge for the atomizer decision trace.

Reads {output}_decisions.json (from `atomize --trace`) and, for every SPLIT
decision, asks a strong model to judge it semantically — the thing lexical code
could never do. Two jobs in one pass:

  1. Classify the split:   GOOD  | BLEED | NONSENSE
       GOOD     = children together mean EXACTLY the original; nothing added/lost.
       BLEED    = a child contains info NOT in the original triplet (pulled from
                  the response or world knowledge) -> the real do-no-harm failure.
       NONSENSE = malformed / incoherent / meaning not preserved.
  2. Recommend GT fixes:   if the source/GT triplets are duplicated, malformed, or
       wrong, it says what to change (it does NOT apply anything).

The judge ALWAYS gets full context (response + entire input set). Run it on the
with-response trace AND the without-response trace, then compare the harm rates
to settle the response A/B.

Usage:
    python scratch/judge_atomizer.py <decisions.json> \
        --model <strong-model> --base-url http://localhost:4000/v1 \
        [--api-key ... | env CHECKER_API_KEY] [--out verdicts.json]
"""
import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

try:
    from dotenv import load_dotenv
    load_dotenv()  # pick up CHECKER_API_KEY etc. from .env
except ImportError:
    pass


SYSTEM = (
    "You are a meticulous judge of knowledge-graph triplet atomization. "
    "You decide whether a SPLIT the atomizer made was correct, and you flag "
    "problems in the ground-truth triplets. You reason before you rule."
)

RUBRIC = """A triplet atomizer split ONE source triplet into atomic facts. Judge the split.

CRITICAL — keep these THREE things separate, do not conflate them:
- The GROUND-TRUTH set below is the CURRENT ground truth, exactly as it stands NOW.
- The CHILDREN are a PROPOSED split. They have NOT been added to the ground truth.
  Do not assume the children are present in the ground-truth set.
- The ORIGINAL triplet is one member of the ground-truth set (the one being split).

VERDICT (judge the split only):
- GOOD: the children together mean EXACTLY the original triplet — no information
  added, none lost, subject/predicate/object roles and direction preserved.
- BLEED: at least one child contains information NOT in the original triplet (pulled
  from the response or world knowledge). The worst error. Example: splitting
  "(odor, similar to, cucumbers)" into "vinegar" + "cucumbers" because the response
  said "vinegar or cucumbers".
- NONSENSE: the children are malformed, incoherent, or change the meaning.

If the original triplet itself contained an internal redundancy (e.g. an object
list with "Turkish" twice) and the atomizer faithfully preserved it, the split is
still GOOD and is_atomizer_harm=false — that redundancy is the EXTRACTOR's.

GT EDIT PLAN (gt_fixes) — produce the EXACT list of edits a human would apply.
READ THIS CAREFULLY: the atomized CHILDREN are NOT in the ground truth, and they
are NOT added automatically. Treat NOTHING as already done. So, if the split is
GOOD and should be reflected in the ground truth, you MUST output:
- one {"action": "add", "triplet": [s, p, o], "reason": "atomic part of the split compound"} for EVERY child, repeating each child explicitly, AND
- one {"action": "remove", "triplet": [s, p, o], "reason": "non-atomic; replaced by the atomic triplets added above"} for the ORIGINAL compound triplet.
Additionally, output {"action": "remove", "triplet": [s, p, o], "reason": "exact duplicate"} for any duplicate triplet in the CURRENT ground-truth set.
Reference every triplet EXACTLY as it appears now. If the ground truth for this
item is already atomic and has no duplicates, gt_fixes = [].

Return ONLY JSON:
{"explanation": "...", "verdict": "GOOD|BLEED|NONSENSE", "is_atomizer_harm": true|false,
 "gt_fixes": [ {"action": "add|remove", "triplet": [s, p, o], "reason": "..."} ]}"""


def fmt_triplets(triplets):
    out = []
    for t in triplets:
        if "triplet" in t:
            s, p, o = (t["triplet"] + ["", "", ""])[:3]
        else:
            s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        out.append(f'  ("{s}", "{p}", "{o}")')
    return "\n".join(out)


def build_user(item, dec):
    orig = dec["original"]
    return (
        f"{RUBRIC}\n\n"
        f"### Response (source text):\n{item.get('response', '') or 'CONTEXT NOT PROVIDED'}\n\n"
        f"### CURRENT ground-truth triplets (full set as it stands now — children NOT added):\n{fmt_triplets(item.get('input_triplets', []))}\n\n"
        f'### The original triplet that was split:\n  ("{orig["subject"]}", "{orig["predicate"]}", "{orig["object"]}")\n\n'
        f"### The atomizer's reasoning for splitting:\n{dec.get('reasoning', '')}\n\n"
        f"### The atomizer's children:\n{fmt_triplets(dec.get('children', []))}\n\n"
        f"### Superficial duplicates in this item's output:\n{json.dumps(item.get('superficial_collisions', []))}"
    )


def _extract_json(raw):
    """Pull a JSON object out of a response that may have prose/reasoning or
    ```json fences around it (reasoning models leak prose into content)."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            pass
    return None


def judge_one(client, model, item, dec):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_user(item, dec)}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    parsed = _extract_json(raw)
    if parsed is not None:
        return parsed
    # Real format failure — do NOT call it NONSENSE (that's a split judgement).
    return {"explanation": f"PARSE_FAIL: {(raw or '')[:300]}", "verdict": "PARSE_ERROR",
            "is_atomizer_harm": False, "gt_fixes": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:4000/v1")
    ap.add_argument("--api-key", default=os.getenv("CHECKER_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-noauth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace = json.load(open(args.trace, encoding="utf-8"))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    verdicts = []
    counts = {"GOOD": 0, "BLEED": 0, "NONSENSE": 0}
    harm = 0
    gt_fixes_all = []

    splits = [(it, d) for it in trace for d in it.get("decisions", []) if d.get("decision") == "split"]
    print(f"Judging {len(splits)} split decisions with {args.model} ...\n")

    for i, (item, dec) in enumerate(splits, 1):
        v = judge_one(client, args.model, item, dec)
        verdict = v.get("verdict", "NONSENSE").upper()
        counts[verdict] = counts.get(verdict, 0) + 1
        if v.get("is_atomizer_harm"):
            harm += 1
        fixes = v.get("gt_fixes") or []
        for fx in fixes:
            gt_fixes_all.append({"id": item.get("id"), **fx})
        verdicts.append({
            "id": item.get("id"),
            "original": dec["original"],
            "children": dec.get("children", []),
            "verdict": verdict,
            "is_atomizer_harm": bool(v.get("is_atomizer_harm")),
            "explanation": v.get("explanation", ""),
            "gt_fixes": fixes,
        })
        mark = "⚠️ HARM" if v.get("is_atomizer_harm") else verdict
        print(f"  [{i}/{len(splits)}] id {item.get('id')}: {mark}")

    out_path = args.out or (os.path.splitext(args.trace)[0] + "_verdicts.json")

    def _has_response(it):
        r = it.get("response") or ""
        return bool(r.strip()) and "NOT PROVIDED" not in r

    report = {
        "model": args.model,
        "response_present": any(_has_response(it) for it in trace),
        "total_splits": len(splits),
        "verdict_counts": counts,
        "atomizer_harm_count": harm,
        "gt_fixes": gt_fixes_all,
        "verdicts": verdicts,
    }
    json.dump(report, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    adds = sum(1 for f in gt_fixes_all if f.get("action") == "add")
    removes = sum(1 for f in gt_fixes_all if f.get("action") == "remove")
    print("\n=== summary ===")
    print(f"  splits judged : {len(splits)}")
    print(f"  GOOD          : {counts['GOOD']}")
    print(f"  BLEED         : {counts['BLEED']}")
    print(f"  NONSENSE      : {counts['NONSENSE']}")
    if counts.get("PARSE_ERROR"):
        print(f"  PARSE_ERROR   : {counts['PARSE_ERROR']}   (format failure, NOT a judgement — re-run these)")
    print(f"  ATOMIZER HARM : {harm}   <-- the real do-no-harm number")
    print(f"  GT edits to review: {adds} add, {removes} remove (you apply by hand)")
    print(f"\n  written -> {out_path}")


if __name__ == "__main__":
    main()
