"""
Find duplicate triplets around an atomize run.

Distinguishes:
  (A) input GT dups        — already in the source; atomizer passes them through
                             (correct — dedup is a SEPARATE step, not the atomizer's job)
  (B) INTRODUCED dups      — appear more times in the output than in the input
                             (a do-no-harm violation: the atomizer created redundancy,
                              e.g. context-bleed pulling a fact from the response)

Usage:
    python scratch/check_atomizer_dups.py \
        results/mini_msmarco_gpt4_answers_corrected.json \
        results/results/mini_msmarco_gpt4_answers_corrected.json
"""
import json
import sys
from collections import Counter

SRC_KEY = "claude2_response_kg"  # adjust if your source kg key differs


def flat(t: dict) -> tuple:
    """Normalize a triplet (legacy [s,p,o] or canonical {s,p,o}) to a comparable key."""
    if "triplet" in t:
        return tuple(str(x).strip().lower() for x in t["triplet"])
    return tuple(str(t.get(k, "")).strip().lower() for k in ("subject", "predicate", "object"))


def by_id(data: list) -> dict:
    return {it.get("id"): (it.get(SRC_KEY) or []) for it in data}


def main(inp_path: str, out_path: str) -> None:
    inp = by_id(json.load(open(inp_path, encoding="utf-8")))
    out = by_id(json.load(open(out_path, encoding="utf-8")))

    print("=== (A) INPUT dups — passthrough, not the atomizer's fault ===")
    in_total = 0
    for sid, triplets in inp.items():
        c = Counter(flat(t) for t in triplets)
        for k, v in c.items():
            if v > 1:
                print(f"  id {sid}: x{v}  {k}")
                in_total += v - 1
    print(f"  -> {in_total} redundant input triplets\n")

    print("=== (B) INTRODUCED dups — atomizer harm (output count > input count) ===")
    introduced = 0
    for sid, triplets in out.items():
        co = Counter(flat(t) for t in triplets)
        ci = Counter(flat(t) for t in inp.get(sid, []))
        for k, v in co.items():
            if v > 1 and v > ci.get(k, 0):
                print(f"  id {sid}: x{v} out (input x{ci.get(k, 0)})  {k}")
                introduced += v - max(ci.get(k, 0), 1)
    if not introduced:
        print("  none — every output dup traces to an input dup (clean passthrough)")
    else:
        print(f"  -> {introduced} INTRODUCED duplicate(s): do-no-harm FAIL")
    print()

    print("=== per-item count delta (output - input) ===")
    for sid in inp:
        di, do = len(inp[sid]), len(out.get(sid, []))
        if di != do:
            print(f"  id {sid}: {di} -> {do}  ({do - di:+d})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
