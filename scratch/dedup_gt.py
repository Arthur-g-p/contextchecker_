"""
Deterministic exact-duplicate triplet removal — no LLM, no judgment.

A duplicate triplet carries zero information, so dropping repeats is mechanical
and provably loss-free (keeps the first occurrence). Works on any triplet set:
the raw GT, or the atomized output (point --key at the right key).

Note: this removes whole-triplet duplicates. An internal-list redundancy like a
compound object listing "Turkish" twice is NOT a whole-triplet dup — but once that
compound is atomized it BECOMES two identical triplets, which this then removes.
So: atomize, then dedup, and every redundancy collapses.

Non-destructive: writes <file>_dedup.json and prints what it removed.

Usage:
    python scratch/dedup_gt.py <file.json> [--key claude2_response_kg] [--out ...]
"""
import argparse
import json

KEY = "claude2_response_kg"


def norm(t):
    if "triplet" in t:
        a = (t["triplet"] + ["", "", ""])[:3]
        return tuple(str(x).strip().lower() for x in a)
    return tuple(str(t.get(k, "")).strip().lower() for k in ("subject", "predicate", "object"))


def spo(t):
    if "triplet" in t:
        return tuple((t["triplet"] + ["", "", ""])[:3])
    return tuple(t.get(k, "") for k in ("subject", "predicate", "object"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))
    removed = []
    for it in data:
        tris = it.get(args.key)
        if not isinstance(tris, list):
            continue
        seen, kept = set(), []
        for t in tris:
            k = norm(t)
            if k in seen:
                removed.append((it.get("id"), spo(t)))
                continue
            seen.add(k)
            kept.append(t)
        it[args.key] = kept

    out = args.out or (args.file.rsplit(".", 1)[0] + "_dedup.json")
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("=== removed exact-duplicate triplets ===")
    for sid, t in removed:
        print(f"  id {sid}: {t}")
    print(f"\n  removed {len(removed)} duplicate(s) across {len(data)} items")
    print(f"  deduped -> {out}   (original untouched)")


if __name__ == "__main__":
    main()
