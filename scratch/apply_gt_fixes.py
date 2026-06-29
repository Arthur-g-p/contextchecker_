"""
Apply the judge's structured GT fixes to a copy of the ground-truth file,
and print a review diff. NON-destructive (writes <gt>_cleaned.json), and
human-in-the-loop: you read the diff and decide whether to adopt the copy.

This is NOT the rejected auto-fixer: an LLM does not rewrite the GT in place.
A judge *recommends* exact ops, a deterministic script applies them to a copy,
you review. Auditable and reversible.

The judge emits ops like:
  {"id": "...", "action": "remove",  "triplet": [s, p, o], "reason": "..."}
  {"id": "...", "action": "replace", "triplet": [s, p, o], "with": [s, p, o], "reason": "..."}

Usage:
    python scratch/apply_gt_fixes.py <gt_file.json> <verdicts.json> [--key claude2_response_kg] [--out cleaned.json]
"""
import argparse
import json

KEY = "claude2_response_kg"


def spo(t):
    if "triplet" in t:
        a = (t["triplet"] + ["", "", ""])[:3]
        return tuple(str(x) for x in a)
    return tuple(str(t.get(k, "")) for k in ("subject", "predicate", "object"))


def set_spo(t, s, p, o):
    if "triplet" in t:
        t["triplet"] = [s, p, o]
    else:
        t["subject"], t["predicate"], t["object"] = s, p, o


def match(t, target):
    a = spo(t)
    b = tuple(str(x) for x in target)
    return a == b or tuple(x.strip().lower() for x in a) == tuple(x.strip().lower() for x in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_file")
    ap.add_argument("verdicts")
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt_file, encoding="utf-8"))
    report = json.load(open(args.verdicts, encoding="utf-8"))
    by_id = {str(it.get("id")): it for it in gt}
    fixes = report.get("gt_fixes", [])

    applied, skipped = [], []
    for fx in fixes:
        item = by_id.get(str(fx.get("id")))
        if not item or args.key not in item:
            skipped.append((fx, "item or key missing"))
            continue
        tris = item[args.key]
        idx = next((i for i, t in enumerate(tris) if match(t, fx.get("triplet", []))), None)
        if idx is None:
            skipped.append((fx, "triplet not found (already fixed? or text drift)"))
            continue
        action = fx.get("action")
        if action == "remove":
            removed = tris.pop(idx)
            applied.append(("REMOVE", fx.get("id"), spo(removed), None))
        elif action == "replace":
            w = fx.get("with")
            if not w or len(w) < 3:
                skipped.append((fx, "replace missing 'with'"))
                continue
            before = spo(tris[idx])
            set_spo(tris[idx], str(w[0]), str(w[1]), str(w[2]))
            applied.append(("REPLACE", fx.get("id"), before, tuple(str(x) for x in w[:3])))
        else:
            skipped.append((fx, f"unknown action '{action}'"))

    out = args.out or (args.gt_file.rsplit(".", 1)[0] + "_cleaned.json")
    json.dump(gt, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("=== applied (REVIEW these) ===")
    for kind, sid, before, after in applied:
        if kind == "REMOVE":
            print(f"  REMOVE  id {sid}: {before}")
        else:
            print(f"  REPLACE id {sid}: {before} -> {after}")
    if skipped:
        print("\n=== NOT applied (look by hand) ===")
        for fx, why in skipped:
            print(f"  [{why}] id {fx.get('id')}: {fx.get('triplet')}  ({fx.get('reason', '')})")

    print(f"\n  applied {len(applied)}, skipped {len(skipped)}")
    print(f"  cleaned GT -> {out}   (original untouched — diff it, then adopt if good)")


if __name__ == "__main__":
    main()
