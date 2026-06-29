"""
Classify every output duplicate by CAUSE — so we stop calling legit atomization "harm".

Buckets per duplicated output triplet (s,p,o) in an item:
  PASSTHROUGH    input already had >= the duplicated count   -> extractor's redundancy, not us
  FAITHFUL_SPLIT an introduced copy's object appears inside a
                 source compound (same subject+predicate)     -> legit split of a list
                 (covers a value listed twice in one compound,
                  or shared across two compounds)
  CONTEXT_BLEED  an introduced copy's object is nowhere in any
                 source triplet's object for that (subj,pred)  -> HARM: added from the response

Only CONTEXT_BLEED is a do-no-harm violation.

IMPORTANT: pass the INPUT that produced THIS output. Do NOT mix datasets
(e.g. mini input vs big output) or every non-shared item reads as "introduced".

Usage:
    python scratch/classify_atomizer_dups.py <input.json> <output.json>
"""
import json
import re
import sys
from collections import Counter, defaultdict

SRC_KEY = "claude2_response_kg"


def load(path):
    raw = open(path, "rb").read()
    t = raw.decode("utf-8", errors="ignore").replace("\x00", "")
    t = re.sub(r",(\s*[}\]])", r"\1", t)  # tolerate trailing commas
    return json.loads(t)


def spo(t):
    if "triplet" in t:
        s, p, o = (str(x) for x in (t["triplet"] + ["", "", ""])[:3])
    else:
        s, p, o = (str(t.get(k, "")) for k in ("subject", "predicate", "object"))
    return s.strip(), p.strip(), o.strip()


def main(inp_path, out_path):
    inp = {it.get("id"): it.get(SRC_KEY) or [] for it in load(inp_path)}
    out = {it.get("id"): it.get(SRC_KEY) or [] for it in load(out_path)}

    totals = Counter()
    bleeds = []  # the only ones that matter

    for sid, out_tris in out.items():
        in_tris = inp.get(sid)
        if in_tris is None:
            print(f"  [skip] id {sid} not in input file — wrong input? skipping to avoid false positives")
            continue

        out_c = Counter(tuple(x.lower() for x in spo(t)) for t in out_tris)
        in_c = Counter(tuple(x.lower() for x in spo(t)) for t in in_tris)
        # source objects grouped by (subject, predicate) AND subjects grouped
        # by (predicate, object) — a faithful split can distribute either side.
        src_obj = defaultdict(list)
        src_subj = defaultdict(list)
        for t in in_tris:
            s, p, o = spo(t)
            src_obj[(s.lower(), p.lower())].append(o.lower())
            src_subj[(p.lower(), o.lower())].append(s.lower())

        for key, n in out_c.items():
            if n <= 1:
                continue
            s, p, o = key
            introduced = n - in_c.get(key, 0)
            if introduced <= 0:
                totals["PASSTHROUGH"] += n - 1
                continue
            # faithful if the object is inside a source compound object (same s,p)
            # OR the subject is inside a source compound subject (same p,o).
            obj_split = any(o in src_o and src_o != o for src_o in src_obj.get((s, p), []))
            subj_split = any(s in src_s and src_s != s for src_s in src_subj.get((p, o), []))
            in_compound = obj_split or subj_split
            if in_compound:
                totals["FAITHFUL_SPLIT"] += introduced
            else:
                totals["CONTEXT_BLEED"] += introduced
                bleeds.append((sid, key))

    print("\n=== duplicate causes (only CONTEXT_BLEED is harm) ===")
    for k in ("PASSTHROUGH", "FAITHFUL_SPLIT", "CONTEXT_BLEED"):
        print(f"  {k:16s} {totals.get(k, 0)}")
    if bleeds:
        print("\n  CONTEXT_BLEED instances (the real do-no-harm failures):")
        for sid, key in bleeds:
            print(f"    id {sid}: {key}")
    else:
        print("\n  no context-bleed — every duplicate is the extractor's redundancy, faithfully surfaced")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
