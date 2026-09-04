# Output conventions — data display rules

Internal document. How numbers are allowed to appear in this project's
terminal output. Every result block must be an instance of exactly one
display method below, obeying that method's rule set. When a print feels
wrong, it is breaking one of these rules — find which.

The reference implementation of the house style is the extractor eval's
`🔎 Matching Quality` block: it obeys every MECE-tree rule and predates
this document.

## Rule set 0 — universal laws

These hold across all display methods.

1. **One number, one home.** Every count lives in exactly one block —
   the block of its dimension (model behavior, tooling health, …).
   Including a count here means excluding it everywhere else. Blocks
   over the same universe must be exhaustive *together* and declare
   their shared denominator (the header states what the total counts
   and what it excludes). Derived rates are not counts: a footer rate
   may be echoed by the variance block, but only under the same name
   and only when its derivation is visible per run.

2. **Rates are 0-1 decimals, everywhere.** The variance block prints
   `0.125`, tree footers print `0.125`, the JSON stores `0.125` — no
   display may speak percent. One number format for rates across the
   whole system.

## Rule set 1 — MECE trees

For counts that partition a total: mutually exclusive, collectively
exhaustive.

1. **Branches carry counts that partition the header total.** The header
   states the total; the branches sum to it — always. If a case exists
   that no branch covers (e.g. an abstention whose evidence is unknown),
   the tree gets an explicit `❓ uncategorized` branch rather than a
   silent hole in the sum. The footer is **single and terminal** — one
   `→` line that closes the tree; never per-branch mini-footers
   interleaved between branches (they break the branch column and turn
   a tree-level derivation into a fake branch property). It derives the
   rate, with its fraction visible: `→ Recall 0.964  (27 / 28)`. Variance (`--runs`)
   aggregates **only footer rates, under the same name** — counts never
   cross runs, and no rate enters the variance block that is not a
   visible footer derivation in the per-run output. The footer is
   required when the tree's metric feeds variance; trees with no
   cross-run metric may omit it (do not retrofit footers for their own
   sake).
2. **Round brackets are the interpretation aid.** Each branch may carry
   one `(…)` describing what the case means for the user —
   `(GT empty, 0 predictions — correct silence)`. Optional, one per
   branch, never load-bearing: the count and label must stand alone.
3. **Branch anatomy: icon · count · plain-English label.** Every branch
   (main and sub) is `<icon> <number> <label>`, label in plain English —
   no underscores, no JSON key names. The mapping from display label to
   JSON key is the code's job, not the reader's.
4. **Foreign numbers go in square brackets only.** A number that is not
   part of the tree's count may appear only as `[…]` and must carry
   real informational value: `7 tasks sent to LLM [7 HTTP requests]`.
   If it neither partitions the total nor informs, it does not print.

## Rule set 2 — rate rows

For independent reliability measurements with different denominators
(items, verdicts, claims). Rendered by `stats.log_rate_rows`.

1. **Each row is its own derivation**: `<icon> <label>: <count> of
   <denominator> <phrase> → <rate_key> <rate>`. Rows never sum and never
   pretend to — glyphs group, they do not partition. No terminal footer:
   every row already is one.
2. **The rate names its key.** `→ extraction_error_rate 0.125` states
   the exact variance/JSON key this number becomes — full lineage from
   terminal line to report field. Rows whose rate is not exported yet
   omit the arrow rather than invent a name.
3. **Always printed.** `0 of 52` is information; an unconfigured axis
   prints `not measured (reason)`. A hidden row is indistinguishable
   from an unmeasured one — hidden ≠ zero.
4. **Causes are sub-branches**, plain English, sorted by count.

## Rule set 3 — grouped metric trees

For scalar metric families (ragcheck / faithcheck 📊 Metrics).

1. **The header declares the universe** (`macro over 8 items`); a metric
   computed over fewer carries a `(x of y items)` round bracket — never
   jargon like `n=7`. The *why* of the gap lives in the sibling
   behavior/reliability blocks, not here.
2. **Groups are bare headers with one em-dash explanation** — plain
   orientation for the reader (`Retriever — did retrieval bring the
   needed evidence`). Each group's branch glyphs close (`└─`); glyphs
   group, they never imply summation.
3. **Labels are full plain English** — no underscores, no
   abbreviations; the column widens before a word shortens. The
   variance block repeats each rate under exactly the name its per-run
   line printed: plain label for tree and footer rates, raw key for
   rate rows and `→ key value` lines.
4. **A single-metric tree** is the degenerate case: one `└─` row, which
   may carry an inline interpretation in its bracket.

## Rule set 4 — the variance block (--runs)

Rendered by `stats.log_variance_block` from each command's
`_VARIANCE_SECTIONS` spec; keys filtered and ordered by
`utils.build_variance(roster=…)`.

1. **Membership is bidirectional.** Every named run-outcome rate printed
   per run enters the variance block, and nothing else does. Adding a
   rate to a per-run block means adding it here; wanting it here means
   giving it a per-run derivation first. Exception: data constants
   (e.g. the majority baseline) are printed rates but not run outcomes —
   they stay out.
2. **Normalized quantities only.** Per-item, per-claim, per-run rates
   and ratios — never raw counts. A constant with a ± on it is noise.
3. **Per-run value, then aggregate.** Each run computes its own rate;
   the block reports mean ± std [min, max] across runs. Pooling has no
   std — it is not a simplification of the right method, it is a
   different method, and wrong here. Null runs leave the mean and are
   annotated (`2/3 runs`).
4. **The block mirrors the per-run structure**: 📊 Metrics (same groups
   as the per-run 📊 block) → ⚪ Behavior (the ⚪ tree's footer rates) →
   💥 Health (the 💥 rows; should-be-zero framing) → ⏱ Time. A section
   exists iff its per-run block exists for the command. Labels are
   plain English, matching the per-run vocabulary.
5. **The run line's keys are a subset of the Metrics roster.**
6. **The zero-variance caching warning reads Metrics only** — all-zero
   Health is the desired state, not caching evidence.

## Other display methods — pending codification

Contracts sketched, to be promoted to rule sets as each is unified:
- **Label table** (checker eval per-label report): label × measure
  grids only; zero-support rows must be annotated (they poison macro
  averages).
- **Matrix** (confusion matrix): confusion only; full labels, house
  alignment, marginal totals on both axes — the corner total must
  reconcile with its sibling tree (judged = correct + wrong).
- **Prose stat block** (Extraction Stats, Atomicity, Duplicates): 2–4
  fixed lines, at most one derived rate, no branch glyphs.
- **Headline scalar**: always with its fraction —
  `→ accuracy 0.803  (102 / 127)` — never bare. A derived scalar with
  no fraction (F1) names its inputs instead:
  `f1: 0.940  (harmonic mean of recall · precision)`.
- **Ordering**: findings blocks → `✅ Done` → `── Execution Stats ──`.
  The Done line closes the findings; the token table is the appendix.
