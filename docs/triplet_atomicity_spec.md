# Triplet Atomicity Specification

Governs the `Atomizer` (`workers/atomizer.py`) prompt and its few-shot examples.
Language-independent: every rule is stated as a **semantic** test, not a keyword
test. Examples are illustrative (English) but the decision is about meaning, so
the same rules apply in any language.

---

## 1. Purpose

Split a knowledge-graph triplet that bundles **more than one independent fact**
into the minimum set of atomic triplets — and leave everything else untouched.
This is a cleanup pass, not a rewrite, not a fixer, not a judge.

---

## 2. Definition

A triplet `(subject, predicate, object)` is **atomic** iff it asserts exactly
**one indivisible fact**: one subject entity, one relation, one object
entity/value, such that you cannot remove or separate any part without losing
or changing information.

A triplet is **non-atomic** iff it can be rewritten as N>1 triplets that,
taken together, mean *exactly* the same thing, and each of which is
*independently* true.

---

## 3. Prime directive — do no harm (ship gate)

The atomizer runs **destructively, in the pipeline, without its own eval**. It is
only allowed to do that if it provably cannot make quality worse. Three
invariants, in priority order:

- **I1 — Information is conserved.** The union of the outputs must be logically
  equivalent to the input. **Never drop a fact** (Apollo) and **never add one**
  (no pulling tokens from the response context).
- **I2 — Roles and direction are preserved.** Never swap subject/object, never
  reword or invert the predicate, never paraphrase. Output S/P/O are the input's,
  re-partitioned — not regenerated.
- **I3 — Bias to KEEP.** When uncertain at any step, return the triplet
  unchanged. A wrong KEEP is harmless (a no-op); a wrong SPLIT destroys
  information. KEEP is the safe default.

If a triplet cannot be split without violating I1–I2, it is **not** a candidate.

---

## 4. Decision procedure (apply in order, per triplet)

1. **Does the triplet assert more than one independent fact?** If no → KEEP.
2. If it *looks* compound, classify the compound and route:
   - Collective / relational conjunction → **KEEP** (§6.1)
   - Compound entity name / single named value → **KEEP** (§6.2)
   - Hedge / qualifier / glued modifier → **KEEP** (§6.3)
   - Nested / reified proposition as object → **KEEP** (§6.4, open Q)
   - Uncertainty "or" (exclusive / unknown-which) → **KEEP** (§6.5, open Q)
   - Genuine **distributive** conjunction (predicate holds for each part
     *alone*, parts mutually independent) → **SPLIT** (§5)
3. If still uncertain → KEEP (I3).
4. Context (the `response`) may be used **only** to resolve a reference
   (e.g. a pronoun: "her mom" → "Grace VanderWaal's mom"). It may **never**
   contribute a new subject, predicate, or object token.

---

## 5. SPLIT — the only cases that decompose

Split **only** a *distributive* conjunction: a list where the predicate applies
to each element on its own and the elements are not bound to each other.

Test: rewrite as one triplet per element. Is each individually true, and does
their union mean exactly the original? If yes → split.

| Input | Output | Why |
|---|---|---|
| `(dog, eats, "meat and vegetables")` | `(dog, eats, meat)` · `(dog, eats, vegetables)` | predicate holds for each object alone |
| `("cats and dogs", are, mammals)` | `(cats, are, mammals)` · `(dogs, are, mammals)` | distributive subject, no mutual relation |

Each output keeps the original subject, predicate, and direction. Only the
distributed slot changes.

---

## 6. KEEP — the dangerous look-alikes (these caused our bugs)

### 6.1 Collective / relational conjunction  → KEEP
The predicate expresses a relation **among** the conjuncts or a **joint**
property. Splitting destroys the relation.
- `("Apollo and Phaedra", marital status, still married)` — they are married
  **to each other**. `(Apollo, …, married)` + `(Phaedra, …, married)` loses that.
  **KEEP.**
- Same shape: "A and B met", "A and B are siblings", "X and Y are equal/similar".

### 6.2 Compound entity name / single named value  → KEEP
A multi-word name or a single descriptive value is one unit.
- `"Adjusted Gross Income (AGI)"`, `"United States of America"`,
  `"50 percent of AGI"` — atomic objects. **KEEP.**

### 6.3 Hedge / qualifier / glued modifier  → KEEP
A qualifier attached to a value is not a separate fact.
- `(snakes, have odor, "very little -- if any")` — "if any" hedges "very little";
  it is not an independent object. **KEEP** (do **not** emit `…, none`).
- `(X, associated with, "increased risk of depression")` — "increased risk of"
  is glued to "depression"; do not peel it. **KEEP.**

### 6.4 Nested / reified proposition as object  → KEEP  *(open — see §8)*
When the object is itself a claim, the triplet is *about the claim*, not the
claim's content.
- `("some studies", suggest, "CBD marijuana can help mental health conditions")` —
  asserts that studies *suggest* X, **not** that X is true. Extracting X as its
  own triplet changes the assertion. **KEEP.**

### 6.5 Uncertainty / exclusive "or"  → KEEP  *(open — see §8)*
- "the answer is X or Y" (we don't know which) — splitting would assert both.
  **KEEP.** (Contrast: an *enumerative* "or" where both genuinely hold may be
  distributive — this is the open call in §8.)

### 6.6 Conditional / scoped subject  → KEEP
- `("snakes whose waste is not cleaned up", can start to, smell)` — the
  condition scopes the subject into one complex entity. **KEEP.**

---

## 7. Not atomicity's concern — pass through unchanged

- **Malformed triplet** (junk predicate, fragment, e.g.
  `(snakes, "odor when not kept clean", present)`): the atomizer is **not a
  fixer**. Fixing would hide bad extractors and corrupt the benchmark. Pass it
  through untouched.
- **Duplicates**: a separate dedup step (bidirectional entailment). Not here.
- **Wording / quality**: never paraphrase.

---

## 8. Open policy questions (decide before finalizing the prompt)

1. **Enumerative "or"** — `(odor, similar to, "vinegar or cucumbers")`: split into
   two "similar to" facts, or keep? (The GT itself stores them as two separate
   triplets, which *implies* split — but "or" carries uncertainty.)
2. **Nested claims (§6.4)** — always keep `(studies, suggest, "…")` whole, or is
   there ever a case to also emit the inner claim (flagged as reported, not
   asserted)?
3. **Modifier-peeling depth** — how aggressively to treat object-glued modifiers
   as atomic vs. separable.

---

## 9. Observed failure modes (the negative tests)

| Failure | Example | Violates |
|---|---|---|
| Collective-relation split | Apollo → two marriages | I1 (drops "to each other") |
| Hedge split | "very little -- if any" → "very little" + "if any" | I1 (phantom fact) |
| Context-bleed | marijuana risk → +depression/+psychosis pulled from response | I1 (adds facts) |
| Role inversion | "owned by her mom" → "owns her mom" *(v1, fixed)* | I2 |
| Self-duplication | emitting the same atomic twice | I1 (redundancy) |

---

## 10. Regression fixture

The `mini_msmarco` GT set contains **~zero** genuinely distributive triplets —
every compound-looking case is collective (Apollo), hedged ("if any"),
object-glued ("increased risk of …"), nested ("studies suggest …"), or already
pre-split (vinegar / cucumbers). **The correct output on this set is the input,
unchanged.** Use it as a "must-not-split" regression fixture: any split on this
set is a do-no-harm failure.
