# Examples

Small, readable inputs for learning the commands one at a time. Each folder is
named after the command it feeds, so the directory tells you what to run.

Outputs land in a `results/` folder next to the input. `results/**` is untracked,
so running these never dirties your checkout.

All five files carry the same eight items about the exoplanet Kepler-22b, with
byte-identical responses. Only the *fields around them* grow. That is the
point of the walkthrough: each step needs strictly more data than the last, and
you can see exactly what each new field buys you.

## The journey

| step | folder | command | data the file must carry | arguments |
| --- | --- | --- | --- | --- |
| 01 | `extract/` | `extract` | `response` | **models** `--extractor-model`<br>**endpoints** `--extractor-base-api` |
| 02 | `check/` | `check` | `response`, `reference`,<br>extracted claims | **models** `--extractor-model` *(required)*, `--checker-model`<br>**endpoints** `--checker-base-api` |
| 03 | `refcheck/` | `refcheck` | `response`, `reference` | **models** `--extractor-model`, `--checker-model` *(both required)*<br>**endpoints** `--extractor-base-api`, `--checker-base-api` |
| 04 | `faithcheck/` | `faithcheck` | `response`, `retrieved_context` | **models** `--extractor-model`, `--checker-model` *(both required)*<br>**endpoints** `--extractor-base-api`, `--checker-base-api` |
| 05 | `ragcheck/` | `ragcheck` | `response`, `retrieved_context`, `gt_answer` | **models** `--extractor-model`, `--checker-model` *(both required)*<br>**endpoints** `--extractor-base-api`, `--checker-base-api` |

Every command also accepts `--output`/`-o`, `--concurrency`, and `--debug`.
Model flags have short aliases (`--model`, `-m`, `-e`, `-c`), but the long names
above are spelled the same way in every command, so they are the ones worth
learning.

`.vscode/launch.json` ships one debug configuration per step, in this order,
with the models and endpoints as prompts.

---

## 01 · extract — `extract/kepler22b.json`

Eight answers about the exoplanet Kepler-22b. `extract` decomposes each
`response` into atomic `(subject, predicate, object)` claims.

**Needed field: `response`.** The `question` field is optional for this command
and is included only so the items read naturally.

```bash
contextchecker extract examples/extract/kepler22b.json --extractor-model <your-model>
```

Read the output claim by claim. A good response should split into several
claims; a short one into one or two.

Extraction only *pulls claims out*. It does not judge them. Nothing here tells
you whether a claim is true — that is step 02, and keeping the two apart is
deliberate.

## 02 · check — `check/kepler22b_extracted.json`

The extracted claims now get checked against their references, which is why
`reference` becomes a needed field.

The shipped file is exactly what step 01 emits, with a `reference` added — so
you can equally point `check` at your own `results/kepler22b_extract.json` once
you have added references to it.

```bash
contextchecker check examples/check/kepler22b_extracted.json \
  --extractor-model openai/gpt-5.6-luna \
  --checker-model <your-model>
```

`--extractor-model` here is **required** and easy to misread: it names the model
that *produced* the claims, not the one you want to judge them. The claims live
under a key derived from it (`openai/gpt-5.6-luna_response_kg`), and that is how
`check` finds them. The model doing the judging is `--checker-model`.

## 03 · refcheck — `refcheck/kepler22b.json`

Extraction and checking in one go, with no artifact file in between. The same
methods and the same prompts are used, so the output should be more or less
identical to running 01 and 02 back to back.

```bash
contextchecker refcheck examples/refcheck/kepler22b.json \
  --extractor-model <your-model> \
  --checker-model <your-model>
```

Use 01 + 02 when you want the intermediate claims on disk to inspect or edit.
Use `refcheck` when you only care about the verdicts.

One difference in the *file*, not the verdicts: steps 01 and 02 emit the bare
item list, so each step's output feeds straight into the next. `refcheck` is a
pipeline, not a building block, so it emits a report — `_args`, `_meta`, and
the items under `results` — the same envelope `ragcheck` and `faithcheck`
write. It also accepts that envelope as input.

## 04 · faithcheck — `faithcheck/kepler22b.json`

This checks which claims are actually **faithful to the context**. No ground
truth is needed yet — which is what makes it usable on live traffic, where no
reference answer exists.

```bash
contextchecker faithcheck examples/faithcheck/kepler22b.json \
  --extractor-model <your-model> \
  --checker-model <your-model>
```

The file is the step 05 file with `gt_answer` removed and nothing else changed,
so comparing the two reports isolates exactly what ground truth buys you.

Without it there is no correctness axis: no `precision`, `recall`, `f1`,
`hallucination` or `self_knowledge`. What survives is `faithfulness` — is this
claim grounded in the retrieved chunks — plus the reliability metrics.

`claim_support` in the report lists, per claim, the chunks that entail it. An
empty list means nothing grounded that claim.

## 05 · ragcheck — `ragcheck/kepler22b.json`

The core feature. It needs the full set of variables:

- a **`gt_answer`** — the ground-truth answer
- a **`retrieved_context`** — the chunks the RAG system retrieved
- an **id** — `query_id` (or `id`) is technically optional and can be added at
  any time, but this is the step where you want it, because it is what makes an
  item identifiable downstream

```bash
contextchecker ragcheck examples/ragcheck/kepler22b.json \
  --extractor-model <your-model> \
  --checker-model <your-model>
```

### On the retrieval metrics in this example

Every item here gets the same four chunks, so there is no retriever being
exercised. Read those two metrics accordingly:

- `claim_recall` still means something — it answers "does this reference
  text actually support the ground truth?"
- `context_precision` flattens, because every chunk is nominally in play for
  every question. Ignore it here.

---

## There are deliberate errors in the responses

**Not every answer is correct, and two of them are not answers at all.**
The files were written to contain a mix on purpose, because the whole point of
claim-level evaluation is that you can *see* which individual fact failed
instead of getting one score.

Hidden in the items there is:

- one claim that is **flatly wrong** — it conflicts with the source material
- claims that are **unsupported** — not contradicted, just never backed up
  (neutral). One of them is also *false*; another is perfectly *true* and still
  unsupported, which is a different thing again
- two items that are an **abstention** — the model declining to answer

Try to spot them yourself first. Extraction alone will not label them — it only
pulls the claims out. You need `check` (step 02) to get verdicts, and that is
exactly the point: extraction and checking are separate jobs.

<details>
<summary>Answer key — open once you have looked</summary>

- **Item 5 (orbital period) is wrong.** It says one year lasts about 365 days.
  The real orbital period is about 290 days. Expect **Contradiction**.
- **Item 1 (distance) is unsupported.** The distance itself — 640 light-years,
  200 parsecs — is correct and should come back **Entailment**. But the answer
  also calls Kepler-22b a *super-Earth*, which the source never says. Expect **Neutral**, not
  Contradiction. One response, two different verdicts.
- **Item 6 is an abstention.** "I don't know." is detected as a full abstention
  and produces no claims.
- **Item 8 is also an abstention**, worded differently: "I have no idea what
  type of star Kepler-22 is". Detection has two paths — a cheap heuristic that
  skips the LLM call on obvious refusals, and the extractor itself, which
  returns zero claims when there is nothing factual to pull out. Item 6 is
  caught by the first, item 8 by the second. Both end up as abstentions.

Items 2, 3, 4 and 7 are accurate. Item 4 is the densest: mass unknown, both
sigma limits, and the radius revision from 2.4 to 2.1 Earth radii. Item 7 is
the thinnest — a single claim — but the context never mentions the Milky Way,
so it is correct without being grounded.

</details>

`edge_cases/` holds hand-made inputs for failure paths (all-abstained,
all-invalid, mixed state) and is not part of the walkthrough.

---

## What `gt_answer` adds — the step 05 differences

Every file carries the same eight responses, so the same answers travel the
whole journey and any difference between two reports comes from the fields, not
from the data.

What `gt_answer` adds is a second axis. `check` only asked "is this claim
supported by the reference". `ragcheck` also asks "did the response say what it
should have said", and those are different failures.

| # | question | how gt_answer differs from the response | what it exercises |
| --- | --- | --- | --- |
| 1 | distance | gt gives the distance only. The response also calls it a *super-Earth* — an extra claim with no counterpart in gt. | **hallucination** — *super-Earth* is neither in context nor in the gt: ungrounded and unverifiable by the gt (independent if it is a true fact!) |
| 2 | discovery | close match. gt is slightly fuller: it adds "where liquid water could exist on the surface". | the clean baseline — **precision** and **faithfulness** with nothing pulling them down. |
| 3 | constellation | gt is one sentence (Cygnus). The response adds the host star being too dim — true, but outside gt. | **noise_sensitivity_in_relevant** — the extra claim is grounded in a chunk but absent from gt, so it costs precision without being a hallucination. |
| 4 | mass | gt keeps both sigma limits and stops there. The response also carries the radius revision 2.4 -> 2.1. | **recall** vs **precision** pulling apart — gt is fully covered, and the response carries more besides. |
| 5 | orbital period | gt says **290 days**. The response says 365. Direct conflict. | **hallucination** again, via the only outright **Contradiction** in the set. |
| 6 | surface composition | gt says the composition is unknown, and the response declines to answer. | **abstention_rate** and **unjustified_abstention_rate** — evidence was retrieved, the model declined. |
| 7 | Milky Way | gt confirms it. The source context never states it — so a correct response here is unsupported by context. | **self_knowledge** — correct against gt, absent from the context. The cleanest single-claim case. |
| 8 | star type | gt gives G-type, mass, volume, 5,518 K. The response abstains anyway. | **unjustified_abstention_rate** again, reached through the extractor rather than the heuristic. |

Items 3 and 4 are the interesting pair: the responses are *correct* but say
things gt does not. That is not an error, it is how precision and recall pull
apart. A response can be fully faithful and still score badly for coverage, or
cover everything and still assert something unsupported.

Item 7 is the cleanest single-claim case in the set - one claim, correct against
gt, absent from the context.

### On the two abstentions

Both items 6 and 8 come back as **unjustified** abstentions. An abstention counts
as justified only when the retriever gave the generator nothing to work with
(item `claim_recall` = 0). Here the context discusses both surface composition
and the host star, so the evidence was there and the model declined anyway.

Item 6 is worth thinking about: the true answer *is* "the composition is
unknown", so "I don't know" is arguably correct content rather than a refusal.
The abstention heuristic sees only the string and cannot tell those two apart.

### Note on pronouns

Items 2 and 4 open with "It" and "Its". The extractor only ever sees the
response, never the question, so it cannot know what "it" refers to - the
subject comes out vague. This is a flaw in the *answer*, not in the extractor:
an answer (in the response or gt answer) that leans only on a pronoun is not will never produce a self-contained atomic claim. Worth writing answers that name their subject for stability. 