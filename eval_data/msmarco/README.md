# msmarco_gpt4 — human-labelled evaluation data

Ground-truth data for the `eval` commands: claim-triplets carrying **human
verdict labels**, used to measure the checker and the extractor rather than the
system under test.

## Origin

From the RefChecker paper, which published ~11k human labels on checker
verdicts. Their benchmark asked 100 questions in each of three context
settings — zero context, noisy context (MSMARCO), and accurate context — giving
300 unique questions. Those were answered by 7 LLMs, producing ~2.1k responses
and ~11k annotated claim-triplets. 23% were double-annotated, with 95.0%
inter-annotator agreement.

**This file is one cell of that grid: the GPT-4 answers under noisy context.**
Noisy context was chosen because it is the setting RAG actually operates in;
GPT-4 because its answers were the strongest available.

The original is reconstructible from the RefChecker repository.

## What was changed, and what was not

> **Five human labels were changed** — listed at the end of this file. Every
> other verdict is exactly as the RefChecker annotators left it.

The extractions around those labels were repaired. The annotation task judged
whether each *given* triplet was Entailment / Neutral / Contradiction / Abstain,
plus flagging outright bad triplets — so extraction **correctness** was checked
by humans, but extraction **completeness** never was. Three classes of problem
followed from that:

1. **Duplication** — both exact and semantic, the same fact appearing more
   than once.
2. **Missing extractions** — claims present in the response that were never
   extracted, and therefore never annotated. Found by re-running extraction and
   then atomization: the first real use case of this project.
3. **Non-atomic extractions** — triplets bundling several facts, and the
   converse, facts split further than they should have been.

Repairs were made partly with LLM assistance, but **every single change was made
under human supervision**. Triplets added to fill the completeness gap carry
labels assigned in that same supervised pass; they are additions, not
corrections to anyone else's judgement.

## What it is used for

| command | what it measures | what it reads |
| --- | --- | --- |
| `eval checker` | checker accuracy against human verdicts | `human_label` on each triplet |
| `eval extractor` | extraction quality | the triplets themselves, matched against a live extraction |

`eval checker` is the straightforward direction: the human labels are ground
truth, and the checker's verdicts are compared against them one to one.

`eval extractor` is harder, because the original extractions were never proven
complete or correct — there is no gold extraction to compare against. The
approach here is reconciliation: run extraction live, then match predicted
triplets against the file's triplets in both directions, so each side has to
find its counterpart. The lessons from the three problem classes above fed
directly back into the extraction and atomization services.

## Contents

| file | items | abstentions | triplets | Entailment | Neutral | Contradiction | always-Ent. baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `msmarco_gpt4_100.json` | 100 | 10 | 487 | 450 (92%) | 35 (7%) | 2 (0.4%) | 92.4% |
| `msmarco_gpt4_50.json` | 50 | 3 | 260 | 237 (91%) | 22 (8%) | 1 (0.4%) | 91.2% |
| `msmarco_gpt4_25.json` | 25 | 1 | 116 | 113 (97%) | 3 (3%) | 0 | 97.4% |
| `msmarco_gpt4_5.json` | 5 | 1 | 15 | 14 (93%) | 1 (7%) | 0 | 93.3% |

An **abstention** here is an item whose response declines to answer — all ten
read some form of "The passages do not provide ...". They carry an empty triplet
list rather than a label of their own, so they contribute no claims to
`eval checker`. There is no `Abstain` value in `human_label`; the empty list is
the abstention.

The last column is the score a checker would get by answering "Entailment" to
every single claim. Read any accuracy figure against it.

## Label distribution — read the checker numbers with this in mind

All 487 triplets in the 100-item file are labelled; the per-slice counts are in
the table above.

The distribution is heavily skewed, which matters when reading `eval checker`:
a checker that answered "Entailment" every single time would score about 92%
accuracy on this file. Accuracy alone is therefore close to meaningless here —
the per-class figures and the confusion matrix are what carry information, and
the two Contradiction cases are far too few to say anything reliable about
contradiction detection.

The skew is expected rather than anomalous. The paper reports the Contradiction
rate falling from 25% with zero context to 13% with noisy context and 6% with
accurate context, averaged across all seven models; this file isolates the
strongest of those models, so its error rate sits below the noisy-context
average.

The evaluation procedure is described in full in the evaluation article.

## Changed labels

| item | triplet | from | to | why |
| --- | --- | --- | --- | --- |
| `71168` | heavy marijuana use → increased risk of → depression | Entailment | Neutral | the passages support increased risk of *psychosis*; depression appears only in the reverse direction, people with depression being more likely to use cannabis |
| `57346` | bus port → refers to → USB port | Entailment | Neutral | inverts genus and species — USB is *a* bus standard; FireWire appears in the same passages as an alternative |
| `537560` | Scarsdale, New York → zip code → 10583 | Neutral | Entailment | stated outright in a postal address, "Scarsdale, New York 10583" |
| `63192` | school psychologist → can diagnose → ADHD | Entailment | Neutral | the passages name who may diagnose, and school psychologists are not among them |
| `1097044` | CRS Designation → earned by completing → 92 hours of live course instruction | Entailment | Neutral | the 92 hours belongs to the Graduate, Realtor Institute entry that begins mid-passage; CRS is awarded by a different body, and no passage gives its hours |
