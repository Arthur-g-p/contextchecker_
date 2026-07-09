# ContextChecker

Claim-level evaluation for LLM outputs: decompose text into atomic claims,
then verify every claim against a reference.

Instead of asking a judge LLM "rate this answer 1-10", ContextChecker
extracts atomic `(subject, predicate, object)` claims from a response with
an extractor LLM, then classifies each claim against a reference with a
checker LLM: Entailment, Contradiction, or Neutral. Every score the toolkit
produces is an aggregate over per-claim verdicts you can read, audit, and
disagree with.

## Commands

| Command | What it does |
| --- | --- |
| `ragcheck` | RAGChecker-style RAG evaluation: 2 extractions + 4 checking directions, one self-contained JSON report (precision, recall, F1, faithfulness, and claim-level detail) |
| `faithcheck` | Faithfulness checking **without ground truth**: response claims vs the retrieved context. Works on live traffic. |
| `refcheck` | Classic reference checking: extraction + checking in one pass |
| `extract` / `check` / `atomize` | The individual building blocks, runnable standalone |
| `eval extractor` / `eval checker` | Meta-evaluation: measure the extractor and checker themselves against labeled data |

## Why this instead of a single judge score

- **Auditable**: every metric decomposes into claim-level verdicts with
  explanations. You can see exactly which fact failed and why.
- **Variance is first-class**: LLM-based metrics are stochastic. Pass
  `--runs N` to any pipeline or eval and get `mean +/- std [min, max]`
  per metric across N full repetitions, plus each run's complete report.
  A single-run score overstates your certainty.
- **The evaluator is itself evaluated**: `eval extractor` and
  `eval checker` measure the measurement tool against ground-truth
  labels, so you know how much to trust the numbers before you compare
  systems with them.
- **Honest failure handling**: per-item errors (context too long, content
  policy, parse failure) are recorded as values next to the affected claim,
  never silently zeroed. Abstentions ("I don't know") are detected and
  excluded from hallucination counts instead of being punished.
- **Crash cache**: on a fatal error (auth, connection, budget), in-flight
  LLM responses are saved to a local sqlite file; the next run resumes
  without re-paying for completed calls.
- **Any OpenAI-compatible endpoint**: point the extractor and checker at
  different models, providers, or local servers independently.

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/Arthur-g-p/contextchecker_.git
cd contextchecker_
uv pip install -e ".[test]"
```

## Quick start

Set the API keys (see `.env-example`):

```
EXTRACTOR_API_KEY=...
CHECKER_API_KEY=...
```

Input is a JSON list of items. For `ragcheck`, each item needs `response`,
`gt_answer`, and `retrieved_context`:

```json
[
  {
    "question": "Who wrote Faust?",
    "response": "Faust was written by Goethe in the 19th century.",
    "gt_answer": "Johann Wolfgang von Goethe wrote Faust.",
    "retrieved_context": ["Goethe published Faust, Part One in 1808. ..."]
  }
]
```

Run it:

```bash
contextchecker ragcheck data.json -e gpt-4o-mini -c gpt-4o-mini
```

Repeat the whole experiment five times and report variance (5x LLM cost):

```bash
contextchecker ragcheck data.json -e gpt-4o-mini -c gpt-4o-mini --runs 5
```

Check faithfulness with no ground truth at all:

```bash
contextchecker faithcheck data.json -e gpt-4o-mini -c gpt-4o-mini
```

Every command writes a single self-contained JSON report including a
`_meta` block (models, config, timings), overall metrics, and per-claim
verdicts. `contextchecker --help` lists all commands and flags.

## Documentation

- [architecture.md](architecture.md) — the full design document (layer
  model, data contract, error propagation)
- [docs/ragchecker.md](docs/ragchecker.md) — the RAG evaluation pipeline
- [docs/faithfulness.md](docs/faithfulness.md) — ground-truth-free
  faithfulness checking

## Scientific lineage

The methodology follows RefChecker (Amazon Science) and RAGChecker
(claim-level RAG evaluation), with modernized components: explicit
abstention handling, a per-item error taxonomy instead of silent nulls,
claim deduplication, and variance reporting. It is an independent
reimplementation, not a fork.

## License

MIT
