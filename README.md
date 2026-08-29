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
| `extract` / `check` / `atomize` | The individual building blocks, runnable standalone — walked through in order under [examples/](examples/README.md) |
| `eval extractor` / `eval checker` | Meta-evaluation: measure the extractor and checker themselves against labeled data. Run `eval checker` first — `eval extractor` uses a checker, so an unqualified one makes its numbers partial |

## Why this instead of a single judge score

- **Auditable**: every metric decomposes into claim-level verdicts with
  explanations. You can see exactly which fact failed and why.
- **Variance is first-class**: LLM-based metrics are stochastic. Pass
  `--runs N` to `ragcheck`, `faithcheck`, `eval extractor` or
  `eval checker` and get `mean +/- std [min, max]` per metric across N
  full repetitions, plus each run's complete report. A single-run score
  overstates your certainty.
- **The evaluator is itself evaluated**: `eval extractor` and
  `eval checker` measure the measurement tool against ground-truth
  labels, so you know how much to trust the numbers before you compare
  systems with them.
- **Honest failure handling**: per-item errors (context too long, content
  policy, parse failure) are recorded as values next to the affected claim,
  never silently zeroed. A claim the checker failed to judge leaves both
  sides of the ratio rather than counting against the system under test,
  and a metric with an empty denominator is `null`, never `0.0`.
  Abstentions ("I don't know") are detected and excluded from
  hallucination counts instead of being punished.
- **Any OpenAI-compatible endpoint**: point the extractor and checker at
  different models, providers, or local servers independently.

## Installation

Requires Python 3.12+. The project uses [uv](https://docs.astral.sh/uv/) for
dependency management — install it first if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# macOS (Homebrew alternative)
brew install uv
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone and install into a virtual environment:

```bash
git clone https://github.com/Arthur-g-p/contextchecker_.git
cd contextchecker_
uv venv
uv pip install -e .
```

`uv venv` reads `.python-version` and downloads CPython 3.12 if it isn't
already present. It does not touch your system Python.

Activate the environment before using the `contextchecker` command:

```bash
source .venv/bin/activate     # macOS / Linux
```

```powershell
.venv\Scripts\activate        # Windows
```

Activation is per-shell — you need it again in every new terminal. To skip it,
prefix commands with `uv run` instead (`uv run contextchecker --help`).

To also install the test dependencies (pytest, pytest-asyncio, coverage):

```bash
uv pip install -e ".[test]"
```

Verify the install:

```bash
contextchecker --help
```

## Quick start

Copy the environment template and fill in your API keys:

```bash
cp .env-example .env
```

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

The input file is a **positional** argument on every command — pass the path
bare, with no flag in front of it.

You do not need to prepare data to start. `examples/` ships a five-step
walkthrough over one dataset, where each step needs a little more than the last,
plus `eval_data/` with human-labelled data for the `eval` commands.

Start at step 01 — decompose responses into claims:

```bash
contextchecker extract examples/extract/kepler22b.json --extractor-model gpt-4o-mini
```

Run the full RAG pipeline (step 05), which needs `gt_answer` and
`retrieved_context` as well:

```bash
contextchecker ragcheck examples/ragcheck/kepler22b.json --extractor-model gpt-4o-mini --checker-model gpt-4o-mini
```

Repeat the whole experiment five times and report variance (5x LLM cost):

```bash
contextchecker ragcheck examples/ragcheck/kepler22b.json --extractor-model gpt-4o-mini --checker-model gpt-4o-mini --runs 5
```

Check faithfulness with no ground truth at all (step 04):

```bash
contextchecker faithcheck examples/faithcheck/kepler22b.json --extractor-model gpt-4o-mini --checker-model gpt-4o-mini
```

The model flags have short aliases (`-e`, `-c`, `-m`), but the long names are
spelled the same way in every command, so those are the ones worth learning.
[examples/README.md](examples/README.md) lists which fields and which arguments
each step needs.

The pipelines and evals (`ragcheck`, `faithcheck`, `refcheck`, `eval
extractor`, `eval checker`) write a single self-contained JSON report: an
`_args` block (what you asked for — every flag of the invocation), a `_meta`
block (what the run turned out to be — counts, timings, derived keys),
metrics where the command computes any, and per-claim verdicts. The building
blocks (`extract`, `check`, `atomize`) instead emit the item list itself,
enriched in place, so each one's output is the next one's input.
`contextchecker --help` lists all commands and flags.

## Using other providers and local models

The extractor and checker are configured independently, and there are two ways
to reach a non-OpenAI endpoint.

**Direct endpoint** — pass a base URL with `--extractor-base-api` (or
`--checker-base-api`). The request goes straight out over the OpenAI SDK, and
the model name must **not** carry a provider prefix:

```bash
contextchecker extract examples/extract/kepler22b.json \
  --extractor-base-api https://openrouter.ai/api/v1 \
  --extractor-model openai/gpt-5.6-luna
```

The same flag points at any OpenAI-compatible server, including a local one
(`--extractor-base-api http://localhost:11434/v1`).

**LiteLLM routing** — omit the base URL and prefix the model with its provider
instead. LiteLLM resolves the endpoint:

```bash
contextchecker extract examples/extract/kepler22b.json --extractor-model openrouter/openai/gpt-5.6-luna
```

Prefer the direct endpoint for very new models: LiteLLM routing depends on the
installed LiteLLM release knowing the model, while a base URL bypasses that
lookup entirely. Either way the key comes from `EXTRACTOR_API_KEY` /
`CHECKER_API_KEY`.

## Documentation

- [examples/README.md](examples/README.md) — the five-step walkthrough:
  what each command needs, what it produces, and a worked dataset with
  deliberate errors planted in it
- [eval_data/msmarco/README.md](eval_data/msmarco/README.md) — provenance of
  the human-labelled evaluation data, what was repaired in it, and what it can
  and cannot measure
- [architecture.md](architecture.md) — the full design document (layer
  model, data contract, error propagation)
- [docs/request_strategies.md](docs/request_strategies.md) — how requests
  reach the model: capability discovery, retry behaviour, timeouts, and the
  settings you can tune
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
