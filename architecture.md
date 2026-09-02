# ContextChecker — Architecture

> Status: written 2026-07-07. Sections marked **[UNVERIFIED]** describe parts
> not re-checked against the code at writing time — verify before relying on them.

## What this package is

ContextChecker is a claim-level evaluation toolkit for LLM outputs. Its core
methodology: decompose text into atomic (subject, predicate, object) claims
via an extractor LLM, then classify each claim against a reference via a
checker LLM (Entailment / Contradiction / Neutral). Everything else — RAG
evaluation (ragcheck), reference checking (refcheck), real-time faithfulness
(faithcheck), summarization coverage — is a composition of those two moves.

## The layer model (strict)

```
CLI (controllers)  →  Pipelines / Services (orchestration)  →  Workers (execution)
```

| Layer | Owns | Does NOT own |
| --- | --- | --- |
| **CLI** (`cli.py`) | Parse flags, resolve paths, read/write files verbatim, catch errors, exit codes | Business logic, validation, output-document assembly |
| **Services** (`services/`) | Validation (dropping), filtering (skipping), payload construction, serialization into the data, delegation to one worker | File I/O, network calls |
| **Pipelines** (`pipelines/`) | Use cases composing *services* (never workers); report assembly | Direct LLM access |
| **Workers** (`workers/`) | Single-task async LLM execution, response parsing, retry rounds, per-item error classification | Orchestration, validation |
| **Evaluators** (`eval/`) | Measurement: run services on prepared data, compare against ground truth, compute metrics, assemble output documents | Mutation semantics of BaseService (they do not inherit it) |

### Foundation modules (leaf dependencies — import nothing from contextchecker)

- `settings.py` — env vars, logger factory, prompt loading. Reads eagerly,
  never crashes on missing config ("Settings reads, Services validate").
- `exceptions.py` — hierarchy: `ContextCheckerError → CLIError / ServiceError
  (InvalidInputError, FilterError) / WorkerError (LLMClientError, ParsingError,
  ContextTooLongError, ContentPolicyError)`. **[UNVERIFIED: exact leaf list]**
- `models.py` — dataclass payloads: the typed contracts between layers
  (ExtractionPayload, CheckingPayload, AtomizationPayload, Direction,
  eval result dataclasses).
- `llmclient.py` — shared LLM client (top-level, not in workers/). Owns
  connection preflight, response-format strategy discovery + process-level
  caching, and drop-params handling. Per-item errors (ContextTooLong,
  ContentPolicy, parse) are returned as *values* in the batch result list;
  fatal errors (auth, connection, budget) propagate.
  See docs/request_strategies.md.
- `stats.py` — `PhaseStats` (per-batch outcomes incl. per-index
  `error_causes`), `RoundResult`, global `TokenStats`, shared log helpers.

## Service vs Pipeline vs Worker vs Direction — the definitions

**Worker**: a dumb async execution unit. Takes payloads, calls the LLM,
parses, retries parse failures in configurable rounds (higher temperature,
then a "vanilla" prompt), classifies per-item failures into causes
(`context_too_long` / `content_policy` / `parse_failure`), exposes
`last_stats`. No validation, no orchestration. Two exist: `Extractor`,
`Checker` (plus `Atomizer`).

**Service**: orchestrates exactly one worker through the canonical 7-step
`run()` shape enforced by `BaseService` (abstract methods):

1. Canonicalize input keys (step 0: `context→reference`, `query→question`)
2. Validate (drop invalid items; raise `InvalidInputError` if none survive)
3. Filter (skip already-processed items; raise `FilterError` if none pending)
4. Log pre-execution (validation, skip, config sections)
5. Execute (delegate to the worker)
6. Serialize results back into the item dicts
7. Log results, return the mutated data

Services are **parameterized** so pipelines can re-target them without new
code: `ExtractionService(source_key, kg_key, error_key, mark_abstention)`,
`CheckingService(kg_key, verdict_namespace, extraction_error_key)`.
Defaults reproduce classic single-use behavior exactly. Services expose
their output contract as read-only properties (`verdict_key`,
`checker_error_key`, `last_stats`, ...) so composing code never touches
privates.

**Pipeline**: a `BaseService` subclass whose `run()` composes *services*
instead of driving a worker. Same caller-facing contract (data in, mutated
data out) — there is deliberately no separate pipeline base class.
Multi-run support (`_run_repeated`, the run line, the variance wiring)
lives on `BaseService` because pipelines deliberately have no own base;
bare services never trigger it. Pipelines
talk to services only, never workers. Three exist:

- `RefCheckerPipeline` — extraction + checking, one document (the classic
  reference-checking use case).
- `RagCheckerPipeline` — 2 extractions + 4 directions + metrics + report
  (see docs/ragchecker.md).
- `FaithfulnessPipeline` — 1 extraction + 1 direction, no ground truth
  (see docs/faithfulness.md).

**Direction** (`models.Direction` + `pipelines/directions.py`): the unit of
comparison in RAGChecker-style pipelines — *claims from one triplet list
checked against one reference source, verdicts written under one namespace*.
It is neither a service nor a worker: it is a **composition helper in the
pipelines layer**. It owns no validate/log/serialize lifecycle (the calling
pipeline does) and makes no LLM calls (the CheckingService it drives does).
It cannot live in `utils.py` because it calls a service, and utils sit below
services in the import graph.

Two modes:

- **Flat** (`per_chunk=False`): shadow items *share* the triplet list objects
  with the source items, so the CheckingService writes verdict keys onto the
  real triplets in place. No fold-back.
- **Matrix** (`per_chunk=True`): one shadow item per (item, chunk) over
  **deep copies** of the triplets (the same verdict key would collide across
  chunks), then per-doc results are folded back onto the original triplets as
  `{namespace}_verdicts: {doc_id: verdict}` dicts (+ `{namespace}_errors`).

Per-direction "nothing to check" (`FilterError`/`InvalidInputError` from the
service) is a normal outcome, logged and swallowed — never a pipeline crash.
A flat direction without a `reference_key` fails fast (programming error).

`directions.py` also hosts the shared input helpers: `unwrap_items` (accepts
the original RAGChecker `{"results": [...]}` envelope), `normalize_chunks`
(chunk dicts or bare strings → `{doc_id, text}`), and `phase_failure_lines`
(log formatting from PhaseStats).

## The data contract

Everything operates on one `list[dict]`, mutated in place, with
model-namespaced keys:

- Extraction writes `{extractor_model}_response_kg` (or a pipeline-chosen kg
  key such as `{extractor_model}_gt_answer_kg`): a list of
  `{subject, predicate, object}` triplet dicts.
- Checking writes onto each triplet: `{namespace}_verdict`,
  `{namespace}_explanation`, `{namespace}_error` (null-verdict cause), where
  the default namespace is `{checker_model}_checker` and pipelines use
  direction namespaces like `{checker_model}_answer2response`.
- Triplets accumulate enrichment keys over time (`human_label`, `atomized`,
  verdicts per direction) — claim-level facts ride on the claim object.
- Item-level outcome markers are sparse (absent = false / not applicable):
  see docs/outcome_markers.md for `is_abstention`, `abstention_source`,
  `{model}_extraction_error`.

**Format governance**: legacy input formats are accepted at the boundary and
converted immediately (`_canonicalize_keys`, `canonicalize_triplets`, the
`{"results": [...]}` envelope) — but new code never *emits* legacy formats.
The old GT triplet shape `{"triplet": [s,p,o], "human_label": ...}` is
input-only.

## CLI conventions

- The CLI calls the app **once**. BaseService family: `run_sync(data)` plus a
  `last_*` attribute for derived artifacts (`AtomizationService.last_trace`,
  `RagCheckerPipeline.last_report`, `FaithfulnessPipeline.last_report`).
  Evaluator family: a returned ready-to-write document (tuple of two at most —
  both evals return `(summary_doc, disagreements_doc)`).
- The CLI never composes output content — evaluators/pipelines assemble the
  full documents including `_meta`; the CLI resolves paths and dumps JSON.
- Commands: `extract`, `check`, `atomize`, `refcheck`, `ragcheck`,
  `faithcheck`, `eval extractor`, `eval checker`.

## Error propagation

```
LLMClient.generate()
  ├─ per-item error (ContextTooLong, ContentPolicy, ParseError)
  │    → returned as VALUE in the batch results list
  │    → worker classifies: permanent vs retryable, records error_causes
  │    → service persists {model}_extraction_error / {namespace}_error
  └─ fatal error (Auth, Connection, Budget)
       → re-raised → propagates through worker + service
       → CLI catches ContextCheckerError → logs → exit(1)
```

An empty result is never left ambiguous on disk: `[]` plus an error marker is
a tooling failure; bare `[]` is an abstention (see docs/outcome_markers.md).

## Logging

- `settings.get_logger(__name__)`, never print. `settings.enable_logging()`
  activates output (CLI does this; library imports stay silent by default).
- House visual language: box header, sectioned blocks (Validation / Skip /
  Config / RESULTS), tree structures, section rules
  (`settings.section_rule`). Recognition value across commands is a product
  feature — keep it consistent.
- **One printing knob, never booleans**: `verbosity: "full" | "compact" |
  "silent"` (validated in `BaseService._init_verbosity`, levels in
  `services/base.VERBOSITY_LEVELS`) plus an optional `section_label`.
  `full` = the classic standalone output (default, byte-identical for all
  single commands). `compact` = pipeline children: labeled section rule +
  API/BL results, no pre-exec sections, no per-phase token table, no done
  line — the composing pipeline owns those and prints the token table once.
  `silent` = nothing (progress bars are not logging and remain) — repeated
  runs and library calls. The former `quiet: bool` is gone.
- Evaluators run their services compact at `--runs 1` and silent in
  variance mode (per-run plumbing is cut; findings blocks still print
  every run). Data/Config announce once; the token table prints once at
  the very end. Evaluators themselves have no verbosity parameter —
  they always narrate.

## Coding rules

1. Dataclass payloads for inter-layer contracts (`models.py`).
2. Exceptions, not sys.exit(); CLI catches and exits cleanly.
3. Logging, not print.
4. Docstrings explain WHY; map pipeline steps.
5. No lazy imports for performance (CLI command-level imports and the
   package-facade wrapper are the sanctioned exceptions).
6. Async workers, sync `run_sync` wrappers.
7. pytest, `unittest.mock.patch`, never real network calls in tests.
8. Fail-fast: validate before doing work.
