"""
CLI controllers — the I/O boundary of the package.

Responsibilities (and nothing more):
- Parse Typer flags and arguments
- Resolve file paths
- Print the box header and output path
- Call the appropriate service
- Catch errors and exit cleanly

All business logic lives in services. All execution lives in workers.
The CLI does NOT format validation/results output — services own that.
"""

import json
from pathlib import Path

import typer

from contextchecker import settings
from contextchecker.exceptions import ContextCheckerError

logger = settings.get_logger(__name__)

app = typer.Typer(
    name="contextchecker",
    help="Claim-level evaluation for LLM outputs: extract atomic claims, then verify each against a reference.",
    no_args_is_help=True,
)


def _print_header(command: str) -> None:
    """Print the branded box header."""
    title = f"  ContextChecker · {command}"
    width = max(len(title) + 2, 50)
    logger.info("╭" + "─" * width + "╮")
    logger.info("│" + title.ljust(width) + "│")
    logger.info("╰" + "─" * width + "╯")
    logger.info("")


# Canonical order for the _args block: identity, then each model role with its
# endpoint, then keys, then behaviour, then runtime. Anything not listed is
# appended alphabetically, so a new flag lands predictably instead of mid-list.
_ARGS_ORDER = (
    "command",
    "input_file",
    "output_file",
    "extractor_model",
    "extractor_base_api",
    "checker_model",
    "checker_base_api",
    "atomizer_model",
    "atomizer_base_api",
    "gt_key",
    "source_kg_key",
    "joint",
    "joint_num",
    "max_words",
    "dedup",
    "runs",
    "concurrency",
    "debug",
)

# Params are captured wholesale from Click, so a future credential flag would
# otherwise land in every report on disk.
_ARGS_DENYLIST = frozenset({
    "api_key", "extractor_api_key", "checker_api_key", "atomizer_api_key",
    "token", "password", "secret",
})


def _capture_args(command: str) -> dict:
    """The ``_args`` block: every parameter of the current invocation.

    Read from Click rather than enumerated by hand, so it cannot drift when a
    flag is added. ``_explicit`` names the keys that were actually typed —
    ``joint=true`` given on the command line and ``joint=true`` by default are
    different facts when reconstructing a run later.
    """
    import click

    ctx = click.get_current_context(silent=True)
    params = dict(ctx.params) if ctx else {}
    explicit = []
    if ctx:
        from click.core import ParameterSource
        explicit = sorted(
            name for name in params
            if name not in _ARGS_DENYLIST
            and ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE
        )

    values = {"command": command}
    for name, value in params.items():
        if name in _ARGS_DENYLIST:
            continue
        values[name] = str(value) if isinstance(value, Path) else value

    known = [k for k in _ARGS_ORDER if k in values]
    unknown = sorted(k for k in values if k not in _ARGS_ORDER)
    ordered = {k: values[k] for k in known + unknown}
    ordered["_explicit"] = explicit
    return ordered


def _resolve_output(
    input_file: Path,
    operation: str,
    explicit: Path | None = None,
    runs: int = 1,
) -> Path:
    """Resolve the output path: results/{stem}_{operation}[_{runs}].json.

    The runs suffix only appears for --runs > 1, and only the four commands
    that accept the flag ever pass it. An explicit -o path skips the naming
    but still gets its parent created and its clobber warned about.
    """
    if explicit is not None:
        path = explicit
    else:
        suffix = f"_{runs}" if runs > 1 else ""
        path = (
            input_file.parent / "results" / f"{input_file.stem}_{operation}{suffix}.json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.warning(" ⚠️  %s exists — it will be overwritten", path)

    return path

@app.command()
def extract(
    input_file: Path = typer.Argument(..., help="Path to JSON input file."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_stem}_extract.json."),
    model: str = typer.Option(None, "--extractor-model", "--model", "-m", help="Model name used as key prefix."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets from the output. On by default (loss-free cleanup)."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run the extraction pipeline on a JSON dataset."""
    from contextchecker.services.extraction import ExtractionService

    # Activate logging (pretty by default, debug if --debug)
    settings.enable_logging(debug=debug)

    _print_header("extract")

    output_file = _resolve_output(input_file, "extract", output_file)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call service
    try:
        service = ExtractionService(model=model, base_url=extractor_base_api, dedup=dedup, concurrency=concurrency)
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)


@app.command()
def check(
    input_file: Path = typer.Argument(..., help="Path to JSON file with extracted triplets."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_stem}_check.json."),
    model: str = typer.Option(None, "--checker-model", "--model", "-m", help="Model name for the checker LLM."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model name that was used for extraction (to find the kg_key)."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Use joint checking (multiple claims per LLM call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per LLM call. Default: 6000 in joint mode, unset in single."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Check extracted claims against reference passages."""
    from contextchecker.services.checking import CheckingService

    # Activate logging
    settings.enable_logging(debug=debug)

    _print_header("check")

    output_file = _resolve_output(input_file, "check", output_file)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call service
    try:
        service = CheckingService(
            concurrency=concurrency,
            model=model,
            extractor_model=extractor_model,
            base_url=checker_base_api,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
        )
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)


@app.command()
def atomize(
    input_file: Path = typer.Argument(..., help="Path to JSON file with extracted triplets."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_stem}_atomize.json."),
    model: str = typer.Option(None, "--atomizer-model", "--model", "-m", help="Model name for the atomizer LLM."),
    source_kg_key: str = typer.Option(..., "--source-kg-key", "-s", help="Key containing triplets to atomize (e.g. 'gemini-3.1_response_kg')."),
    atomizer_base_api: str = typer.Option(None, "--atomizer-base-api", help="Optional base URL for the atomizer LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact duplicate claims from the atomized output (e.g. a split reproducing an existing fact). On by default (loss-free)."),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Also write a per-triplet decision trace to {output}_decisions.json."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Atomize compound triplets into atomic facts."""
    from contextchecker.services.atomization import AtomizationService

    # Activate logging
    settings.enable_logging(debug=debug)

    _print_header("atomize")

    output_file = _resolve_output(input_file, "atomize", output_file)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call service
    try:
        service = AtomizationService(
            concurrency=concurrency,
            model=model,
            source_kg_key=source_kg_key,
            base_url=atomizer_base_api,
            dedup=dedup,
        )
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)

    # Write the decision trace artifact (one record per item: full input
    # triplets, every decision + reasoning + children, duplicates_removed).
    if trace:
        trace_file = output_file.with_name(output_file.stem + "_decisions.json")
        trace_file.write_text(
            json.dumps(service.last_trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Written: %s", trace_file)


@app.command()
def refcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'reference')."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_stem}_refcheck.json."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for extraction; also the {model}_response_kg key prefix."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for checking."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run RefChecker: extraction + checking in one pass, one output document."""
    from contextchecker.pipelines.refchecker import RefCheckerPipeline

    # Activate logging (pretty by default, debug if --debug)
    settings.enable_logging(debug=debug)

    _print_header("refcheck")

    output_file = _resolve_output(input_file, "refcheck", output_file)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call pipeline (CLI owns I/O; pipeline composes the services)
    try:
        pipeline = RefCheckerPipeline(
            concurrency=concurrency,
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
        )
        pipeline.run_sync(data)
        report = pipeline.last_report
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    report = {"_args": _capture_args("refcheck"), **report}
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)


@app.command()
def ragcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'gt_answer' + 'retrieved_context')."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Report path. Defaults to results/{input_stem}_ragcheck[_{runs}].json."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for both extractions (response + gt_answer)."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for all four checking directions."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    runs: int = typer.Option(1, "--runs", help="Repeat the whole run N times and report variance (N x LLM cost)."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run RagChecker: 2 extractions + 4 checking directions, one report file."""
    from contextchecker.pipelines.ragchecker import RagCheckerPipeline

    settings.enable_logging(debug=debug)

    _print_header("ragcheck")

    output_file = _resolve_output(input_file, "ragcheck", output_file, runs=runs)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call pipeline
    try:
        pipeline = RagCheckerPipeline(
            concurrency=concurrency,
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            runs=runs,
        )
        pipeline.run_sync(data)
        report = pipeline.last_report
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    report = {"_args": _capture_args("ragcheck"), **report}
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)


@app.command()
def faithcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'retrieved_context'; no ground truth)."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Report path. Defaults to results/{input_stem}_faithcheck[_{runs}].json."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for response claim extraction."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for the retrieved2response checks."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    runs: int = typer.Option(1, "--runs", help="Repeat the whole run N times and report variance (N x LLM cost)."),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run faithfulness checking without ground truth: response claims vs retrieved context."""
    from contextchecker.pipelines.faithfulness import FaithfulnessPipeline

    settings.enable_logging(debug=debug)

    _print_header("faithcheck")

    output_file = _resolve_output(input_file, "faithcheck", output_file, runs=runs)

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Call pipeline
    try:
        pipeline = FaithfulnessPipeline(
            concurrency=concurrency,
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            runs=runs,
        )
        pipeline.run_sync(data)
        report = pipeline.last_report
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    report = {"_args": _capture_args("faithcheck"), **report}
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written: %s", output_file)


# ── Eval subcommand group ────────────────────────────────────────────────────

eval_app = typer.Typer(
    name="eval",
    help="Evaluate pipeline components against ground truth.",
    no_args_is_help=True,
)
app.add_typer(eval_app)


@eval_app.command("checker")
def eval_checker(
    input_file: Path = typer.Argument(
        ..., help="Path to eval JSON with GT triplets + human_label."
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output path. Defaults to results/{input_stem}_checker_eval[_{runs}].json. A *_disagreements.json sibling is written alongside."
    ),
    checker_model: str = typer.Option(
        ..., "--checker-model", "-m", help="Model name for the checker LLM."
    ),
    checker_base_api: str = typer.Option(
        None, "--checker-base-api", help="Optional base URL for checker API."
    ),
    gt_key: str = typer.Option(
        "claude2_response_kg", "--gt-key", help="Key containing GT triplets with human_label."
    ),
    joint: bool = typer.Option(
        True, "--joint/--no-joint", help="Joint checking mode. Default: on."
    ),
    joint_num: int = typer.Option(
        settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint call."
    ),
    max_words: int = typer.Option(
        None, "--max-words", help="Word budget per call."
    ),
    runs: int = typer.Option(
        1, "--runs", help="Repeat the whole eval N times and report variance (N x LLM cost)."
    ),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output."
    ),
):
    """Evaluate checker accuracy against human-labeled GT triplets."""
    from contextchecker.eval.checkereval import CheckerEvaluator

    settings.enable_logging(debug=debug)
    _print_header("eval checker")

    output_file = _resolve_output(input_file, "checker_eval", output_file, runs=runs)
    disagree_file = output_file.with_name(
        output_file.stem + "_disagreements.json"
    )

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Evaluate
    try:
        evaluator = CheckerEvaluator(
            concurrency=concurrency,
            checker_model=checker_model,
            gt_key=gt_key,
            checker_base_url=checker_base_api,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            runs=runs,
        )
        summary_doc, disagreements_doc = evaluator.run_sync(data)
        _args = _capture_args("eval checker")
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write both documents verbatim — the evaluator assembled them.
    output_file.write_text(
        json.dumps({"_args": _args, **summary_doc}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    disagree_file.write_text(
        json.dumps({"_args": _args, **disagreements_doc}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("")
    logger.info("Written: %s", output_file)
    logger.info("Written: %s", disagree_file)



@eval_app.command("extractor")
def eval_extractor(
    input_file: Path = typer.Argument(
        ..., help="Path to eval JSON with GT triplets + response text."
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output path. Defaults to results/{input_stem}_extractor_eval[_{runs}].json."
    ),
    extractor_model: str = typer.Option(
        ..., "--extractor-model", "-e",
        help="Model name for extraction.",
    ),
    extractor_base_api: str = typer.Option(
        None, "--extractor-base-api",
        help="Optional base URL for the extractor LLM API.",
    ),
    gt_key: str = typer.Option(
        "claude2_response_kg", "--gt-key",
        help="Key containing GT triplets.",
    ),
    # LLM matching config
    checker_model: str = typer.Option(
        ..., "--checker-model",
        help="LLM model for 2-pass matching.",
    ),
    checker_base_api: str = typer.Option(
        None, "--checker-base-api",
        help="Optional base URL for the checker/matching LLM API.",
    ),
    joint_num: int = typer.Option(
        settings.DEFAULT_JOINT_NUM, "--joint-num",
        help="Max claims per joint LLM call.",
    ),
    max_words: int = typer.Option(
        None, "--max-words", help="Word budget per call.",
    ),
    atomizer_model: str = typer.Option(
        None, "--atomizer-model",
        help="Optional: model for the atomicity axis. Needs ATOMIZER_API_KEY; skipped if unset.",
    ),
    atomizer_base_api: str = typer.Option(
        None, "--atomizer-base-api",
        help="Optional base URL for the atomizer LLM API.",
    ),
    runs: int = typer.Option(
        1, "--runs", help="Repeat the whole eval N times and report variance (N x LLM cost)."
    ),
    concurrency: int = typer.Option(10, "--concurrency", help="Max simultaneous LLM requests, per LLM client. Default: 10."),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output.",
    ),
):
    """Evaluate extractor quality: extract live, then match against GT using LLM 2-pass."""
    from contextchecker.eval.extractoreval import ExtractorEvaluator

    settings.enable_logging(debug=debug)
    _print_header("eval extractor")

    output_file = _resolve_output(input_file, "extractor_eval", output_file, runs=runs)
    disagree_file = output_file.with_name(
        output_file.stem + "_disagreements.json"
    )

    # Load input
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", input_file, exc)
        raise typer.Exit(code=1)

    # Evaluate
    try:
        evaluator = ExtractorEvaluator(
            concurrency=concurrency,
            extractor_model=extractor_model,
            gt_key=gt_key,
            extractor_base_url=extractor_base_api,
            checker_model=checker_model,
            checker_base_url=checker_base_api,
            joint_num=joint_num,
            max_words=max_words,
            atomizer_model=atomizer_model,
            atomizer_base_url=atomizer_base_api,
            runs=runs,
        )
        summary_doc, disagreements_doc = evaluator.run_sync(data)
        _args = _capture_args("eval extractor")
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write both documents verbatim — the evaluator assembled them.
    output_file.write_text(
        json.dumps({"_args": _args, **summary_doc}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    disagree_file.write_text(
        json.dumps({"_args": _args, **disagreements_doc}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    logger.info("")
    logger.info("Written: %s", output_file)
    logger.info("Written: %s", disagree_file)


if __name__ == "__main__":
    app()
