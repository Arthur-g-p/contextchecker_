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

# Todo: add concurrency to extractor and checker
@app.command()
def extract(
    input_file: Path = typer.Argument(..., help="Path to JSON input file."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_filename}."),
    model: str = typer.Option(None, "--model", "-m", help="Model name used as key prefix."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the LLM API."),
    max_retries: int = typer.Option(2, "--max-retries", help="Max retry rounds for failed parse errors. Default: 2. Max: Amount of retry stategies defined."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets from the output. On by default (loss-free cleanup)."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run the extraction pipeline on a JSON dataset."""
    from contextchecker.services.extraction import ExtractionService

    # Activate logging (pretty by default, debug if --debug)
    settings.enable_logging(debug=debug)

    _print_header("extract")

    # Resolve output path — default: results/{filename} next to input
    if output_file is None:
        output_file = input_file.parent / "results" / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
        service = ExtractionService(model=model, base_url=extractor_base_api, max_retries=max_retries, dedup=dedup)
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)


@app.command()
def check(
    input_file: Path = typer.Argument(..., help="Path to JSON file with extracted triplets."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_filename}."),
    model: str = typer.Option(None, "--model", "-m", help="Model name for the checker LLM."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model name that was used for extraction (to find the kg_key)."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Use joint checking (multiple claims per LLM call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per LLM call. Default: 6000 in joint mode, unset in single."),
    max_retries: int = typer.Option(None, "--max-retries", help="Max retry rounds for API/parsing failures."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Check extracted claims against reference passages."""
    from contextchecker.services.checking import CheckingService

    # Activate logging
    settings.enable_logging(debug=debug)

    _print_header("check")

    # Resolve output path — default: results/{filename} next to input
    if output_file is None:
        output_file = input_file.parent / "results" / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            model=model,
            extractor_model=extractor_model,
            base_url=checker_base_api,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            max_retries=max_retries,
        )
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)


@app.command()
def atomize(
    input_file: Path = typer.Argument(..., help="Path to JSON file with extracted triplets."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_filename}."),
    model: str = typer.Option(None, "--model", "-m", help="Model name for the atomizer LLM."),
    source_kg_key: str = typer.Option(..., "--source-kg-key", "-s", help="Key containing triplets to atomize (e.g. 'gemini-3.1_response_kg')."),
    atomizer_base_api: str = typer.Option(None, "--atomizer-base-api", help="Optional base URL for the atomizer LLM API."),
    max_retries: int = typer.Option(2, "--max-retries", help="Max retry rounds for failed parse errors. Default: 2."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact duplicate claims from the atomized output (e.g. a split reproducing an existing fact). On by default (loss-free)."),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Also write a per-triplet decision trace to {output}_decisions.json."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Atomize compound triplets into atomic facts."""
    from contextchecker.services.atomization import AtomizationService

    # Activate logging
    settings.enable_logging(debug=debug)

    _print_header("atomize")

    # Resolve output path — default: results/{filename} next to input
    if output_file is None:
        output_file = input_file.parent / "results" / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            model=model,
            source_kg_key=source_kg_key,
            base_url=atomizer_base_api,
            max_retries=max_retries,
            dedup=dedup,
        )
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)

    # Write the decision trace artifact (one record per item: full input
    # triplets, every decision + reasoning + children, duplicates_removed).
    if trace:
        trace_file = output_file.with_name(output_file.stem + "_decisions.json")
        trace_file.write_text(
            json.dumps(service.last_trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Decision trace written to %s", trace_file)


@app.command()
def refcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'reference')."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_filename}."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for extraction; also the {model}_response_kg key prefix."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for checking."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    extractor_max_retries: int = typer.Option(2, "--extractor-max-retries", help="Max retry rounds for extraction parse errors. Default: 2."),
    checker_max_retries: int = typer.Option(None, "--checker-max-retries", help="Max retry rounds for checking failures."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run RefChecker: extraction + checking in one pass, one output document."""
    from contextchecker.pipelines.refchecker import RefCheckerPipeline

    # Activate logging (pretty by default, debug if --debug)
    settings.enable_logging(debug=debug)

    _print_header("refcheck")

    # Resolve output path - default: results/{filename} next to input
    if output_file is None:
        output_file = input_file.parent / "results" / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            extractor_max_retries=extractor_max_retries,
            checker_max_retries=checker_max_retries,
        )
        result = pipeline.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)


@app.command()
def ragcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'gt_answer' + 'retrieved_context')."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Report path. Defaults to results/{input_stem}_ragcheck.json."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for both extractions (response + gt_answer)."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for all four checking directions."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    extractor_max_retries: int = typer.Option(2, "--extractor-max-retries", help="Max retry rounds for extraction parse errors. Default: 2."),
    checker_max_retries: int = typer.Option(None, "--checker-max-retries", help="Max retry rounds for checking failures."),
    runs: int = typer.Option(1, "--runs", help="Repeat the whole run N times and report variance (N x LLM cost)."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run RagChecker: 2 extractions + 4 checking directions, one report file."""
    from contextchecker.pipelines.ragchecker import RagCheckerPipeline

    settings.enable_logging(debug=debug)

    _print_header("ragcheck")

    # Resolve output path - the report IS the output artifact
    if output_file is None:
        output_file = input_file.parent / "results" / f"{input_file.stem}_ragcheck.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            extractor_max_retries=extractor_max_retries,
            checker_max_retries=checker_max_retries,
            runs=runs,
        )
        pipeline.run_sync(data)
        report = pipeline.last_report
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write the single report artifact
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", output_file)


@app.command()
def faithcheck(
    input_file: Path = typer.Argument(..., help="Path to JSON input file (needs 'response' + 'retrieved_context'; no ground truth)."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Report path. Defaults to results/{input_stem}_faithcheck.json."),
    extractor_model: str = typer.Option(..., "--extractor-model", "-e", help="Model for response claim extraction."),
    checker_model: str = typer.Option(..., "--checker-model", "-c", help="Model for the retrieved2response checks."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the extractor LLM API."),
    checker_base_api: str = typer.Option(None, "--checker-base-api", help="Optional base URL for the checker LLM API."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Remove exact (s,p,o) duplicate triplets. On by default."),
    joint: bool = typer.Option(True, "--joint/--no-joint", help="Joint checking (multiple claims per call). Default: on."),
    joint_num: int = typer.Option(settings.DEFAULT_JOINT_NUM, "--joint-num", help="Max claims per joint LLM call."),
    max_words: int = typer.Option(None, "--max-words", help="Word budget per checker call. Default: 6000 in joint mode."),
    extractor_max_retries: int = typer.Option(2, "--extractor-max-retries", help="Max retry rounds for extraction parse errors. Default: 2."),
    checker_max_retries: int = typer.Option(None, "--checker-max-retries", help="Max retry rounds for checking failures."),
    runs: int = typer.Option(1, "--runs", help="Repeat the whole run N times and report variance (N x LLM cost)."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output with timestamps and module names."),
):
    """Run faithfulness checking without ground truth: response claims vs retrieved context."""
    from contextchecker.pipelines.faithfulness import FaithfulnessPipeline

    settings.enable_logging(debug=debug)

    _print_header("faithcheck")

    # Resolve output path - the report IS the output artifact
    if output_file is None:
        output_file = input_file.parent / "results" / f"{input_file.stem}_faithcheck.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            extractor_model=extractor_model,
            checker_model=checker_model,
            extractor_base_url=extractor_base_api,
            checker_base_url=checker_base_api,
            dedup=dedup,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            extractor_max_retries=extractor_max_retries,
            checker_max_retries=checker_max_retries,
            runs=runs,
        )
        pipeline.run_sync(data)
        report = pipeline.last_report
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write the single report artifact
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", output_file)


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
        None, "--output", "-o", help="Output JSON path for eval results."
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
    max_retries: int = typer.Option(
        None, "--max-retries", help="Max retry rounds for API/parsing failures."
    ),
    runs: int = typer.Option(
        1, "--runs", help="Repeat the whole eval N times and report variance (N x LLM cost)."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output."
    ),
):
    """Evaluate checker accuracy against human-labeled GT triplets."""
    from contextchecker.eval.checkereval import CheckerEvaluator

    settings.enable_logging(debug=debug)
    _print_header("eval checker")

    # Resolve output path
    if output_file is None:
        output_file = (
            input_file.parent / "results" / f"checker_eval_{input_file.stem}.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

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
            checker_model=checker_model,
            gt_key=gt_key,
            checker_base_url=checker_base_api,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            max_retries=max_retries,
            runs=runs,
        )
        result_doc = evaluator.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write the document verbatim — the evaluator assembled it.
    output_file.write_text(
        json.dumps(result_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("")
    logger.info("Results written to %s", output_file)



@eval_app.command("extractor")
def eval_extractor(
    input_file: Path = typer.Argument(
        ..., help="Path to eval JSON with GT triplets + response text."
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path for eval results."
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
    max_retries: int = typer.Option(
        None, "--max-retries",
        help="Max retry rounds for API/parsing failures.",
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
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output.",
    ),
):
    """Evaluate extractor quality: extract live, then match against GT using LLM 2-pass."""
    from contextchecker.eval.extractoreval import ExtractorEvaluator

    settings.enable_logging(debug=debug)
    _print_header("eval extractor")

    # Resolve output paths
    if output_file is None:
        output_file = (
            input_file.parent / "results" / f"extractor_eval_{input_file.stem}.json"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
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
            extractor_model=extractor_model,
            gt_key=gt_key,
            extractor_base_url=extractor_base_api,
            checker_model=checker_model,
            checker_base_url=checker_base_api,
            joint_num=joint_num,
            max_words=max_words,
            max_retries=max_retries,
            atomizer_model=atomizer_model,
            atomizer_base_url=atomizer_base_api,
            runs=runs,
        )
        summary_doc, disagreements_doc = evaluator.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write both documents verbatim — the evaluator assembled them.
    output_file.write_text(
        json.dumps(summary_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    disagree_file.write_text(
        json.dumps(disagreements_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("")
    logger.info("Results written to %s", output_file)
    logger.info("Disagreements written to %s", disagree_file)


if __name__ == "__main__":
    app()
