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
    help="Evaluate LLM-generated knowledge graphs against ground truth.",
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


@app.command()
def extract(
    input_file: Path = typer.Argument(..., help="Path to JSON input file."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to results/{input_filename}."),
    model: str = typer.Option(None, "--model", "-m", help="Model name used as key prefix."),
    extractor_base_api: str = typer.Option(None, "--extractor-base-api", help="Optional base URL for the LLM API."),
    max_retries: int = typer.Option(2, "--max-retries", help="Max retry rounds for failed parse errors. Default: 2."),
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
        service = ExtractionService(model=model, base_url=extractor_base_api, max_retries=max_retries)
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
        )
        result = service.run_sync(data)
    except ContextCheckerError as exc:
        logger.error("")
        logger.error("❌ %s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)


if __name__ == "__main__":
    app()
