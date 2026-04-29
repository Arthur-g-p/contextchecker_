"""
CLI controllers — the I/O boundary of the package.

Responsibilities (and nothing more):
- Parse Typer flags and arguments
- Resolve file paths
- Call the appropriate service
- Write output / report results

All business logic lives in services. All execution lives in workers.
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


@app.command()
def extract(
    input_file: Path = typer.Argument(..., help="Path to JSON input file."),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path. Defaults to input_file with '_extracted' suffix."),
    model: str = typer.Option("gemini-2.0-flash", "--model", "-m", help="Model name used as key prefix."),
):
    """Run the extraction pipeline on a JSON dataset."""
    from contextchecker.services.extraction import run_extract_service

    # Resolve output path
    if output_file is None:
        output_file = input_file.with_stem(input_file.stem + "_extracted")

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
        result = run_extract_service(data, model)
    except ContextCheckerError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        raise typer.Exit(code=1)

    # Write output
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_file)

