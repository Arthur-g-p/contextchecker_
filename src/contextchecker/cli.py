"""
CLI controllers — the I/O boundary of the package.

Responsibilities (and nothing more):
- Parse Typer flags and arguments
- Resolve file paths
- Call the appropriate service
- Write output / report results

All business logic lives in services. All execution lives in workers.
"""

import typer

app = typer.Typer(
    name="contextchecker",
    help="Evaluate LLM-generated knowledge graphs against ground truth.",
    no_args_is_help=True,
)
