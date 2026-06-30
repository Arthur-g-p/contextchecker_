"""
Package-wide configuration defaults.

Reads environment variables and prompt templates at import time.
Services validate that the config they need is present before running.

Logging architecture:
    - Every module calls get_logger(__name__) → silent by default (NullHandler).
    - CLI calls enable_logging() at startup → pretty output via PrettyFormatter.
    - Library users call contextchecker.enable_logging() to opt in.
    - --debug flag switches to DebugFormatter (timestamp + module prefix).
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Logging ──────────────────────────────────────────────────────────────────

class PrettyFormatter(logging.Formatter):
    """Default CLI formatter — just the message, no noise."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class DebugFormatter(logging.Formatter):
    """Debug formatter — timestamp + source module prefix on every line."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        src = record.name.removeprefix("contextchecker.")
        # Pad source to 12 chars for alignment
        return f"{ts} | {src:<12} | {record.getMessage()}"


# Install a NullHandler on the root logger so library users get silence.
# Without this, Python's lastResort handler prints WARNING+ to stderr.
logging.getLogger("contextchecker").addHandler(logging.NullHandler())


def enable_logging(debug: bool = False) -> None:
    """Activate console output for the contextchecker package.

    Called automatically by the CLI. Library users can call this
    explicitly to see output:

        import contextchecker
        contextchecker.enable_logging()        # pretty output
        contextchecker.enable_logging(debug=True)  # + timestamps & module & sometimes more info and certain events
    """
    root = logging.getLogger("contextchecker")

    # Remove any existing handlers (avoid duplicates on repeated calls)
    for h in root.handlers[:]:
        if not isinstance(h, logging.NullHandler):
            root.removeHandler(h)

    formatter = DebugFormatter() if debug else PrettyFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for *name*.

    The logger inherits from the 'contextchecker' root logger.
    No handlers are attached here — enable_logging() controls output.
    """
    return logging.getLogger(f"contextchecker.{name}")


# ── API keys ─────────────────────────────────────────────────────────────────
# Read eagerly, validated lazily by the service that needs them.

EXTRACTOR_API_KEY: str | None = os.getenv("EXTRACTOR_API_KEY")
CHECKER_API_KEY: str | None = os.getenv("CHECKER_API_KEY")
ATOMIZER_API_KEY: str | None = os.getenv("ATOMIZER_API_KEY")

LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "120.0"))


# ── Checker defaults ────────────────────────────────────────────────────────
# Joint mode groups multiple claims per LLM call. These control chunking.

DEFAULT_JOINT_NUM: int = 10
"""Max claims per joint LLM call."""

DEFAULT_MAX_WORDS: int = 6000
"""Word budget for joint prompts (~8k tokens ≈ 6000 words).
Only applies in joint mode by default. Single mode has no budget
unless the user explicitly sets --max-words."""


# ── Console formatting ───────────────────────────────────────────────────────

SECTION_WIDTH: int = 60
"""Column width for console section banners (── LABEL ───…)."""


def section_rule(label: str, width: int = SECTION_WIDTH) -> str:
    """Render a fixed-width section banner: ``── LABEL ───…`` padded to `width`.

    Centralizes the divider so every phase/result banner lines up regardless of
    label length.
    """
    prefix = f"── {label} "
    if len(prefix) >= width:
        return prefix
    return prefix + "─" * (width - len(prefix))


# ── Prompts ──────────────────────────────────────────────────────────────────

PROMPT_PATH: Path = Path(__file__).parent / "prompt_map.json"


def _load_prompts() -> dict[str, str]:
    """Load prompt templates from prompt_map.json shipped with the package."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"prompt_map.json not found at {PROMPT_PATH}. Package may be corrupted."
        )

    text = PROMPT_PATH.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt_map.json is not valid JSON: {exc}") from exc


PROMPTS: dict[str, str] = _load_prompts()
