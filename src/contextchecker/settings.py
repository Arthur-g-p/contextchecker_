"""
Package-wide configuration defaults.

Reads environment variables and prompt templates at import time.
Services validate that the config they need is present before running.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("CONTEXTCHECKER_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

_logger = logging.getLogger("contextchecker.settings")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for *name*."""
    logger = logging.getLogger(f"contextchecker.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


# ── API keys ─────────────────────────────────────────────────────────────────
# Read eagerly, validated lazily by the service that needs them.

EXTRACTOR_API_KEY: str | None = os.getenv("EXTRACTOR_API_KEY")
if not EXTRACTOR_API_KEY:
    _logger.warning("EXTRACTOR_API_KEY not set — extraction commands will fail.")

CHECKER_API_KEY: str | None = os.getenv("CHECKER_API_KEY")
if not CHECKER_API_KEY:
    _logger.warning("CHECKER_API_KEY not set — checking commands will fail.")

LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "120.0"))


# ── Prompts ──────────────────────────────────────────────────────────────────

def _load_prompts() -> dict[str, str]:
    """Load prompt templates from prompt_map.json shipped with the package."""
    prompt_path = Path(__file__).parent / "prompt_map.json"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"prompt_map.json not found at {prompt_path}. Package may be corrupted."
        )

    text = prompt_path.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt_map.json is not valid JSON: {exc}") from exc


PROMPTS: dict[str, str] = _load_prompts()

