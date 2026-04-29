"""
Package-wide configuration defaults.

Central place for constants, environment variable lookups, and default
values. Every module reads config from here instead of hardcoding magic
strings.

This file is a leaf dependency — it imports nothing from contextchecker.
"""

import os
import json
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("CONTEXTCHECKER_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Return a namespaced logger for *name*.

    Why a factory instead of module-level getLogger calls?
    Centralises format + level so every module gets the same setup without
    duplicating boilerplate.
    """
    logger = logging.getLogger(f"contextchecker.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


# ── LLM defaults ────────────────────────────────────────────────────────────

EXTRACTOR_API_KEY = os.getenv("EXTRACTOR_API_KEY")
if not EXTRACTOR_API_KEY:
    print("CRITICAL: EXTRACTOR_API_KEY is missing from .env file.")
    sys.exit("CRITICAL: EXTRACTOR_API_KEY is missing from .env file.")


CHECKER_API_KEY = os.getenv("CHECKER_API_KEY")
if not CHECKER_API_KEY:
    print("CRITICAL: CHECKER_API_KEY is missing from .env file.")
    sys.exit("CRITICAL: CHECKER_API_KEY is missing from .env file.")

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))


def _load_prompts():
    """
    Internal function to load prompts once.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, 'prompt_map.json')
    
    try:
        with open(prompt_path, 'r') as file:
            print(f"   Successfully loaded prompts from {prompt_path}")
            return json.load(file)
    except FileNotFoundError:
        print(f"CRITICAL: Could not find prompt_map.json file at {prompt_path}.")
        sys.exit(f"CRITICAL: Could not find prompt_map.json file at {prompt_path}.") 
    except json.JSONDecodeError:
        print(f"CRITICAL: prompt_map.json is not valid JSON.")
        sys.exit(f"CRITICAL: prompt_map.json is not valid JSON.") 
    except Exception as e:
        print(f"An unexpected error occurred loading prompts: {e}")
        sys.exit(f"An unexpected error occurred loading prompts: {e}") 

# 2. Execution (Runs ONCE on first import)
PROMPTS = _load_prompts()
