"""
ContextChecker exception hierarchy.

Every module in the package imports its errors from here.
This file is a leaf dependency — it imports nothing from contextchecker.

Hierarchy
---------
ContextCheckerError          ← catch-all for any package error
├── CLIError                 ← bad flags, missing files, I/O issues
├── ServiceError             ← orchestration / pipeline failures
│   ├── InvalidInputError    ← validation rejects the data
│   └── FilterError          ← nothing left after filtering
└── WorkerError              ← execution-level failures
    ├── LLMClientError       ← network / API errors from LLM calls
    └── ParsingError         ← LLM returned unparseable output
"""


class ContextCheckerError(Exception):
    """Root exception for the entire contextchecker package."""


# ── CLI layer ────────────────────────────────────────────────────────────────

class CLIError(ContextCheckerError):
    """Raised when the CLI receives invalid arguments or encounters I/O issues."""


# ── Service layer ────────────────────────────────────────────────────────────

class ServiceError(ContextCheckerError):
    """Raised when a service-level orchestration step fails."""


class InvalidInputError(ServiceError):
    """Raised when input data fails validation (e.g. missing required keys)."""


class FilterError(ServiceError):
    """Raised when filtering leaves zero items to process."""


# ── Worker layer ─────────────────────────────────────────────────────────────

class WorkerError(ContextCheckerError):
    """Raised when a worker-level execution step fails."""


class LLMClientError(WorkerError):
    """Raised on network or API errors during LLM calls."""


class ParsingError(WorkerError):
    """Raised when LLM output cannot be parsed into the expected structure."""
