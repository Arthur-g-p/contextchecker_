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
    │   ├── ContextTooLongError  ← input exceeded model context window
    │   ├── ContentPolicyError   ← safety filter rejected content
    │   ├── FinishReasonLengthError ← answer cut off by an output-token limit
    │   └── LLMTimeoutError      ← request exceeded its time budget
    └── ParsingError         ← LLM returned unparseable output

Compat aliases (used by llmclient.py, will migrate later):
    LLMError      → LLMClientError
    LLMParseError → ParsingError
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


class ContextTooLongError(LLMClientError):
    """Raised when input exceeds the model's context window."""


class ContentPolicyError(LLMClientError):
    """Raised when the model's safety filter rejects the content."""


class FinishReasonLengthError(LLMClientError):
    """Raised when the answer was cut off by an output-token limit."""


class LLMTimeoutError(LLMClientError):
    """Raised when a request exceeded its time budget, after one retry."""


class ParsingError(WorkerError):
    """Raised when LLM output cannot be parsed into the expected structure."""


# ── Compat aliases (used by llmclient.py — migrate later) ────────────────────

LLMError = LLMClientError
LLMParseError = ParsingError
