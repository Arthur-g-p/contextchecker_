"""
contextchecker — Evaluate LLM-generated knowledge graphs against ground truth.

Public API:
    from contextchecker import enable_logging
    from contextchecker import check_faithfulness   # real-time, no GT
    from contextchecker.services.extraction import ExtractionService
"""

from contextchecker.settings import enable_logging


def check_faithfulness(*args, **kwargs):
    """Real-time faithfulness for a single response (no ground truth).

    Lazy import: the facade must not drag the pipeline stack into
    ``import contextchecker`` for users who only want the version string.
    See pipelines/faithfulness.py for the full signature.
    """
    from contextchecker.pipelines.faithfulness import check_faithfulness as _impl
    return _impl(*args, **kwargs)


__all__ = ["enable_logging", "check_faithfulness"]
