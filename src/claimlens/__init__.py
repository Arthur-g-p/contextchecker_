"""
claimlens — Evaluate LLM-generated knowledge graphs against ground truth.

Public API:
    from claimlens import enable_logging
    from claimlens import check_faithfulness   # real-time, no GT
    from claimlens.services.extraction import ExtractionService
"""

from claimlens.settings import enable_logging


def check_faithfulness(*args, **kwargs):
    """Real-time faithfulness for a single response (no ground truth).

    Lazy import: the facade must not drag the pipeline stack into
    ``import claimlens`` for users who only want the version string.
    See pipelines/faithfulness.py for the full signature.
    """
    from claimlens.pipelines.faithfulness import check_faithfulness as _impl
    return _impl(*args, **kwargs)


__all__ = ["enable_logging", "check_faithfulness"]
