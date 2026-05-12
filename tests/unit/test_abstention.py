"""
Unit tests for pure helper functions in the extraction pipeline.

These test _is_full_abstention — a pure function with no dependencies.
"""

import pytest

from contextchecker.services.extraction import _is_full_abstention


class TestIsFullAbstention:
    """Verify abstention detection across edge cases."""

    # ── Definite abstentions ─────────────────────────────────────

    def test_empty_string(self):
        assert _is_full_abstention("") is True

    def test_whitespace_only(self):
        assert _is_full_abstention("   \n\t  ") is True

    def test_none_input(self):
        assert _is_full_abstention(None) is True

    def test_exact_refusal_phrase(self):
        assert _is_full_abstention("I don't know") is True

    def test_refusal_with_punctuation(self):
        assert _is_full_abstention("I don't know.") is True

    def test_refusal_case_insensitive(self):
        assert _is_full_abstention("I DON'T KNOW") is True

    def test_refusal_phrase_information_not_provided(self):
        assert _is_full_abstention("Information not provided.") is True

    # ── NOT abstentions (real content) ───────────────────────────

    def test_real_content_short(self):
        assert _is_full_abstention("The capital of France is Paris.") is False

    def test_real_content_with_refusal_substring(self):
        """A long response that contains a refusal phrase should NOT be flagged."""
        text = (
            "The answer is complex. I don't know the exact figure, but "
            "the population of Tokyo is estimated to be around 14 million "
            "in the city proper."
        )
        assert _is_full_abstention(text) is False

    def test_refusal_below_threshold(self):
        """Refusal phrase present but not dominant (below 85% coverage)."""
        text = "I don't know, but here is some additional context about the topic."
        assert _is_full_abstention(text) is False

    # ── Threshold edge cases ─────────────────────────────────────

    def test_custom_threshold_stricter(self):
        """With threshold=1.0, padded text should NOT trigger."""
        # "ok i dont know" → 14 chars, phrase is 11 → 11/14 = 0.78 < 1.0
        assert _is_full_abstention("ok, i dont know", threshold=1.0) is False

    def test_custom_threshold_looser(self):
        """With threshold=0.5, modest padding still triggers."""
        # "i i dont know" → 13 chars, phrase is 11 → 11/13 = 0.84 >= 0.5
        text = "I, I don't know"
        assert _is_full_abstention(text, threshold=0.5) is True
