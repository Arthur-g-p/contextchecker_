"""Unit tests for _resolve_output — the CLI's output-path naming rule."""

from pathlib import Path
from unittest.mock import patch

import pytest

from contextchecker.cli import _resolve_output


# ── Naming per operation ─────────────────────────────────────────────────────

OPERATIONS = [
    "extract",
    "check",
    "atomize",
    "refcheck",
    "ragcheck",
    "faithcheck",
    "checker_eval",
    "extractor_eval",
]


class TestNaming:

    @pytest.mark.parametrize("operation", OPERATIONS)
    def test_stem_and_operation(self, tmp_path, operation):
        src = tmp_path / "science_facts.json"
        out = _resolve_output(src, operation)
        assert out.name == f"science_facts_{operation}.json"

    def test_lands_in_results_next_to_input(self, tmp_path):
        src = tmp_path / "nested" / "science_facts.json"
        src.parent.mkdir()
        out = _resolve_output(src, "faithcheck")
        assert out.parent == tmp_path / "nested" / "results"

    def test_input_extension_is_not_carried_into_stem(self, tmp_path):
        """Output must not reuse the input's full filename."""
        src = tmp_path / "data.json"
        out = _resolve_output(src, "extract")
        assert out.name == "data_extract.json"


class TestRunsSuffix:

    def test_single_run_has_no_suffix(self, tmp_path):
        out = _resolve_output(tmp_path / "d.json", "ragcheck", runs=1)
        assert out.name == "d_ragcheck.json"

    def test_multi_run_appends_count(self, tmp_path):
        out = _resolve_output(tmp_path / "d.json", "ragcheck", runs=5)
        assert out.name == "d_ragcheck_5.json"

    def test_default_runs_is_one(self, tmp_path):
        """Commands without --runs never pass it, so they must not get _N."""
        out = _resolve_output(tmp_path / "d.json", "extract")
        assert out.name == "d_extract.json"


class TestExplicitPath:

    def test_explicit_bypasses_naming(self, tmp_path):
        explicit = tmp_path / "mine.json"
        out = _resolve_output(tmp_path / "d.json", "ragcheck", explicit, runs=5)
        assert out == explicit

    def test_explicit_parent_is_created(self, tmp_path):
        explicit = tmp_path / "deep" / "nested" / "mine.json"
        _resolve_output(tmp_path / "d.json", "extract", explicit)
        assert explicit.parent.is_dir()


class TestDirectoryCreation:

    def test_results_dir_created_when_missing(self, tmp_path):
        out = _resolve_output(tmp_path / "d.json", "extract")
        assert out.parent.is_dir()

    def test_existing_results_dir_is_reused(self, tmp_path):
        (tmp_path / "results").mkdir()
        out = _resolve_output(tmp_path / "d.json", "extract")
        assert out.parent == tmp_path / "results"


class TestOverwriteWarning:

    def test_warns_when_target_exists(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "d_extract.json").write_text("{}")
        with patch("contextchecker.cli.logger") as mock_log:
            _resolve_output(tmp_path / "d.json", "extract")
        mock_log.warning.assert_called_once()

    def test_silent_when_target_is_new(self, tmp_path):
        with patch("contextchecker.cli.logger") as mock_log:
            _resolve_output(tmp_path / "d.json", "extract")
        mock_log.warning.assert_not_called()

    def test_explicit_path_also_warns(self, tmp_path):
        """-o is not exempt: an explicit target gets clobbered just the same."""
        explicit = tmp_path / "mine.json"
        explicit.write_text("{}")
        with patch("contextchecker.cli.logger") as mock_log:
            _resolve_output(tmp_path / "d.json", "extract", explicit)
        mock_log.warning.assert_called_once()


class TestDisagreementsSibling:
    """eval extractor derives a second file from the resolved stem."""

    def test_sibling_follows_the_new_name(self, tmp_path):
        out = _resolve_output(tmp_path / "msmarco_5.json", "extractor_eval", runs=3)
        sibling = out.with_name(out.stem + "_findings.json")
        assert sibling.name == "msmarco_5_extractor_eval_3_findings.json"
        assert sibling.parent == out.parent
