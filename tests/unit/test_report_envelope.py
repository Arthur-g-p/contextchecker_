"""Unit tests for the report envelope: the _args / _meta contract.

Every report type must expose the same _meta core in the same order, and
_args must be ordered and redacted the same way regardless of command. The
conformance test is the point of this module — four report builders drifted
apart once already.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from claimlens.cli import _ARGS_ORDER, _capture_args
from claimlens.utils import REPORT_SCHEMA_VERSION, build_meta


META_CORE = (
    "schema_version",
    "report_type",
    "claimlensversion",
    "timestamp",
    "duration_seconds",
    "total_items",
    "evaluated_items",
    "dropped_items",
)


def _meta(**over):
    kwargs = dict(
        timestamp="2026-08-09T16:03:12",
        duration_seconds=36.44,
        total_items=8,
        evaluated_items=8,
        dropped_items=0,
    )
    kwargs.update(over)
    return build_meta(kwargs.pop("report_type", "ragcheck"), **kwargs)


class TestBuildMeta:

    def test_core_keys_present_in_order(self):
        assert tuple(_meta())[:len(META_CORE)] == META_CORE

    def test_schema_version_is_current(self):
        assert _meta()["schema_version"] == REPORT_SCHEMA_VERSION

    def test_duration_rounded(self):
        assert _meta(duration_seconds=36.4444)["duration_seconds"] == 36.4

    def test_extras_appended_after_the_core(self):
        m = _meta(report_type="extractor_eval", pred_key="m_response_kg",
                  matching="llm-2-pass")
        assert tuple(m)[:len(META_CORE)] == META_CORE
        assert tuple(m)[len(META_CORE):] == ("pred_key", "matching")

    def test_no_arguments_leak_into_meta(self):
        """Models are given, not discovered — they belong in _args."""
        assert "extractor_model" not in _meta()
        assert "checker_model" not in _meta()
        assert "gt_key" not in _meta()

    def test_version_is_reported(self):
        assert _meta()["claimlensversion"]


class TestMetaConformance:
    """All report types must agree on the core, whatever extras they add."""

    @pytest.mark.parametrize("report_type,extras", [
        ("ragcheck", {}),
        ("faithcheck", {}),
        ("refcheck", {}),
        ("checker_eval", {}),
        ("extractor_eval", {"pred_key": "m_response_kg", "matching": "llm-2-pass"}),
    ])
    def test_core_is_identical(self, report_type, extras):
        m = _meta(report_type=report_type, **extras)
        assert tuple(m)[:len(META_CORE)] == META_CORE
        assert m["report_type"] == report_type

    def test_runs_extra_does_not_reshape_the_core(self):
        """--runs adds a key; a reader of one report still reads a nested one."""
        m = _meta(runs_completed=5)
        assert tuple(m)[:len(META_CORE)] == META_CORE


# ── _args ────────────────────────────────────────────────────────────────────

class _FakeCtx:
    def __init__(self, params, explicit=()):
        self.params = params
        self._explicit = set(explicit)

    def get_parameter_source(self, name):
        from click.core import ParameterSource
        return (ParameterSource.COMMANDLINE if name in self._explicit
                else ParameterSource.DEFAULT)


def _capture(params, explicit=(), command="ragcheck"):
    with patch("click.get_current_context", return_value=_FakeCtx(params, explicit)):
        return _capture_args(command)


class TestCaptureArgs:

    def test_command_is_first(self):
        assert next(iter(_capture({"joint": True}))) == "command"

    def test_known_keys_follow_canonical_order(self):
        args = _capture({"debug": True, "extractor_model": "m", "input_file": "f"})
        order = [k for k in args if k in _ARGS_ORDER]
        assert order == ["command", "input_file", "extractor_model", "debug"]

    def test_unknown_keys_sorted_after_known_ones(self):
        args = _capture({"zeta": 1, "alpha": 2, "joint": True})
        keys = [k for k in args if k != "_explicit"]
        assert keys[-2:] == ["alpha", "zeta"]

    def test_paths_serialized(self):
        args = _capture({"input_file": Path("examples/x.json")})
        assert args["input_file"] == "examples/x.json"
        assert isinstance(args["input_file"], str)

    def test_explicit_lists_only_command_line_values(self):
        args = _capture({"joint": True, "runs": 5}, explicit=("runs",))
        assert args["_explicit"] == ["runs"]

    def test_explicit_is_last(self):
        assert list(_capture({"joint": True}))[-1] == "_explicit"

    def test_defaults_are_recorded_but_not_marked_explicit(self):
        args = _capture({"joint": True})
        assert args["joint"] is True
        assert args["_explicit"] == []


class TestRedaction:
    """Params are captured wholesale, so the denylist is the only guard."""

    @pytest.mark.parametrize("secret", [
        "api_key", "extractor_api_key", "checker_api_key", "atomizer_api_key",
        "token", "password", "secret",
    ])
    def test_credentials_never_reach_the_report(self, secret):
        args = _capture({secret: "sk-live-should-never-appear", "joint": True})
        assert secret not in args
        assert "sk-live-should-never-appear" not in str(args)

    def test_secret_excluded_from_explicit_too(self):
        args = _capture({"api_key": "sk-x"}, explicit=("api_key",))
        assert args["_explicit"] == []
