"""Unit tests for the holy-data aggregation and display layer.

Covers the rule-set implementations from docs/output_conventions.md:
- rule set 4: build_variance roster + sectioned variance block
- rule set 1: log_mece_tree (sum enforcement, footer law)
- rule set 2: log_rate_rows (hidden ≠ zero, key lineage)
- checker eval's derived rates (support 0 → None, never a fake 0.0)

Ordered by relevance: JSON-affecting aggregation first, display last.
"""

import logging

import pytest

from contextchecker.utils import build_variance
from contextchecker.stats import (
    log_mece_tree,
    log_rate_rows,
    log_run_line,
    log_variance_block,
    roster_from_sections,
)
from contextchecker.pipelines.directions import abstention_counts

# NOTE: settings.get_logger(__name__) double-prefixes module
# loggers ("contextchecker.contextchecker.stats"), so tests
# capture at the root level rather than by logger name.


# ── Rule set 4: the variance surface (feeds JSON, not just display) ──────────

class TestBuildVarianceRoster:
    def test_roster_filters_counters_and_orders(self):
        runs = [{"b": 1.0, "a": 0.5, "total_items": 7},
                {"b": 0.0, "a": 0.5, "total_items": 7}]
        means, variance = build_variance(runs, roster=["a", "b"])
        assert list(means) == ["a", "b"]          # roster order, not dict order
        assert "total_items" not in means         # counters never varianced
        assert variance["b"]["min"] == 0.0

    def test_legacy_sweep_unchanged_without_roster(self):
        means, _ = build_variance([{"a": 1.0, "total_items": 7}])
        assert "total_items" in means             # pre-rule-set-4 behavior kept

    def test_absent_roster_key_surfaces_as_null(self):
        """The roster is a promised metric surface — a key missing from
        every run appears as mean None rather than vanishing."""
        means, variance = build_variance([{"a": 1.0}], roster=["a", "ghost"])
        assert means["ghost"] is None
        assert variance["ghost"]["n"] == 0

    def test_null_runs_lower_support_never_zeroed(self):
        means, variance = build_variance([{"a": 0.4}, {"a": None}],
                                         roster=["a"])
        assert means["a"] == 0.4
        assert variance["a"]["n"] == 1


class TestRosterFromSections:
    def test_flattens_in_display_order(self):
        sections = {"metrics": [("G1", ["m1", "m2"]), (None, ["m3"])],
                    "behavior": ["b1"], "health": ["h1"]}
        assert roster_from_sections(sections) == ["m1", "m2", "m3", "b1", "h1"]

    def test_missing_sections_tolerated(self):
        assert roster_from_sections({"metrics": [("G", ["x"])]}) == ["x"]


# ── Checker eval derived rates (top-level JSON keys) ─────────────────────────

class TestCheckerEvalDerivedRates:
    """A data gap must never masquerade as a score."""

    def _result(self, gt, pred, parse_errors=0):
        from contextchecker.eval.checkereval import CheckerEvaluator
        dummy = object.__new__(CheckerEvaluator)   # ctor needs an API key
        return CheckerEvaluator._build_result(
            dummy, gt, pred, parse_errors, 1,
            {"missing_gt": 0, "missing_context": 0, "empty_gt": 0})

    def test_zero_support_label_is_none_not_zero(self):
        r = self._result(["Entailment"] * 4, ["Entailment"] * 4)
        assert r.entailment_f1 == 1.0
        assert r.contradiction_f1 is None
        assert r.neutral_f1 is None

    def test_macro_f1_matches_report(self):
        gt = ["Entailment", "Entailment", "Neutral", "Neutral"]
        pred = ["Entailment", "Neutral", "Neutral", "Neutral"]
        r = self._result(gt, pred)
        assert r.macro_f1 == round(r.report["macro avg"]["f1-score"], 4)

    def test_checker_failure_rate_over_issued(self):
        r = self._result(["Entailment"] * 4, ["Entailment"] * 4,
                         parse_errors=1)
        assert r.checker_failure_rate == round(1 / 5, 4)

    def test_failure_rate_none_when_nothing_issued(self):
        r = self._result([], [], parse_errors=0)
        assert r.checker_failure_rate is None


# ── Sectioned variance block (display of rule set 4) ─────────────────────────

class TestVarianceBlockSections:
    def _render(self, caplog, runs, sections):
        means, variance = build_variance(
            runs, roster=roster_from_sections(sections))
        with caplog.at_level(logging.INFO):
            log_variance_block(len(runs), means, variance, sections=sections)
        return caplog.text

    def test_sections_mirror_per_run_blocks(self, caplog):
        sections = {"metrics": [("Overall", ["m1"])],
                    "behavior": ["b1"], "health": ["h1"]}
        text = self._render(
            caplog, [{"m1": 0.4, "b1": 0.1, "h1": 0.0},
                     {"m1": 0.6, "b1": 0.1, "h1": 0.0}], sections)
        assert "Overall" in text
        assert "Abstention Behavior" in text      # mirrors the per-run ⚪ name
        assert "💥 Reliability" in text  # mirrors the per-run block name

    def test_empty_behavior_section_absent(self, caplog):
        sections = {"metrics": [(None, ["m1"])],
                    "behavior": [], "health": ["h1"]}
        text = self._render(caplog, [{"m1": 0.5, "h1": 0.0}] * 2, sections)
        assert "Abstention Behavior" not in text

    def test_zero_variance_warning_reads_metrics_only(self, caplog):
        """All-zero Health is the desired state, not caching evidence."""
        sections = {"metrics": [(None, ["m1"])],
                    "behavior": [], "health": ["h1"]}
        text = self._render(
            caplog, [{"m1": 0.4, "h1": 0.0}, {"m1": 0.6, "h1": 0.0}], sections)
        assert "Zero variance" not in text

    def test_zero_variance_warning_fires_on_flat_metrics(self, caplog):
        sections = {"metrics": [(None, ["m1"])],
                    "behavior": [], "health": ["h1"]}
        text = self._render(caplog, [{"m1": 0.4, "h1": 0.0}] * 2, sections)
        assert "Zero variance" in text

    def test_plain_english_labels(self, caplog):
        sections = {"metrics": [(None, ["claim_recall"])],
                    "behavior": [], "health": []}
        text = self._render(caplog, [{"claim_recall": 0.9}] * 2, sections)
        assert "claim recall:" in text
        assert "claim_recall:" not in text


# ── Rule set 1: MECE tree renderer ───────────────────────────────────────────

class TestMeceTree:
    def _render(self, caplog, *args, **kwargs):
        with caplog.at_level(logging.INFO):
            log_mece_tree(*args, **kwargs)
        return caplog.text

    def test_sum_violation_logs_warning(self, caplog):
        text = self._render(caplog, "⚪ T", 5, "items",
                            [("✅", 2, "a", None), ("❌", 2, "b", None)])
        assert "MECE violation" in text

    def test_clean_sum_no_warning(self, caplog):
        text = self._render(caplog, "⚪ T", 4, "items",
                            [("✅", 2, "a", None), ("❌", 2, "b", None)])
        assert "MECE violation" not in text

    def test_footer_zero_denominator_is_na(self, caplog):
        text = self._render(caplog, "⚪ T", 0, "items", [],
                            footer=[("rate", 0, 0)])
        assert "rate n/a" in text

    def test_footer_suffix_for_subset_denominator(self, caplog):
        text = self._render(caplog, "🔎 T", 127, "claims",
                            [("✅", 89, "ok", None), ("❌", 21, "no", None),
                             ("💥", 17, "unjudged", None)],
                            footer=[("accuracy", 89, 110, "judged")])
        assert "(89 / 110 judged)" in text

    def test_last_branch_closes_without_footer(self, caplog):
        text = self._render(caplog, "⚪ T", 3, "items",
                            [("✅", 1, "a", None), ("❌", 2, "b", None)])
        assert "└─ ❌ 2 b" in text


# ── Rule set 2: rate rows renderer ───────────────────────────────────────────

class TestRateRows:
    def _render(self, caplog, rows, **kwargs):
        with caplog.at_level(logging.INFO):
            log_rate_rows("💥 Reliability", rows, **kwargs)
        return caplog.text

    def test_zero_row_always_prints(self, caplog):
        """Hidden ≠ zero: 0 of 52 is information."""
        text = self._render(
            caplog, [("🔎", "Checking", 0, 52, "verdicts missing",
                      "checker_failure_rate", None)])
        assert "0 of 52 verdicts missing" in text
        assert "checker_failure_rate 0.000" in text

    def test_not_measured_state(self, caplog):
        text = self._render(
            caplog, [("🧬", "Atomization", None, 0, "no --atomizer-model",
                      None, None)])
        assert "not measured" in text
        assert "no --atomizer-model" in text

    def test_arrowless_without_exported_key(self, caplog):
        text = self._render(
            caplog, [("🔎", "Checker", 2, 129, "claims unjudged", None, None)])
        assert "→" not in text                    # never invent a key name

    def test_causes_plain_english_sorted_by_count(self, caplog):
        text = self._render(
            caplog, [("📝", "Extraction", 4, 8, "items failed",
                      "extraction_error_rate",
                      {"parse_failure": 1, "timeout": 3})])
        assert "parse failure: 1" in text         # underscore translated
        assert text.index("timeout: 3") < text.index("parse failure: 1")

    def test_zero_denominator_rate_is_na(self, caplog):
        text = self._render(
            caplog, [("🔎", "Checking", 0, 0, "verdicts missing",
                      "checker_failure_rate", None)])
        assert "checker_failure_rate n/a" in text


# ── Run line ─────────────────────────────────────────────────────────────────

class TestRunLine:
    def test_presence_filter_skips_missing_and_null(self, caplog):
        with caplog.at_level(logging.INFO):
            log_run_line(1, 3, 12.5, {"a": 0.5, "b": None}, ("a", "b", "c"))
        message = caplog.records[-1].getMessage()
        assert "a 0.500" in message
        assert "b" not in message                 # null metric dropped
        assert "c" not in message                 # absent metric dropped

    def test_fallback_when_no_metrics(self, caplog):
        with caplog.at_level(logging.INFO):
            log_run_line(2, 3, 1.0, {}, ("a",))
        assert "done" in caplog.text


# ── Shared behavior counts (3.1 SSOT) ────────────────────────────────────────

class TestAbstentionCounts:
    def test_partition_and_error_precedence(self):
        """An errored item is never an abstention, even if flagged as one
        (outcome_markers doctrine)."""
        entries = [
            {},
            {"is_abstention": True},
            {"extraction_errors": {"response": "parse_failure"}},
            {"is_abstention": True,
             "extraction_errors": {"response": "parse_failure"}},
        ]
        counts = abstention_counts(entries)
        assert counts == {"evaluated": 4, "errored": 2,
                          "abstained": 1, "answered": 1}

    def test_empty_universe(self):
        assert abstention_counts([]) == {
            "evaluated": 0, "errored": 0, "abstained": 0, "answered": 0}


# ── Extractor eval surfaced rates (JSON keys, docs rule set 4) ───────────────

class TestExtractorEvalSurfacedRates:
    def _result(self, atomicity=None, duplicates=None):
        from types import SimpleNamespace
        from contextchecker.eval.extractoreval import (
            ExtractorEvaluator, _ItemBucket,
        )
        gt_item = {"gt": [{"s": 1}], "pred": [{"s": 1}]}
        buckets = _ItemBucket(
            to_compare=[gt_item, dict(gt_item)],
            abstention_misread=[{"pred": [{"s": 1}, {"s": 2}]}],
            answer_missed=[],
            abstention_recognized=[{}],
            extraction_error=[{"err": "parse_failure"}],
        )
        dummy = SimpleNamespace(_gt_key="gt", _pred_key="pred",
                                _error_key="err")
        return ExtractorEvaluator._build_result(
            dummy, [], buckets, 5, atomicity, duplicates)

    def test_behavior_rates_condition_on_the_annotation_split(self):
        # unanswerable = 1 justified + 1 unwarranted; answerable = 2 answered
        # + 0 unjustified (SQuAD 2.0's NoAns / HasAns split)
        r = self._result()
        assert r.abstention_recognized_rate == 0.5
        assert r.answer_missed_rate == 0.0
        assert r.abstention_misread_rate == 0.5

    def test_extraction_error_rate_over_attempted(self):
        # attempted = behavioral(4) + errored(1); errored charged here only
        r = self._result()
        assert r.extraction_error_rate == 0.2

    def test_unmeasured_axes_are_none_not_zero(self):
        r = self._result(atomicity=None, duplicates=None)
        assert r.atomicity_rate is None
        assert r.claim_density is None
        assert r.atomization_failure_rate is None
        assert r.duplicate_rate is None

    def test_measured_axes_surface(self):
        atomicity = {"extracted_claims": 24, "evaluated_claims": 22,
                     "atomic_units": 25, "new_claims_from_splits": 3,
                     "non_atomic": 2, "failed": 2,
                     "atomicity_rate": 0.9091, "information_density": 1.14,
                     "splits": []}
        duplicates = {"predicted_claims": 24, "unique_claims": 23,
                      "duplicate_claims": 1, "duplicate_rate": 0.0417,
                      "items": []}
        r = self._result(atomicity=atomicity, duplicates=duplicates)
        assert r.atomicity_rate == 0.9091
        assert r.claim_density == 1.14
        assert r.atomization_failure_rate == round(2 / 24, 4)
        assert r.duplicate_rate == 0.0417

    def test_bucket_rate_empty_universe_is_none(self):
        from contextchecker.eval.extractoreval import _bucket_rate
        assert _bucket_rate(0, 0) is None
        assert _bucket_rate(1, 4) == 0.25


# ── Ragcheck abstention breakdown (judge: retrieval evidence) ────────────────

class TestAbstentionBreakdown:
    def test_split_predicate_and_uncategorized(self):
        from contextchecker.pipelines.ragchecker import _abstention_breakdown
        entries = [
            {"metrics": {"claim_recall": 0.8}},                        # answered
            {"is_abstention": True, "metrics": {"claim_recall": 0.0}},    # unjustified — all chunks irrelevant
            {"is_abstention": True, "metrics": {"claim_recall": 0.5}},    # unjustified — relevant chunk present
            {"is_abstention": True, "metrics": {}},                       # unjustified — relevance unknown
            {"gt_no_answer": True, "is_abstention": True, "metrics": {}}, # justified
            {"gt_no_answer": True, "metrics": {}},                        # unwarranted
            {"extraction_errors": {"response": "x"}, "metrics": {}},      # errored
        ]
        c = _abstention_breakdown(entries)
        # Judge A decides the verdict, Judge B only apportions the cause.
        assert c["answerable"] == 4 and c["unanswerable"] == 2
        assert c["answered_answerable"] == 1
        assert c["justified"] == 1 and c["unwarranted"] == 1
        assert c["unjustified"] == 3
        assert c["all_chunks_irrelevant"] == 1
        assert c["relevant_chunk_present"] == 1
        assert c["relevance_unknown"] == 1
        assert c["abstained"] == 4 and c["errored"] == 1


# ── Verdict cell counts (no-results leave the denominator) ───────────────────

class TestVerdictCellCounts:
    def test_ragcheck_skips_errored_items(self):
        from contextchecker.pipelines.ragchecker import _verdict_cell_counts
        ok = {"answer2response": [{"verdict": "Entailment"}, {"verdict": None}],
              "response2answer": [{"verdict": "Neutral"}],
              "retrieved2response": [[{"verdict": None}]],
              "retrieved2answer": [[{"verdict": "Entailment"}]]}
        errored = {"extraction_errors": {"response": "x"},
                   "answer2response": [{"verdict": None}] * 9,
                   "response2answer": [], "retrieved2response": [],
                   "retrieved2answer": []}
        total, none = _verdict_cell_counts([ok, errored])
        assert (total, none) == (5, 2)   # errored item's cells never counted

    def test_faithcheck_matrix_counts(self):
        from contextchecker.pipelines.faithfulness import _verdict_cell_counts
        ok = {"retrieved2response": [[{"verdict": "Entailment"},
                                      {"verdict": None}],
                                     [{"verdict": "Neutral"}]]}
        errored = {"extraction_errors": {"response": "x"},
                   "retrieved2response": [[{"verdict": None}]]}
        total, none = _verdict_cell_counts([ok, errored])
        assert (total, none) == (3, 1)


# ── Matching footer formatting ───────────────────────────────────────────────

class TestFmtRatio:
    def _fmt(self, *args, **kwargs):
        from contextchecker.eval.extractoreval import ExtractorEvaluator
        return ExtractorEvaluator._fmt_ratio(*args, **kwargs)

    def test_plain_fraction(self):
        assert self._fmt(0.964, 27, 28) == "0.964  (27 / 28)"

    def test_judged_suffix_when_denominator_shrank(self):
        assert self._fmt(0.963, 26, 27, judged=True) == "0.963  (26 / 27 judged)"

    def test_none_value_is_nothing_judged(self):
        assert "nothing judged" in self._fmt(None, 0, 0)
