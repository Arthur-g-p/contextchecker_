"""Unit tests for the custom metric helpers in claimlens.eval.metrics."""

import pytest
from claimlens.eval.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class TestAccuracyScore:
    """Tests for accuracy_score function."""

    def test_basic_accuracy(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 0, 1]
        # Matches: [0, 2] (pred matches true at index 0 and 2) -> 2 / 6 = 0.5 (indices 0 and 2 matches: 0==0, 2==2, 0==0, so index 0, 2, 3 are matches, wait:
        # y_true[0]=0, y_pred[0]=0 (match)
        # y_true[1]=1, y_pred[1]=2 (no match)
        # y_true[2]=2, y_pred[2]=2 (match)
        # y_true[3]=0, y_pred[3]=0 (match)
        # y_true[4]=1, y_pred[4]=0 (no match)
        # y_true[5]=2, y_pred[5]=1 (no match)
        # Total matches: 3. Total: 6. Accuracy = 3 / 6 = 0.5.
        assert accuracy_score(y_true, y_pred) == 0.5

    def test_perfect_accuracy(self):
        y_true = ["A", "B", "C"]
        y_pred = ["A", "B", "C"]
        assert accuracy_score(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        y_true = [1, 2, 3]
        y_pred = [3, 1, 2]
        assert accuracy_score(y_true, y_pred) == 0.0

    def test_empty_lists(self):
        assert accuracy_score([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            accuracy_score([1, 2], [1])


class TestF1Score:
    """Tests for f1_score function."""

    def test_binary_f1_perfect(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 1, 0, 1]
        assert f1_score(y_true, y_pred, pos_label=1) == 1.0

    def test_binary_f1_partial(self):
        # tp = 2, fp = 1 (index 3 is True=0, Pred=1), fn = 1 (index 1 is True=1, Pred=0)
        y_true = [1, 1, 1, 0]
        y_pred = [1, 0, 1, 1]
        # precision = 2 / 3, recall = 2 / 3, f1 = 2/3
        assert pytest.approx(f1_score(y_true, y_pred, pos_label=1)) == 2 / 3

    def test_binary_f1_zero_division(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        # pos_label=1 -> tp=0, fp=0, fn=0
        assert f1_score(y_true, y_pred, pos_label=1, zero_division=0) == 0.0
        assert f1_score(y_true, y_pred, pos_label=1, zero_division=1) == 1.0

    def test_macro_f1(self):
        # Labels are 0, 1, 2.
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        # Sets/Labels: {0, 1, 2}
        # For class 0:
        # tp=2 (indices 0,3), fp=1 (index 4), fn=0
        # p = 2/3, r = 2/2 = 1.0, f1 = 2 * (2/3) * 1 / (5/3) = 4/3 / (5/3) = 4/5 = 0.8
        # For class 1:
        # tp=0, fp=2 (indices 2,5), fn=2 (indices 1,4)
        # p=0, r=0, f1=0
        # For class 2:
        # tp=0, fp=1 (index 1), fn=2 (indices 2,5)
        # p=0, r=0, f1=0
        # Macro F1 = (0.8 + 0.0 + 0.0) / 3 = 0.8 / 3
        expected = 0.8 / 3
        assert pytest.approx(f1_score(y_true, y_pred, average="macro")) == expected

    def test_empty_lists(self):
        assert f1_score([], [], pos_label=1, zero_division=0) == 0.0
        assert f1_score([], [], pos_label=1, zero_division=1) == 1.0

    def test_unsupported_average(self):
        with pytest.raises(ValueError, match="not supported"):
            f1_score([1], [1], average="micro")


class TestConfusionMatrix:
    """Tests for confusion_matrix function."""

    def test_basic_confusion_matrix(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 0, 1]
        # Labels inferred: [0, 1, 2]
        # cm[i][j] where i is true, j is pred
        # Pair (0, 0) -> true=0, pred=0 (index 0)
        # Pair (1, 2) -> true=1, pred=2 (index 1)
        # Pair (2, 2) -> true=2, pred=2 (index 2)
        # Pair (0, 0) -> true=0, pred=0 (index 3)
        # Pair (1, 0) -> true=1, pred=0 (index 4)
        # Pair (2, 1) -> true=2, pred=1 (index 5)
        # Counts:
        # Row 0 (true=0): col 0 = 2, col 1 = 0, col 2 = 0
        # Row 1 (true=1): col 0 = 1, col 1 = 0, col 2 = 1
        # Row 2 (true=2): col 0 = 0, col 1 = 1, col 2 = 1
        # CM should be:
        # [[2, 0, 0],
        #  [1, 0, 1],
        #  [0, 1, 1]]
        expected = [
            [2, 0, 0],
            [1, 0, 1],
            [0, 1, 1]
        ]
        assert confusion_matrix(y_true, y_pred) == expected

    def test_custom_labels(self):
        y_true = ["A", "B", "A"]
        y_pred = ["B", "B", "A"]
        # Matrix using custom labels order: ["B", "A"]
        # Row 0 is "B": pred "B" -> 1. pred "A" -> 0
        # Row 1 is "A": pred "B" -> 1. pred "A" -> 1
        expected = [
            [1, 0],
            [1, 1]
        ]
        assert confusion_matrix(y_true, y_pred, labels=["B", "A"]) == expected


class TestClassificationReport:
    """Tests for classification_report function."""

    def test_output_dict(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 0, 1]
        # true: 1, 0, 1, 1, 0, 1 -> 4 ones, 2 zeros
        # pred: 1, 0, 1, 0, 0, 1 -> 3 ones, 3 zeros
        # class 0: tp=2 (indices 1,4), fp=1 (index 3), fn=0. support=2
        # class 0: p = 2/3, r = 1.0, f1 = 2*(2/3)*1/(5/3) = 0.8
        # class 1: tp=3 (indices 0,2,5), fp=0, fn=1 (index 3). support=4
        # class 1: p = 1.0, r = 3/4 = 0.75, f1 = 2*1*0.75/(1.75) = 1.5/1.75 = 6/7 = 0.857...
        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
        
        assert isinstance(report, dict)
        assert "0" in report
        assert "1" in report
        assert pytest.approx(report["0"]["precision"]) == 2/3
        assert report["0"]["recall"] == 1.0
        assert pytest.approx(report["0"]["f1-score"]) == 0.8
        assert report["0"]["support"] == 2

        assert report["1"]["precision"] == 1.0
        assert report["1"]["recall"] == 0.75
        assert pytest.approx(report["1"]["f1-score"]) == 6/7
        assert report["1"]["support"] == 4

        assert pytest.approx(report["accuracy"]) == 5/6
        assert pytest.approx(report["macro avg"]["precision"]) == (2/3 + 1.0) / 2
        assert pytest.approx(report["weighted avg"]["f1-score"]) == (0.8 * 2 + (6/7) * 4) / 6

    def test_output_str(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 0, 1]
        report_str = classification_report(y_true, y_pred, labels=[0, 1], digits=3, output_dict=False)
        assert isinstance(report_str, str)
        # Check alignment and header presence
        assert "precision" in report_str
        assert "recall" in report_str
        assert "f1-score" in report_str
        assert "support" in report_str
        assert "macro avg" in report_str
        assert "weighted avg" in report_str
        assert "accuracy" in report_str
        # Verify specific values are formatted with 3 digits
        assert "0.667" in report_str
        assert "0.857" in report_str
