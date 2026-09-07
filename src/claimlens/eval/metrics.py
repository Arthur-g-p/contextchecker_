"""Lightweight, pure-Python implementations of scikit-learn evaluation metrics.

Provides:
  - accuracy_score
  - f1_score
  - confusion_matrix
  - classification_report
"""

def accuracy_score(y_true: list, y_pred: list) -> float:
    """Compute the accuracy classification score.

    Parameters
    ----------
    y_true : list
        Ground truth (correct) labels.
    y_pred : list
        Predicted labels.

    Returns
    -------
    float
        The accuracy score (fraction of correctly classified samples).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def f1_score(
    y_true: list,
    y_pred: list,
    pos_label=1,
    average: str = "binary",
    zero_division: int = 0
) -> float:
    """Compute the F1 score.

    Parameters
    ----------
    y_true : list
        Ground truth (correct) labels.
    y_pred : list
        Predicted labels.
    pos_label : object, default=1
        The class to report if average='binary'.
    average : str, default='binary'
        Determines the type of averaging performed on the data.
        Supported: 'binary', 'macro'.
    zero_division : {0, 1}, default=0
        Sets the value to return when there is a zero division.

    Returns
    -------
    float
        The F1 score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        return float(zero_division)

    if average == "binary":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

        prec_denom = tp + fp
        precision = tp / prec_denom if prec_denom > 0 else float(zero_division)

        rec_denom = tp + fn
        recall = tp / rec_denom if rec_denom > 0 else float(zero_division)

        f1_denom = precision + recall
        f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else float(zero_division)
        return f1
    
    elif average == "macro":
        labels = sorted(list(set(y_true) | set(y_pred)))
        if not labels:
            return float(zero_division)
        f1s = []
        for label in labels:
            f1s.append(f1_score(y_true, y_pred, pos_label=label, average="binary", zero_division=zero_division))
        return sum(f1s) / len(labels)
    
    else:
        raise ValueError(f"average={average} is not supported. Use 'binary' or 'macro'.")


def confusion_matrix(y_true: list, y_pred: list, labels: list | None = None) -> list[list[int]]:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : list
        Ground truth (correct) labels.
    y_pred : list
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix. If None, those that appear at least
        once in y_true or y_pred are used in sorted order.

    Returns
    -------
    list of list of int
        Confusion matrix where rows are actual classes and columns are predicted classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    label_to_index = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    cm = [[0] * n_labels for _ in range(n_labels)]

    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t]][label_to_index[p]] += 1

    return cm


def classification_report(
    y_true: list,
    y_pred: list,
    labels: list | None = None,
    target_names: list[str] | None = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: int = 0
) -> str | dict:
    """Build a text report or dictionary showing the main classification metrics.

    Parameters
    ----------
    y_true : list
        Ground truth (correct) labels.
    y_pred : list
        Predicted labels.
    labels : list, optional
        List of label indices to include in the report.
    target_names : list of str, optional
        Optional display names matching the labels (in the same order).
    digits : int, default=2
        Number of digits for formatting output floating point values (when returning string).
    output_dict : bool, default=False
        If True, return a dictionary of metrics.
    zero_division : {0, 1}, default=0
        Sets the value to return when there is a zero division.

    Returns
    -------
    str or dict
        Text summary or dictionary representation of precision, recall, f1-score, and support.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    if target_names is None:
        target_names = [str(label) for label in labels]
    elif len(target_names) != len(labels):
        raise ValueError("target_names must have the same length as labels.")

    # Calculate metrics per label
    metrics_by_label = {}
    total_support = 0
    total_correct = 0

    for label, name in zip(labels, target_names):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        prec_denom = tp + fp
        precision = tp / prec_denom if prec_denom > 0 else float(zero_division)

        rec_denom = tp + fn
        recall = tp / rec_denom if rec_denom > 0 else float(zero_division)

        f1_denom = precision + recall
        f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else float(zero_division)

        metrics_by_label[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
        total_support += support
        total_correct += tp

    # Calculate global / average metrics
    accuracy = total_correct / len(y_true) if len(y_true) > 0 else float(zero_division)

    macro_precision = sum(metrics_by_label[name]["precision"] for name in target_names) / len(labels) if labels else 0.0
    macro_recall = sum(metrics_by_label[name]["recall"] for name in target_names) / len(labels) if labels else 0.0
    macro_f1 = sum(metrics_by_label[name]["f1-score"] for name in target_names) / len(labels) if labels else 0.0

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    if total_support > 0:
        weighted_precision = sum(
            metrics_by_label[name]["precision"] * metrics_by_label[name]["support"] for name in target_names
        ) / total_support
        weighted_recall = sum(
            metrics_by_label[name]["recall"] * metrics_by_label[name]["support"] for name in target_names
        ) / total_support
        weighted_f1 = sum(
            metrics_by_label[name]["f1-score"] * metrics_by_label[name]["support"] for name in target_names
        ) / total_support

    # Return dict if requested
    if output_dict:
        result_dict = {}
        for name in target_names:
            result_dict[name] = metrics_by_label[name]
        result_dict["accuracy"] = accuracy
        result_dict["macro avg"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_support
        }
        result_dict["weighted avg"] = {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": total_support
        }
        return result_dict

    # Otherwise construct text report
    max_len = max(len(name) for name in target_names) if target_names else 0
    max_len = max(max_len, len("weighted avg"), len("macro avg"))

    report = []
    header = f"{'':>{max_len}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"
    report.append(header)
    report.append("")

    for name in target_names:
        m = metrics_by_label[name]
        line = (
            f"{name:>{max_len}}  "
            f"{m['precision']:.{digits}f}  "
            f"{m['recall']:.{digits}f}  "
            f"{m['f1-score']:.{digits}f}  "
            f"{m['support']:>9}"
        )
        report.append(line)

    report.append("")
    acc_line = (
        f"{'accuracy':>{max_len}}  "
        f"{'':>9}  "
        f"{'':>9}  "
        f"{accuracy:.{digits}f}  "
        f"{total_support:>9}"
    )
    report.append(acc_line)

    macro_line = (
        f"{'macro avg':>{max_len}}  "
        f"{macro_precision:.{digits}f}  "
        f"{macro_recall:.{digits}f}  "
        f"{macro_f1:.{digits}f}  "
        f"{total_support:>9}"
    )
    report.append(macro_line)

    weighted_line = (
        f"{'weighted avg':>{max_len}}  "
        f"{weighted_precision:.{digits}f}  "
        f"{weighted_recall:.{digits}f}  "
        f"{weighted_f1:.{digits}f}  "
        f"{total_support:>9}"
    )
    report.append(weighted_line)
    report.append("")

    return "\n".join(report)
