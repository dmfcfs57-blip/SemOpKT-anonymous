"""Interaction, student, concept, calibration, and selective-risk metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan")


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    if len(labels) == 0:
        return float("nan")
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, min(bins, len(order)))
    error = 0.0
    for indices in groups:
        if len(indices) == 0:
            continue
        confidence = float(np.mean(probabilities[indices]))
        accuracy = float(np.mean(labels[indices]))
        error += len(indices) / len(labels) * abs(confidence - accuracy)
    return float(error)


def _macro_auc(frame: pd.DataFrame, group_column: str) -> tuple[float, int, int]:
    values: list[float] = []
    invalid = 0
    for _, group in frame.groupby(group_column, sort=False):
        auc = safe_auc(group["correct"].to_numpy(), group["probability"].to_numpy())
        if np.isnan(auc):
            invalid += 1
        else:
            values.append(auc)
    return (float(np.mean(values)) if values else float("nan"), len(values), invalid)


def selective_risk(frame: pd.DataFrame, coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)) -> dict[str, float]:
    probabilities = frame["probability"].to_numpy(dtype=float)
    labels = frame["correct"].to_numpy(dtype=int)
    entropy = -(
        probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0))
        + (1.0 - probabilities) * np.log(np.clip(1.0 - probabilities, 1.0e-12, 1.0))
    )
    order = np.argsort(entropy, kind="mergesort")
    errors = (probabilities >= 0.5).astype(int) != labels
    cumulative = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    coverage_axis = np.arange(1, len(errors) + 1) / len(errors)
    result = {"aurc": float(np.trapezoid(cumulative, coverage_axis))}
    for coverage in coverages:
        count = max(1, int(np.floor(len(errors) * coverage)))
        result[f"error_at_{int(round(coverage * 100))}"] = float(np.mean(errors[order[:count]]))
    return result


def compute_metrics(frame: pd.DataFrame, ece_bins: int = 15) -> dict[str, Any]:
    required = {"correct", "probability", "student_id", "kc_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction table is empty")
    labels = frame["correct"].to_numpy(dtype=int)
    probabilities = np.clip(frame["probability"].to_numpy(dtype=float), 1.0e-7, 1.0 - 1.0e-7)
    positive_rate = float(np.mean(labels))
    metrics: dict[str, Any] = {
        "count": int(len(frame)),
        "students": int(frame["student_id"].nunique()),
        "concepts": int(frame["kc_id"].nunique()),
        "positive_rate": positive_rate,
        "auc": safe_auc(labels, probabilities),
        "auprc": float(average_precision_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan"),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities, bins=ece_bins),
    }
    macro_kc, valid_kc, invalid_kc = _macro_auc(frame, "kc_id")
    macro_student, valid_student, invalid_student = _macro_auc(frame, "student_id")
    metrics.update(
        {
            "macro_kc_auc": macro_kc,
            "macro_kc_valid": valid_kc,
            "macro_kc_single_label": invalid_kc,
            "macro_student_auc": macro_student,
            "macro_student_valid": valid_student,
            "macro_student_single_label": invalid_student,
        }
    )
    metrics.update(selective_risk(frame))
    return metrics


def metrics_by_history_window(frame: pd.DataFrame, ece_bins: int = 15) -> pd.DataFrame:
    windows = ((1, 1, "Q1"), (2, 5, "Q2--5"), (6, 10, "Q6--10"), (11, 20, "Q11--20"), (21, 50, "Q21--50"))
    rows: list[dict[str, Any]] = []
    for start, end, name in windows:
        subset = frame[frame["position"].between(start, end)]
        if subset.empty:
            continue
        rows.append({"window": name, **compute_metrics(subset, ece_bins=ece_bins)})
    return pd.DataFrame(rows)

