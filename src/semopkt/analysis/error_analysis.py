"""Pre-registered semantic-distance, frequency, history, and error groups."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from semopkt.evaluation.metrics import compute_metrics
from semopkt.data.sequences import ConceptVocabulary
from semopkt.data.schema import parse_component_cell


def _quantile_groups(
    reference: np.ndarray, values: np.ndarray, prefix: str = "Q"
) -> np.ndarray:
    finite = reference[np.isfinite(reference)]
    if len(finite) < 2:
        return np.full(len(values), f"{prefix}1", dtype=object)
    boundaries = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
    indices = np.searchsorted(boundaries, values, side="right") + 1
    return np.asarray([f"{prefix}{index}" for index in indices], dtype=object)


def prediction_covariates(
    training: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    vocabulary: ConceptVocabulary,
) -> pd.DataFrame:
    """Attach groups whose cut points are determined from training data only."""

    ordered_test = test.sort_values(
        ["student_id", "position", "source_row_id"], kind="mergesort"
    ).copy()
    ordered_test["history_length"] = ordered_test.groupby("student_id").cumcount()
    cumulative = ordered_test.groupby("student_id")["correct"].cumsum() - ordered_test["correct"]
    ordered_test["history_accuracy"] = cumulative / ordered_test["history_length"].replace(0, np.nan)
    ordered_test["prior_same_kc"] = ordered_test.groupby(["student_id", "kc_id"]).cumcount()
    ordered_test["first_kc_encounter"] = ordered_test["prior_same_kc"].eq(0)

    ordered_train = training.sort_values(
        ["student_id", "position", "source_row_id"], kind="mergesort"
    ).copy()
    training_history_length = ordered_train.groupby("student_id").cumcount().to_numpy(dtype=float)
    training_cumulative = (
        ordered_train.groupby("student_id")["correct"].cumsum() - ordered_train["correct"]
    )
    training_history_accuracy = (
        training_cumulative
        / pd.Series(training_history_length, index=ordered_train.index).replace(0, np.nan)
    ).to_numpy(dtype=float)

    frequency = training.groupby("kc_id").size()
    if len(frequency) >= 3:
        lower, upper = frequency.quantile([1.0 / 3.0, 2.0 / 3.0])
    else:
        lower = upper = float(frequency.max()) if len(frequency) else 0.0

    def frequency_group(kc_id: str) -> str:
        count = int(frequency.get(kc_id, 0))
        if count <= lower:
            return "rare"
        if count <= upper:
            return "medium"
        return "frequent"

    matrix = np.asarray(vocabulary.embeddings, dtype=np.float64)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-8, None)
    train_indices = np.asarray(
        sorted(
            {
                vocabulary.text_to_index[str(text)]
                for text in training["kc_text_norm"].astype(str).unique()
            }
        ),
        dtype=int,
    )
    if len(train_indices):
        train_matrix = matrix[train_indices]
        target_indices = np.asarray(
            [vocabulary.text_to_index[str(text)] for text in ordered_test["kc_text_norm"]],
            dtype=int,
        )
        target_distance = 1.0 - np.max(matrix[target_indices] @ train_matrix.T, axis=1)
        if len(train_indices) > 1:
            similarity = train_matrix @ train_matrix.T
            np.fill_diagonal(similarity, -np.inf)
            reference_distance = 1.0 - np.max(similarity, axis=1)
        else:
            reference_distance = np.asarray([0.0])
    else:
        target_distance = np.full(len(ordered_test), np.nan)
        reference_distance = np.asarray([np.nan])

    covariates = ordered_test[
        [
            "source_row_id",
            "kc_text_norm",
            "history_length",
            "history_accuracy",
            "prior_same_kc",
            "first_kc_encounter",
        ]
    ].copy()
    covariates["component_count"] = ordered_test["kc_components"].map(
        lambda value: len(parse_component_cell(value))
    )
    covariates["semantic_distance"] = target_distance
    covariates["semantic_distance_group"] = _quantile_groups(
        reference_distance, target_distance
    )
    covariates["frequency_group"] = ordered_test["kc_id"].astype(str).map(frequency_group)
    covariates["history_length_group"] = _quantile_groups(
        training_history_length,
        covariates["history_length"].to_numpy(dtype=float),
    )
    covariates["history_accuracy_group"] = _quantile_groups(
        training_history_accuracy,
        covariates["history_accuracy"].to_numpy(dtype=float),
    )
    covariates["target_label_group"] = ordered_test["correct"].map(
        {0: "incorrect", 1: "correct"}
    )
    covariates["first_encounter_group"] = covariates["first_kc_encounter"].map(
        {False: "repeat", True: "first"}
    )
    if covariates["source_row_id"].duplicated().any():
        raise ValueError("Source row IDs must be unique for prediction covariates")
    return predictions.merge(
        covariates,
        on="source_row_id",
        how="left",
        validate="many_to_one",
    )


def assign_training_frequency_groups(
    training: pd.DataFrame, predictions: pd.DataFrame
) -> pd.Series:
    counts = training.groupby("kc_id").size()
    if len(counts) < 3:
        return pd.Series("unstratified", index=predictions.index)
    lower, upper = counts.quantile([1.0 / 3.0, 2.0 / 3.0])

    def category(kc_id: str) -> str:
        count = int(counts.get(kc_id, 0))
        if count <= lower:
            return "rare"
        if count <= upper:
            return "medium"
        return "frequent"

    return predictions["kc_id"].astype(str).map(category)


def grouped_metrics(
    predictions: pd.DataFrame,
    groups: Mapping[str, pd.Series],
    ece_bins: int = 15,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, assignments in groups.items():
        aligned = assignments.reindex(predictions.index)
        for value in sorted(aligned.dropna().unique().tolist()):
            subset = predictions[aligned == value]
            if subset.empty:
                continue
            rows.append(
                {
                    "group_type": group_name,
                    "group": value,
                    **compute_metrics(subset, ece_bins=ece_bins),
                }
            )
    return pd.DataFrame(rows)


def high_confidence_errors(predictions: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    probabilities = predictions["probability"].to_numpy(dtype=float)
    labels = predictions["correct"].to_numpy(dtype=int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    incorrect = (probabilities >= 0.5).astype(int) != labels
    result = predictions.loc[incorrect & (confidence >= threshold)].copy()
    result["confidence"] = confidence[incorrect & (confidence >= threshold)]
    return result.sort_values("confidence", ascending=False, kind="mergesort")


def classify_error_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Assign the pre-registered, non-exclusive descriptive error categories."""

    probability = predictions["probability"].to_numpy(dtype=float)
    labels = predictions["correct"].to_numpy(dtype=int)
    incorrect = (probability >= 0.5).astype(int) != labels
    result = predictions.loc[incorrect].copy()
    categories: list[str] = []
    for row in result.itertuples(index=False):
        values: list[str] = []
        if getattr(row, "semantic_distance_group", None) == "Q1":
            values.append("semantic_near_prediction_failure")
        descriptor = str(getattr(row, "kc_text_norm", ""))
        if len(descriptor.split()) <= 1:
            values.append("ambiguous_short_descriptor")
        history_accuracy = float(getattr(row, "history_accuracy", np.nan))
        if np.isfinite(history_accuracy):
            prediction = float(row.probability) >= 0.5
            if (history_accuracy >= 0.75 and not prediction) or (
                history_accuracy <= 0.25 and prediction
            ):
                values.append("contradictory_history")
        if getattr(row, "frequency_group", None) == "rare":
            values.append("extreme_long_tail")
        if int(getattr(row, "component_count", 1)) > 1:
            values.append("multi_component_item")
        confidence = max(float(row.probability), 1.0 - float(row.probability))
        if confidence >= 0.90:
            values.append("high_confidence_error")
        categories.append(";".join(values or ["uncategorized"]))
    result["error_category"] = categories
    result["confidence"] = np.maximum(
        result["probability"].to_numpy(dtype=float),
        1.0 - result["probability"].to_numpy(dtype=float),
    )
    return result.sort_values("confidence", ascending=False, kind="mergesort")
