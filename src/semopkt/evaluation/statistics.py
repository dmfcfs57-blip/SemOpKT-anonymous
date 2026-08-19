"""Hierarchical and student-paired bootstrap inference with Holm correction."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from semopkt.evaluation.metrics import safe_auc


def paired_student_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    metric: Callable[[np.ndarray, np.ndarray], float] = safe_auc,
) -> dict[str, float]:
    keys = ["dataset", "student_id", "question_id", "kc_id", "position", "correct"]
    left = reference[keys + ["probability"]].rename(columns={"probability": "reference"})
    right = comparison[keys + ["probability"]].rename(columns={"probability": "comparison"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired predictions do not have identical target rows")
    student_groups = [
        group.index.to_numpy()
        for _, group in merged.groupby(["dataset", "student_id"], sort=True)
    ]
    if len(student_groups) < 2:
        raise ValueError("Paired student bootstrap requires at least two students")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        sampled = rng.integers(0, len(student_groups), size=len(student_groups))
        indices = np.concatenate([student_groups[index] for index in sampled])
        labels = merged.loc[indices, "correct"].to_numpy(dtype=int)
        ref = metric(labels, merged.loc[indices, "reference"].to_numpy(dtype=float))
        cmp = metric(labels, merged.loc[indices, "comparison"].to_numpy(dtype=float))
        differences[iteration] = ref - cmp
    alpha = 1.0 - confidence
    observed = metric(
        merged["correct"].to_numpy(dtype=int), merged["reference"].to_numpy(dtype=float)
    ) - metric(
        merged["correct"].to_numpy(dtype=int), merged["comparison"].to_numpy(dtype=float)
    )
    lower, upper = np.nanquantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = 2.0 * min(
        (np.sum(differences <= 0.0) + 1) / (resamples + 1),
        (np.sum(differences >= 0.0) + 1) / (resamples + 1),
    )
    return {
        "difference": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(1.0, p_value)),
        "resamples": int(resamples),
        "students": int(len(student_groups)),
    }


def paired_cell_student_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    cell_columns: Sequence[str],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Average paired AUC differences while resampling students within cells."""

    target_keys = [
        *cell_columns,
        "dataset",
        "student_id",
        "question_id",
        "kc_id",
        "position",
        "correct",
    ]
    target_keys = list(dict.fromkeys(target_keys))
    left = reference[target_keys + ["probability"]].rename(
        columns={"probability": "reference"}
    )
    right = comparison[target_keys + ["probability"]].rename(
        columns={"probability": "comparison"}
    )
    merged = left.merge(right, on=target_keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Cell-stratified predictions do not have identical target rows")
    cells: list[tuple[list[np.ndarray], pd.DataFrame]] = []
    observed_cells: list[float] = []
    for _, cell in merged.groupby(list(cell_columns), dropna=False, sort=True):
        groups = [
            group.index.to_numpy()
            for _, group in cell.groupby(["dataset", "student_id"], sort=True)
        ]
        labels = cell["correct"].to_numpy(dtype=int)
        observed_cells.append(
            safe_auc(labels, cell["reference"].to_numpy(dtype=float))
            - safe_auc(labels, cell["comparison"].to_numpy(dtype=float))
        )
        cells.append((groups, merged))
    if not cells:
        raise ValueError("No paired cells are available")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        cell_differences: list[float] = []
        for groups, source in cells:
            sampled = rng.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in sampled])
            labels = source.loc[indices, "correct"].to_numpy(dtype=int)
            if np.unique(labels).size < 2:
                continue
            cell_differences.append(
                safe_auc(labels, source.loc[indices, "reference"].to_numpy(dtype=float))
                - safe_auc(labels, source.loc[indices, "comparison"].to_numpy(dtype=float))
            )
        draws[iteration] = float(np.mean(cell_differences)) if cell_differences else np.nan
    alpha = 1.0 - confidence
    lower, upper = np.nanquantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = 2.0 * min(
        (np.sum(draws <= 0.0) + 1) / (np.sum(np.isfinite(draws)) + 1),
        (np.sum(draws >= 0.0) + 1) / (np.sum(np.isfinite(draws)) + 1),
    )
    return {
        "difference": float(np.nanmean(observed_cells)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(1.0, p_value)),
        "resamples": int(resamples),
        "cells": int(len(cells)),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def hierarchical_seed_interval(
    values: pd.DataFrame,
    value_column: str,
    dataset_column: str = "dataset",
    seed_column: str = "seed",
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    grouped = values.groupby([dataset_column, seed_column], sort=True)[value_column].mean().reset_index()
    datasets = sorted(grouped[dataset_column].unique().tolist())
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        observations: list[float] = []
        for dataset in sampled_datasets:
            subset = grouped[grouped[dataset_column] == dataset]
            sampled_rows = rng.integers(0, len(subset), size=len(subset))
            observations.extend(subset.iloc[sampled_rows][value_column].astype(float).tolist())
        draws[iteration] = float(np.mean(observations))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(grouped[value_column].mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "resamples": int(resamples),
    }
