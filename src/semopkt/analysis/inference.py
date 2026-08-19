"""Pre-registered paired inference for the system-level comparison family."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from semopkt.analysis.aggregate import load_predictions
from semopkt.evaluation.statistics import holm_adjust, paired_cell_student_bootstrap
from semopkt.utils.io import write_table


DEFAULT_COMPARISONS = (
    "simpleKT",
    "GKT",
    "csKT",
    "UKT",
    "EAKT",
    "MAHKT",
    "CLST",
)


def system_comparison_inference(
    runs_root: str | Path,
    output_path: str | Path,
    comparison_models: Sequence[str] = DEFAULT_COMPARISONS,
    reference_model: str = "SemOpKT",
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 271828,
) -> pd.DataFrame:
    predictions = load_predictions(runs_root, experiment="E1", calibrated=False)
    if predictions.empty:
        raise ValueError("No E1 prediction files were found")
    reference = predictions[predictions["model"] == reference_model]
    if reference.empty:
        raise ValueError(f"No predictions found for reference model {reference_model}")
    rows = []
    for index, model in enumerate(comparison_models):
        comparison = predictions[predictions["model"] == model]
        if comparison.empty:
            continue
        result = paired_cell_student_bootstrap(
            reference,
            comparison,
            cell_columns=["dataset", "seed", "train_size"],
            resamples=resamples,
            confidence=confidence,
            seed=seed + index,
        )
        rows.append(
            {
                "scope": "all_system_conditions",
                "dataset": None,
                "train_size": None,
                "reference_model": reference_model,
                "comparison_model": model,
                **result,
            }
        )
        conditions = reference[["dataset", "train_size"]].drop_duplicates()
        for condition_index, condition in conditions.iterrows():
            dataset = condition["dataset"]
            train_size = condition["train_size"]
            reference_condition = reference[
                (reference["dataset"] == dataset)
                & (reference["train_size"] == train_size)
            ]
            comparison_condition = comparison[
                (comparison["dataset"] == dataset)
                & (comparison["train_size"] == train_size)
            ]
            if reference_condition.empty or comparison_condition.empty:
                continue
            result = paired_cell_student_bootstrap(
                reference_condition,
                comparison_condition,
                cell_columns=["seed"],
                resamples=resamples,
                confidence=confidence,
                seed=seed + 1000 + index * 100 + int(condition_index),
            )
            rows.append(
                {
                    "scope": "dataset_train_size",
                    "dataset": dataset,
                    "train_size": train_size,
                    "reference_model": reference_model,
                    "comparison_model": model,
                    **result,
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("None of the requested comparison models had E1 predictions")
    frame["p_value_holm"] = np.nan
    for scope, indices in frame.groupby("scope", sort=False).groups.items():
        frame.loc[indices, "p_value_holm"] = holm_adjust(
            frame.loc[indices, "p_value"].to_numpy(dtype=float)
        )
    frame["reject_at_0_05"] = frame["p_value_holm"] < 0.05
    write_table(frame, output_path)
    return frame
