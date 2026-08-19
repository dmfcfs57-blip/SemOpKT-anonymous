"""Published-anchor comparison for the E0 reproduction audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from semopkt.analysis.aggregate import aggregate_runs
from semopkt.utils.io import read_table, write_json, write_table


REQUIRED_REFERENCE_COLUMNS = {
    "dataset",
    "model",
    "train_size",
    "reference_auc",
}


def reproduction_audit(
    runs_root: str | Path,
    reference_path: str | Path,
    output_root: str | Path,
    tolerance: float = 0.010,
) -> dict[str, Any]:
    """Compare locally rerun E0 cells with an independently supplied anchor table."""

    reference = read_table(reference_path)
    missing = sorted(REQUIRED_REFERENCE_COLUMNS - set(reference.columns))
    if missing:
        raise ValueError(f"Reproduction reference table is missing columns: {missing}")
    if reference[["dataset", "model", "train_size"]].duplicated().any():
        raise ValueError("Reproduction reference cells must be unique")
    metrics = aggregate_runs(runs_root)
    local = metrics[
        (metrics["experiment"] == "E0")
        & (metrics["calibration"] == "uncalibrated")
    ]
    if local.empty:
        raise ValueError("No completed E0 runs were found")
    local = (
        local.groupby(["dataset", "model", "train_size"], as_index=False)
        .agg(
            local_auc=("auc", "mean"),
            local_auc_sd=("auc", "std"),
            seeds=("seed", "nunique"),
        )
    )
    cells = reference.merge(
        local,
        on=["dataset", "model", "train_size"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    cells["auc_difference"] = cells["local_auc"] - cells["reference_auc"]
    cells["absolute_auc_difference"] = cells["auc_difference"].abs()
    cells["within_tolerance"] = (
        cells["_merge"].eq("both")
        & cells["absolute_auc_difference"].le(float(tolerance))
    )
    cells["audit_status"] = cells["_merge"].map(
        {
            "left_only": "missing_local_cell",
            "right_only": "missing_reference_cell",
            "both": "compared",
        }
    ).astype(str)
    compared = cells[cells["_merge"] == "both"]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_table(cells.drop(columns="_merge"), output / "E0_reproduction_cells.csv")
    for dataset, frame in cells.groupby("dataset", dropna=False, sort=True):
        write_table(
            frame.drop(columns="_merge"),
            output / f"E0_{dataset}_reproduction.csv",
        )
    issues = cells[~cells["within_tolerance"]].drop(columns="_merge").copy()
    if not issues.empty:
        issues["diagnostic_checks"] = (
            "dataset_version;truncation_before_split;student_split;"
            "probability_extraction;target_positions;interval_definition"
        )
    write_table(issues, output / "E0_reproduction_issues.csv")
    summary = {
        "tolerance": float(tolerance),
        "reference_cells": int(len(reference)),
        "local_cells": int(len(local)),
        "compared_cells": int(len(compared)),
        "within_tolerance_cells": int(compared["within_tolerance"].sum()),
        "outside_tolerance_cells": int((~compared["within_tolerance"]).sum()),
        "missing_local_cells": int((cells["_merge"] == "left_only").sum()),
        "missing_reference_cells": int((cells["_merge"] == "right_only").sum()),
        "passed": bool(
            len(compared) == len(reference) == len(local)
            and compared["within_tolerance"].all()
        ),
    }
    write_json(output / "E0_reproduction_summary.json", summary)
    if not compared.empty:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        ordered = compared.sort_values(
            ["dataset", "model", "train_size"], kind="mergesort"
        ).copy()
        labels = (
            ordered["dataset"].astype(str)
            + "/"
            + ordered["model"].astype(str)
            + "/K="
            + ordered["train_size"].astype(int).astype(str)
        )
        axis.bar(range(len(ordered)), ordered["auc_difference"])
        axis.axhline(float(tolerance), color="black", linestyle="--", linewidth=1)
        axis.axhline(-float(tolerance), color="black", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(ordered)), labels, rotation=90, fontsize=6)
        axis.set_ylabel("Local minus reference AUC")
        figure.tight_layout()
        figure.savefig(output / "E0_reproduction_difference.pdf", bbox_inches="tight")
        figure.savefig(
            output / "E0_reproduction_difference.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)
    return summary
