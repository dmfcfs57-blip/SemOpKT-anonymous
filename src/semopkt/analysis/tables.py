"""Generate CSV, Markdown, and LaTeX tables directly from run metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from semopkt.analysis.aggregate import aggregate_runs, seed_summary
from semopkt.utils.io import atomic_write_text, write_table

METRICS = ["auc", "auprc", "accuracy", "nll", "brier", "ece", "macro_kc_auc", "macro_student_auc"]


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines) + "\n"


def _latex(frame: pd.DataFrame) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        return "".join(replacements.get(character, character) for character in text).replace(
            "\n", " "
        )

    alignment = "l" * len(frame.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", r"\hline"]
    lines.append(" & ".join(cell(column) for column in frame.columns) + r" \\")
    lines.append(r"\hline")
    lines.extend(
        " & ".join(cell(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_formats(frame: pd.DataFrame, base: Path) -> None:
    write_table(frame, base.with_suffix(".csv"))
    atomic_write_text(base.with_suffix(".md"), _markdown(frame))
    atomic_write_text(base.with_suffix(".tex"), _latex(frame))


def generate_tables(runs_root: str | Path, output_root: str | Path) -> dict[str, Path]:
    metrics = aggregate_runs(runs_root)
    if metrics.empty:
        raise ValueError("No completed runs were found")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    products: dict[str, Path] = {}
    experiment_groups: dict[str, list[str]] = {
        "E0_sanity": ["dataset", "model", "train_size", "calibration"],
        "E1_system": ["dataset", "model", "train_size", "calibration"],
        "E2_full": ["dataset", "model", "calibration"],
        "E3_early": ["dataset", "model", "train_size", "calibration"],
        "E4_random_unseen": ["dataset", "model", "holdout_ratio", "calibration"],
        "E5_cluster_unseen": ["dataset", "model", "holdout_ratio", "calibration"],
        "E6_double": ["dataset", "protocol", "model", "calibration"],
        "E7_online": ["dataset", "model", "calibration"],
        "E8_transfer": [
            "protocol",
            "source_dataset",
            "target_dataset",
            "condition",
            "model",
            "adaptation_size",
            "calibration",
        ],
        "E9_ablation": ["dataset", "protocol", "model", "train_size", "calibration"],
        "E10_text": ["dataset", "protocol", "condition", "model", "calibration"],
        "E11_operator": ["dataset", "protocol", "model", "calibration"],
        "E12_resolution": ["dataset", "protocol", "condition", "model", "calibration"],
        "E14_robustness": ["dataset", "condition", "model", "calibration"],
        "E15_calibration": ["dataset", "protocol", "model", "calibration"],
        "E16_efficiency_curve": ["dataset", "model", "train_size", "calibration"],
        "E17_sensitivity": ["dataset", "condition", "model", "calibration"],
        "E18_runtime": ["dataset", "model", "calibration"],
        "E19_errors": ["dataset", "model", "calibration"],
    }
    for name, groups in experiment_groups.items():
        experiment = name.split("_", 1)[0]
        subset = metrics[metrics["experiment"] == experiment]
        if subset.empty:
            continue
        available_groups = [column for column in groups if column in subset.columns]
        summary = seed_summary(subset, available_groups, [column for column in METRICS if column in subset])
        base = output / name
        _write_formats(summary, base)
        products[name] = base.with_suffix(".csv")
    audit_columns = [
        column
        for column in (
            "experiment",
            "dataset",
            "protocol",
            "model",
            "seed",
            "total_parameters",
            "trainable_parameters",
            "best_epoch",
            "train_seconds",
            "peak_memory_bytes",
        )
        if column in metrics.columns
    ]
    audit = metrics[audit_columns].drop_duplicates().sort_values(audit_columns[:5])
    _write_formats(audit, output / "run_audit")
    products["run_audit"] = output / "run_audit.csv"
    return products
