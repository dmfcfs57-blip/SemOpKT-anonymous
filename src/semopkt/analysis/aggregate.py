"""Read-only aggregation of completed run records and prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semopkt.utils.hashing import hash_file


def completed_run_directories(root: str | Path) -> list[Path]:
    directories: list[Path] = []
    for completion in sorted(Path(root).rglob("complete.json")):
        directory = completion.parent
        with completion.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "complete":
            raise ValueError(f"Completion marker does not have complete status: {completion}")
        run_record = directory / "run.json"
        if not run_record.exists() or run_record.read_bytes() != completion.read_bytes():
            raise ValueError(f"Run and completion records differ: {directory}")
        expected = record.get("artifact_hashes")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"Completed run lacks artifact hashes: {directory}")
        actual_paths = {
            path.relative_to(directory).as_posix(): path
            for path in directory.rglob("*")
            if path.is_file() and path.name not in {"run.json", "complete.json"}
        }
        if set(actual_paths) != set(expected):
            raise ValueError(f"Completed run artifact set changed: {directory}")
        for relative, path in actual_paths.items():
            if hash_file(path) != expected[relative]:
                raise ValueError(f"Completed run artifact hash mismatch: {path}")
        directories.append(directory)
    return directories


def _portable_run_directory(directory: Path, root: str | Path) -> str:
    root_path = Path(root).resolve()
    try:
        return directory.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return directory.name


def aggregate_runs(root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in completed_run_directories(root):
        with (directory / "complete.json").open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "complete":
            continue
        specification = dict(record["specification"])
        for calibration_key, metric_key in (
            ("uncalibrated", "metrics_uncalibrated"),
            ("calibrated", "metrics_calibrated"),
        ):
            if metric_key not in record:
                continue
            metrics = dict(record[metric_key])
            reporting_allowed = bool(
                record.get("target_audit", {}).get(
                    "primary_auc_reporting_allowed", True
                )
            )
            if not reporting_allowed:
                for column in (
                    "auc",
                    "auprc",
                    "macro_kc_auc",
                    "macro_student_auc",
                ):
                    metrics[column] = float("nan")
            rows.append(
                {
                    **specification,
                    "calibration": calibration_key,
                    **metrics,
                    "best_epoch": record.get("best_epoch"),
                    "total_parameters": record.get("total_parameters"),
                    "trainable_parameters": record.get("trainable_parameters"),
                    "train_seconds": record.get("train_seconds"),
                    "peak_memory_bytes": record.get("peak_memory_bytes"),
                    "auc_reporting_allowed": reporting_allowed,
                    "run_directory": _portable_run_directory(directory, root),
                }
            )
    return pd.DataFrame.from_records(rows)


def load_predictions(
    root: str | Path,
    experiment: str | None = None,
    calibrated: bool = True,
) -> pd.DataFrame:
    file_name = "predictions.csv" if calibrated else "predictions_uncalibrated.csv"
    frames: list[pd.DataFrame] = []
    for directory in completed_run_directories(root):
        path = directory / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if experiment is not None and not frame.empty and str(frame["experiment"].iloc[0]) != experiment:
            continue
        frame["run_directory"] = _portable_run_directory(directory, root)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_summary(
    metrics: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    normal_quantile = 1.959963984540054
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        row["seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for column in value_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            half_width = normal_quantile * standard_error if np.isfinite(standard_error) else float("nan")
            row[f"{column}_mean"] = mean
            row[f"{column}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            row[f"{column}_ci_lower"] = mean - half_width
            row[f"{column}_ci_upper"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)
