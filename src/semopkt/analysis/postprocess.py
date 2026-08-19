"""Cross-run products for paired and protocol-specific analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semopkt.analysis.aggregate import completed_run_directories
from semopkt.analysis.aggregate import aggregate_runs
from semopkt.analysis.learning_curves import learning_curve_summary
from semopkt.analysis.qualitative import select_cases
from semopkt.evaluation.metrics import compute_metrics
from semopkt.utils.io import write_table


def _record(directory: Path) -> dict[str, Any]:
    with (directory / "complete.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _specification_columns(record: dict[str, Any]) -> dict[str, Any]:
    specification = dict(record["specification"])
    return {
        key: specification.get(key)
        for key in (
            "experiment",
            "dataset",
            "protocol",
            "model",
            "seed",
            "train_size",
            "holdout_ratio",
            "condition",
            "source_dataset",
            "target_dataset",
            "adaptation_size",
        )
    }


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), child, target)
    elif not isinstance(value, (list, tuple)):
        target[prefix] = value


def _auxiliary_records(directories: list[Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[dict[str, Any]]] = {
        "E12_expansion": [],
        "E13_propagation": [],
        "E14_robustness": [],
        "E18_efficiency": [],
        "E18_query_scaling": [],
        "E11_operator_efficiency": [],
    }
    file_products = {
        "E3_history_windows": "history_window_metrics.csv",
        "E7_online_adaptation": "online_adaptation_metrics.csv",
        "E19_error_groups": "group_metrics.csv",
    }
    frame_tables: dict[str, list[pd.DataFrame]] = {key: [] for key in file_products}
    for directory in directories:
        record = _record(directory)
        specification = _specification_columns(record)
        for name, file_name in file_products.items():
            path = directory / file_name
            if path.exists():
                frame = pd.read_csv(path)
                for key, value in specification.items():
                    frame[key] = value
                frame_tables[name].append(frame)
        for name, record_key in (
            ("E12_expansion", "expansion"),
            ("E13_propagation", "propagation"),
            ("E14_robustness", "robustness"),
        ):
            value = record.get(record_key)
            if value:
                row = dict(specification)
                _flatten("", value, row)
                tables[name].append(row)
        efficiency = record.get("efficiency")
        experiment = specification.get("experiment")
        if efficiency and experiment in {"E11", "E18"}:
            name = (
                "E11_operator_efficiency"
                if experiment == "E11"
                else "E18_efficiency"
            )
            row = dict(specification)
            _flatten("", efficiency, row)
            tables[name].append(row)
            if experiment == "E18":
                for query_row in efficiency.get("query_scaling", []):
                    tables["E18_query_scaling"].append(
                        {**specification, **query_row}
                    )
    result = {
        name: pd.DataFrame.from_records(rows) for name, rows in tables.items() if rows
    }
    result.update(
        {
            name: pd.concat(frames, ignore_index=True)
            for name, frames in frame_tables.items()
            if frames
        }
    )
    return result


def _e10_paraphrase_drift(directories: list[Path]) -> pd.DataFrame:
    runs: dict[tuple[Any, ...], dict[str, Path]] = {}
    for directory in directories:
        record = _record(directory)
        specification = record["specification"]
        if specification.get("experiment") != "E10":
            continue
        key = tuple(
            specification.get(column)
            for column in (
                "dataset",
                "protocol",
                "model",
                "seed",
                "train_size",
                "holdout_ratio",
            )
        )
        runs.setdefault(key, {})[str(specification.get("condition"))] = directory
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "student_id", "source_row_id", "correct", "position"]
    for key, conditions in runs.items():
        if not {"original", "paraphrase"}.issubset(conditions):
            continue
        original = pd.read_csv(conditions["original"] / "predictions_uncalibrated.csv")
        paraphrase = pd.read_csv(conditions["paraphrase"] / "predictions_uncalibrated.csv")
        paired = original[keys + ["probability"]].rename(
            columns={"probability": "original_probability"}
        ).merge(
            paraphrase[keys + ["probability"]].rename(
                columns={"probability": "paraphrase_probability"}
            ),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(original) or len(paired) != len(paraphrase):
            raise ValueError("E10 paraphrase comparison is not fully paired")
        dataset, protocol, model, seed, train_size, holdout_ratio = key
        difference = (
            paired["original_probability"] - paired["paraphrase_probability"]
        ).abs()
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "model": model,
                "seed": seed,
                "train_size": train_size,
                "holdout_ratio": holdout_ratio,
                "count": len(paired),
                "mean_absolute_prediction_difference": float(difference.mean()),
                "median_absolute_prediction_difference": float(difference.median()),
                "p95_absolute_prediction_difference": float(np.quantile(difference, 0.95)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _e12_protocol_expansion_drift(directories: list[Path]) -> pd.DataFrame:
    """Pair old-concept predictions from reduced and full vocabulary fits."""

    full_runs: dict[tuple[str, str, int], Path] = {}
    expansion_runs: list[tuple[dict[str, Any], Path]] = []
    for directory in directories:
        specification = _record(directory)["specification"]
        if specification.get("experiment") != "E12":
            continue
        protocol = str(specification.get("protocol"))
        key = (
            str(specification.get("dataset")),
            str(specification.get("model")),
            int(specification.get("seed")),
        )
        if protocol == "vocabulary_full":
            if key in full_runs:
                raise ValueError(f"Duplicate E12 full-vocabulary run for {key}")
            full_runs[key] = directory
        elif protocol == "vocabulary_expansion":
            expansion_runs.append((specification, directory))

    keys = [
        "dataset",
        "student_id",
        "source_row_id",
        "question_id",
        "kc_id",
        "correct",
        "position",
    ]
    rows: list[dict[str, Any]] = []
    for specification, directory in expansion_runs:
        run_key = (
            str(specification["dataset"]),
            str(specification["model"]),
            int(specification["seed"]),
        )
        full_directory = full_runs.get(run_key)
        if full_directory is None:
            continue
        expansion = pd.read_csv(directory / "predictions_uncalibrated.csv")
        full = pd.read_csv(full_directory / "predictions_uncalibrated.csv")
        old = expansion[expansion["seen_status"] == "seen"].copy()
        added = expansion[expansion["seen_status"] == "unseen"].copy()
        if old.empty:
            raise ValueError(f"E12 expansion run has no old-concept targets: {directory}")
        paired = old[keys + ["probability"]].rename(
            columns={"probability": "reduced_vocabulary_probability"}
        ).merge(
            full[keys + ["probability"]].rename(
                columns={"probability": "full_vocabulary_probability"}
            ),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(old):
            raise ValueError(
                "E12 reduced/full vocabulary runs do not share every old-concept target"
            )
        reduced_metrics = compute_metrics(
            paired[keys + ["reduced_vocabulary_probability"]].rename(
                columns={"reduced_vocabulary_probability": "probability"}
            )
        )
        full_metrics = compute_metrics(
            paired[keys + ["full_vocabulary_probability"]].rename(
                columns={"full_vocabulary_probability": "probability"}
            )
        )
        added_metrics = compute_metrics(added) if not added.empty else None
        drift = (
            paired["reduced_vocabulary_probability"]
            - paired["full_vocabulary_probability"]
        ).abs()
        row: dict[str, Any] = {
            "dataset": specification["dataset"],
            "model": specification["model"],
            "seed": specification["seed"],
            "condition": specification.get("condition"),
            "holdout_ratio": specification.get("holdout_ratio"),
            "paired_old_targets": int(len(paired)),
            "added_targets": int(len(added)),
            "protocol_mean_absolute_old_prediction_drift": float(drift.mean()),
            "protocol_median_absolute_old_prediction_drift": float(drift.median()),
            "protocol_p95_absolute_old_prediction_drift": float(
                np.quantile(drift, 0.95)
            ),
            "reduced_vocabulary_old_auc": reduced_metrics["auc"],
            "full_vocabulary_old_auc": full_metrics["auc"],
            "reduced_vocabulary_old_nll": reduced_metrics["nll"],
            "full_vocabulary_old_nll": full_metrics["nll"],
            "comparison_requires_independent_global_fits": True,
            "query_only_effect": False,
        }
        if added_metrics is not None:
            row.update(
                {
                    "added_auc": added_metrics["auc"],
                    "added_nll": added_metrics["nll"],
                    "added_brier": added_metrics["brier"],
                }
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _e1_rank_products(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subset = metrics[
        (metrics["experiment"] == "E1")
        & (metrics["calibration"] == "uncalibrated")
    ]
    if subset.empty:
        return {}
    conditions = (
        subset.groupby(["dataset", "train_size", "model"], as_index=False)["auc"]
        .mean()
        .rename(columns={"auc": "mean_auc"})
    )
    conditions["rank"] = conditions.groupby(["dataset", "train_size"])[
        "mean_auc"
    ].rank(method="average", ascending=False)
    model_summary = (
        conditions.groupby("model", as_index=False)
        .agg(
            average_rank=("rank", "mean"),
            first_place_conditions=("rank", lambda value: int((value == 1.0).sum())),
            evaluated_conditions=("rank", "size"),
            mean_auc=("mean_auc", "mean"),
        )
        .sort_values(["average_rank", "model"], kind="mergesort")
    )
    reference = conditions[conditions["model"] == "SemOpKT"][
        ["dataset", "train_size", "mean_auc"]
    ].rename(columns={"mean_auc": "semopkt_auc"})
    paired = conditions[conditions["model"] != "SemOpKT"].merge(
        reference,
        on=["dataset", "train_size"],
        how="inner",
        validate="many_to_one",
    )
    if not paired.empty:
        paired["difference_semopkt_minus_baseline"] = (
            paired["semopkt_auc"] - paired["mean_auc"]
        )
        tolerance = 5.0e-7
        paired["outcome"] = np.where(
            paired["difference_semopkt_minus_baseline"] > tolerance,
            "win",
            np.where(
                paired["difference_semopkt_minus_baseline"] < -tolerance,
                "loss",
                "tie",
            ),
        )
        outcomes = (
            paired.pivot_table(
                index="model",
                columns="outcome",
                values="dataset",
                aggfunc="size",
                fill_value=0,
            )
            .reset_index()
            .rename(columns={"model": "comparison_model"})
        )
        for column in ("win", "tie", "loss"):
            if column not in outcomes:
                outcomes[column] = 0
    else:
        outcomes = pd.DataFrame()
    products = {
        "E1_condition_ranks": conditions.sort_values(
            ["dataset", "train_size", "rank", "model"], kind="mergesort"
        ),
        "E1_model_rank_summary": model_summary,
    }
    if not outcomes.empty:
        products["E1_semopkt_win_tie_loss"] = outcomes
    return products


def _e8_transfer_products(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subset = metrics[
        (metrics["experiment"] == "E8")
        & (metrics["calibration"] == "uncalibrated")
    ].copy()
    if subset.empty:
        return {}
    target_only = subset[subset["protocol"] == "target_only"][
        ["target_dataset", "model", "seed", "adaptation_size", "auc"]
    ].rename(columns={"auc": "target_only_auc"})
    transfer = subset[
        subset["protocol"].isin(
            ["source_target_adaptation", "multi_source_target_adaptation"]
        )
    ]
    paired = transfer.merge(
        target_only,
        on=["target_dataset", "model", "seed", "adaptation_size"],
        how="inner",
        validate="many_to_one",
    )
    products: dict[str, pd.DataFrame] = {}
    if not paired.empty:
        paired["transfer_minus_target_only_auc"] = (
            paired["auc"] - paired["target_only_auc"]
        )
        paired["negative_transfer"] = paired[
            "transfer_minus_target_only_auc"
        ] < 0.0
        products["E8_negative_transfer_pairs"] = paired
        products["E8_negative_transfer_summary"] = (
            paired.groupby(
                [
                    "protocol",
                    "source_dataset",
                    "target_dataset",
                    "model",
                    "condition",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                conditions=("negative_transfer", "size"),
                negative_transfer_conditions=("negative_transfer", "sum"),
                negative_transfer_rate=("negative_transfer", "mean"),
                mean_auc_difference=("transfer_minus_target_only_auc", "mean"),
            )
        )
    dedup = subset[
        subset["condition"].isin(["overlap_retained", "near_duplicates_removed"])
    ].copy()
    if not dedup.empty:
        dedup["adaptation_key"] = dedup["adaptation_size"].fillna(-1).astype(int)
        keys = [
            "protocol",
            "source_dataset",
            "target_dataset",
            "model",
            "seed",
            "adaptation_key",
        ]
        retained = dedup[dedup["condition"] == "overlap_retained"][
            keys + ["auc"]
        ].rename(columns={"auc": "overlap_retained_auc"})
        removed = dedup[dedup["condition"] == "near_duplicates_removed"][
            keys + ["auc"]
        ].rename(columns={"auc": "near_duplicates_removed_auc"})
        comparison = retained.merge(
            removed, on=keys, how="inner", validate="one_to_one"
        )
        if not comparison.empty:
            comparison["removed_minus_retained_auc"] = (
                comparison["near_duplicates_removed_auc"]
                - comparison["overlap_retained_auc"]
            )
            products["E8_near_duplicate_effect"] = comparison
    return products


def _e20_cases(directories: list[Path]) -> pd.DataFrame:
    runs: dict[tuple[Any, ...], dict[str, Path]] = {}
    seeds: dict[tuple[Any, ...], int] = {}
    for directory in directories:
        record = _record(directory)
        specification = record["specification"]
        if specification.get("experiment") != "E20":
            continue
        key = tuple(
            specification.get(column)
            for column in ("dataset", "protocol", "train_size", "holdout_ratio", "seed")
        )
        runs.setdefault(key, {})[str(specification["model"])] = directory
        seeds[key] = int(specification["seed"])
    frames: list[pd.DataFrame] = []
    protocol_rules = {
        "qualitative_system": ["random", "median_gain", "failure"],
        "qualitative_unseen": ["unseen_concept"],
        "qualitative_double": ["double_cold_start"],
        "qualitative": None,
    }
    for key, models in runs.items():
        if not {"SemOpKT", "CLST"}.issubset(models):
            continue
        semopkt = pd.read_csv(models["SemOpKT"] / "predictions.csv")
        baseline = pd.read_csv(models["CLST"] / "predictions.csv")
        protocol = str(key[1])
        cases = select_cases(
            semopkt,
            baseline,
            seeds[key],
            rules=protocol_rules.get(protocol),
        )
        trace_path = models["SemOpKT"] / "field_traces.csv"
        if trace_path.exists():
            traces = pd.read_csv(trace_path)
            cases = cases.merge(
                traces,
                on=[
                    "dataset",
                    "student_id",
                    "source_row_id",
                    "position",
                    "kc_id",
                    "correct",
                ],
                how="left",
                validate="many_to_one",
            )
        dataset, protocol, train_size, holdout_ratio, seed = key
        cases.insert(0, "dataset_condition", dataset)
        cases.insert(1, "protocol_condition", protocol)
        cases.insert(2, "train_size", train_size)
        cases.insert(3, "holdout_ratio", holdout_ratio)
        cases.insert(4, "seed", seed)
        frames.append(cases)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def postprocess_runs(runs_root: str | Path, output_root: str | Path) -> dict[str, Path]:
    directories = completed_run_directories(runs_root)
    if not directories:
        raise ValueError("No completed runs were found")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    products: dict[str, Path] = {}
    metrics = aggregate_runs(runs_root)
    tables = _auxiliary_records(directories)
    tables.update(_e1_rank_products(metrics))
    tables.update(_e8_transfer_products(metrics))
    paraphrase = _e10_paraphrase_drift(directories)
    if not paraphrase.empty:
        tables["E10_paraphrase_drift"] = paraphrase
    expansion_drift = _e12_protocol_expansion_drift(directories)
    if not expansion_drift.empty:
        tables["E12_protocol_expansion_drift"] = expansion_drift
    cases = _e20_cases(directories)
    if not cases.empty:
        tables["E20_qualitative_cases"] = cases
    learning_curves = learning_curve_summary(metrics)
    if not learning_curves.empty:
        tables["E16_learning_curve_fits"] = learning_curves
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        write_table(frame, path)
        products[name] = path
    return products
