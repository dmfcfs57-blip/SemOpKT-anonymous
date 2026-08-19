"""Deterministic paper figures generated from aggregated result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semopkt.analysis.aggregate import (
    aggregate_runs,
    completed_run_directories,
    load_predictions,
)
from semopkt.analysis.postprocess import _e20_cases


def _finish(figure: plt.Figure, path: Path, tight_layout: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_figures(runs_root: str | Path, output_root: str | Path) -> list[Path]:
    metrics = aggregate_runs(runs_root)
    if metrics.empty:
        raise ValueError("No completed runs were found")
    output = Path(output_root)
    products: list[Path] = []
    efficiency_curve = metrics[
        (metrics["experiment"] == "E16")
        & (metrics["calibration"] == "uncalibrated")
    ]
    system = (
        efficiency_curve
        if not efficiency_curve.empty
        else metrics[
            (metrics["experiment"] == "E1")
            & (metrics["calibration"] == "calibrated")
        ]
    )
    if not system.empty:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for model, group in system.groupby("model", sort=True):
            curve = group.groupby("train_size")["auc"].agg(["mean", "sem"]).reset_index()
            axis.errorbar(curve["train_size"], curve["mean"], yerr=1.96 * curve["sem"], marker="o", label=model)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Gradient-training students")
        axis.set_ylabel("AUC")
        axis.legend(ncol=2, fontsize=7)
        path = output / "data_efficiency"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    robustness = metrics[(metrics["experiment"] == "E14") & (metrics["calibration"] == "calibrated")]
    if not robustness.empty:
        pivot = robustness.groupby(["condition", "model"])["auc"].mean().unstack("model")
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("AUC")
        axis.set_xlabel("History perturbation")
        axis.legend(fontsize=7, ncol=2)
        path = output / "robustness"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    early_rows: list[pd.DataFrame] = []
    group_rows: list[pd.DataFrame] = []
    propagation_rows: list[dict[str, object]] = []
    expansion_rows: list[dict[str, object]] = []
    scaling_rows: list[dict[str, object]] = []
    adaptation_rows: list[pd.DataFrame] = []
    stability_rows: list[pd.DataFrame] = []
    propagation_matrices: list[tuple[dict[str, object], Path]] = []
    for directory in completed_run_directories(runs_root):
        with (directory / "complete.json").open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        specification = record["specification"]
        history_path = directory / "history_window_metrics.csv"
        if history_path.exists():
            frame = pd.read_csv(history_path)
            frame["model"] = specification["model"]
            frame["dataset"] = specification["dataset"]
            frame["seed"] = specification["seed"]
            early_rows.append(frame)
        group_path = directory / "group_metrics.csv"
        if group_path.exists():
            frame = pd.read_csv(group_path)
            frame["model"] = specification["model"]
            frame["dataset"] = specification["dataset"]
            group_rows.append(frame)
        adaptation_path = directory / "online_adaptation_metrics.csv"
        if adaptation_path.exists():
            frame = pd.read_csv(adaptation_path)
            frame["model"] = specification["model"]
            frame["dataset"] = specification["dataset"]
            frame["seed"] = specification["seed"]
            adaptation_rows.append(frame)
        stability_path = directory / "old_concept_stability_summary.csv"
        if stability_path.exists():
            frame = pd.read_csv(stability_path)
            frame["model"] = specification["model"]
            frame["dataset"] = specification["dataset"]
            frame["seed"] = specification["seed"]
            stability_rows.append(frame)
        if record.get("propagation"):
            propagation_rows.append({**specification, **record["propagation"]})
            matrix_path = directory / "propagation_matrices.npz"
            if matrix_path.exists():
                propagation_matrices.append((specification, matrix_path))
        if record.get("expansion"):
            row = dict(specification)
            old = record["expansion"].get("old_concept_metrics") or {}
            added = record["expansion"].get("added_concept_metrics") or {}
            row.update(
                {
                    "old_auc": old.get("auc"),
                    "added_auc": added.get("auc"),
                    "old_drift": record["expansion"].get(
                        "old_concept_mean_absolute_drift"
                    ),
                }
            )
            expansion_rows.append(row)
        if record.get("efficiency"):
            for query in record["efficiency"].get("query_scaling", []):
                scaling_rows.append({**specification, **query})
    if early_rows:
        early = pd.concat(early_rows, ignore_index=True)
        order = ["Q2--5", "Q6--10", "Q11--20", "Q21--50"]
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for model, group in early.groupby("model", sort=True):
            curve = group.groupby("window")["auc"].mean().reindex(order).dropna()
            axis.plot(curve.index, curve.values, marker="o", label=model)
        axis.set_xlabel("Target position window")
        axis.set_ylabel("AUC")
        axis.legend(fontsize=7, ncol=2)
        path = output / "new_student_history"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    calibration = load_predictions(runs_root, experiment="E15", calibrated=True)
    if not calibration.empty:
        calibration = calibration[calibration["protocol"] == "calibration_system"]
    if not calibration.empty:
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for model, group in calibration.groupby("model", sort=True):
            grouped_bins = pd.qcut(
                group["probability"], q=min(10, len(group)), duplicates="drop"
            )
            reliability = group.groupby(grouped_bins, observed=True).agg(
                confidence=("probability", "mean"), accuracy=("correct", "mean")
            )
            axes[0].plot(
                reliability["confidence"], reliability["accuracy"], marker="o", label=model
            )
            probability = group["probability"].to_numpy(dtype=float)
            labels = group["correct"].to_numpy(dtype=int)
            entropy = -probability * np.log(np.clip(probability, 1.0e-12, 1.0)) - (
                1.0 - probability
            ) * np.log(np.clip(1.0 - probability, 1.0e-12, 1.0))
            order_index = np.argsort(entropy, kind="mergesort")
            errors = (probability >= 0.5).astype(int) != labels
            coverage = np.arange(1, len(errors) + 1) / len(errors)
            risk = np.cumsum(errors[order_index]) / np.arange(1, len(errors) + 1)
            axes[1].plot(coverage, risk, label=model)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axes[0].set(xlabel="Mean predicted probability", ylabel="Observed correctness")
        axes[1].set(xlabel="Coverage", ylabel="Risk")
        axes[0].legend(fontsize=7)
        path = output / "calibration_selective_risk"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    if propagation_rows:
        propagation = pd.DataFrame.from_records(propagation_rows)
        summary = propagation.groupby("model")[["spearman", "ndcg_at_5", "top5_overlap"]].mean()
        figure, axis = plt.subplots(figsize=(7.0, 4.0))
        summary.plot(kind="bar", ax=axis)
        axis.set_ylabel("Alignment")
        axis.set_xlabel("Model")
        axis.legend(fontsize=7)
        path = output / "propagation_alignment"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    if propagation_matrices:
        semopkt_matrices = [
            item
            for item in propagation_matrices
            if item[0].get("model") == "SemOpKT"
        ]
        selected = sorted(
            semopkt_matrices or propagation_matrices,
            key=lambda item: (
                str(item[0].get("dataset")),
                int(item[0].get("seed", 0)),
                str(item[0].get("model")),
            ),
        )[0]
        matrix = np.load(selected[1])
        empirical = np.asarray(matrix["empirical"], dtype=float)
        influence = np.asarray(matrix["influence"], dtype=float)
        limit = float(
            np.nanquantile(
                np.abs(np.concatenate([empirical.ravel(), influence.ravel()])),
                0.98,
            )
        )
        limit = max(limit, 1.0e-8)
        figure, axes = plt.subplots(
            1, 2, figsize=(10.0, 4.2), constrained_layout=True
        )
        for axis, values, title in (
            (axes[0], empirical, "Adjusted empirical association"),
            (axes[1], influence, "Response-conditioned model influence"),
        ):
            image = axis.imshow(
                values,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="auto",
            )
            axis.set(title=title, xlabel="Target concept", ylabel="Source concept")
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        path = output / "propagation_matrices"
        _finish(figure, path, tight_layout=False)
        products.append(path.with_suffix(".pdf"))
    if expansion_rows:
        expansion = pd.DataFrame.from_records(expansion_rows).dropna(
            subset=["added_auc"]
        )
        if not expansion.empty:
            figure, axis = plt.subplots(figsize=(7.0, 4.0))
            for model, group in expansion.groupby("model", sort=True):
                curve = group.groupby("condition")["added_auc"].mean()
                axis.plot(curve.index, curve.values, marker="o", label=model)
            axis.set(xlabel="Visible vocabulary condition", ylabel="Added-concept AUC")
            axis.legend(fontsize=7)
            path = output / "vocabulary_expansion"
            _finish(figure, path)
            products.append(path.with_suffix(".pdf"))
    if group_rows:
        groups = pd.concat(group_rows, ignore_index=True)
        groups = groups[groups["group_type"] == "semantic_distance_group"]
        if not groups.empty:
            figure, axis = plt.subplots(figsize=(6.4, 4.0))
            for model, group in groups.groupby("model", sort=True):
                curve = group.groupby("group")["auc"].mean().sort_index()
                axis.plot(curve.index, curve.values, marker="o", label=model)
            axis.set(xlabel="Nearest-training semantic-distance quartile", ylabel="AUC")
            axis.legend(fontsize=7)
            path = output / "semantic_distance_errors"
            _finish(figure, path)
            products.append(path.with_suffix(".pdf"))
    if adaptation_rows:
        adaptation = pd.concat(adaptation_rows, ignore_index=True)
        order = ["0", "1", "2", "4+"]
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        adaptation["group"] = adaptation["group"].astype(str)
        for model, group in adaptation.groupby("model", sort=True):
            curve = group.groupby("group")["auc"].mean().reindex(order).dropna()
            axis.plot(curve.index, curve.values, marker="o", label=model)
        axis.set(
            xlabel="Prior observations of the held-out concept",
            ylabel="AUC",
        )
        axis.legend(fontsize=7, ncol=2)
        path = output / "online_concept_adaptation"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    if stability_rows:
        stability = pd.concat(stability_rows, ignore_index=True)
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for model, group in stability.groupby("model", sort=True):
            curve = group.groupby("prior_target_observations")[
                "old_concept_mean_absolute_drift"
            ].mean()
            axis.plot(curve.index, curve.values, marker="o", label=model)
        axis.set(
            xlabel="Prior target-concept observations",
            ylabel="Old-concept mean absolute drift",
        )
        axis.legend(fontsize=7, ncol=2)
        path = output / "old_concept_stability"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    sensitivity = metrics[
        (metrics["experiment"] == "E17")
        & (metrics["calibration"] == "uncalibrated")
    ]
    if not sensitivity.empty:
        pareto = sensitivity.groupby("condition", as_index=False).agg(
            auc=("auc", "mean"),
            train_seconds=("train_seconds", "mean"),
            trainable_parameters=("trainable_parameters", "mean"),
        )
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        sizes = 20.0 + 80.0 * (
            pareto["trainable_parameters"]
            / max(1.0, float(pareto["trainable_parameters"].max()))
        )
        axis.scatter(pareto["train_seconds"], pareto["auc"], s=sizes, alpha=0.75)
        for row in pareto.itertuples(index=False):
            axis.annotate(
                str(row.condition),
                (float(row.train_seconds), float(row.auc)),
                fontsize=5,
                alpha=0.8,
            )
        axis.set(xlabel="Mean training time (s)", ylabel="AUC")
        path = output / "sensitivity_pareto"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    if scaling_rows:
        scaling = pd.DataFrame.from_records(scaling_rows)
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for model, group in scaling.groupby("model", sort=True):
            curve = group.groupby("query_count")["latency_median_ms"].mean()
            axis.plot(curve.index, curve.values, marker="o", label=model)
        axis.set(xlabel="Cached concept queries", ylabel="Median latency (ms)")
        axis.legend(fontsize=7)
        path = output / "query_scaling"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    cases = _e20_cases(completed_run_directories(runs_root))
    required_case_columns = {
        "selection_rule",
        "probability_before",
        "probability_after",
        "current_concept_change",
        "related_mean_absolute_change",
        "unrelated_mean_absolute_change",
    }
    if not cases.empty and required_case_columns.issubset(cases.columns):
        selected_cases = cases.sort_values(
            ["dataset_condition", "selection_rule"], kind="mergesort"
        ).reset_index(drop=True)
        labels = (
            selected_cases["dataset_condition"].astype(str)
            + "/"
            + selected_cases["selection_rule"].astype(str)
        )
        x = np.arange(len(selected_cases))
        figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
        width = 0.38
        axes[0].bar(
            x - width / 2,
            selected_cases["probability_before"],
            width,
            label="Before update",
        )
        axes[0].bar(
            x + width / 2,
            selected_cases["probability_after"],
            width,
            label="After update",
        )
        axes[0].set_ylabel("Target probability")
        axes[0].legend(fontsize=7)
        axes[1].plot(
            x,
            selected_cases["current_concept_change"].abs(),
            marker="o",
            label="Current concept",
        )
        axes[1].plot(
            x,
            selected_cases["related_mean_absolute_change"],
            marker="s",
            label="Related concepts",
        )
        axes[1].plot(
            x,
            selected_cases["unrelated_mean_absolute_change"],
            marker="^",
            label="Unrelated concepts",
        )
        axes[1].set(
            ylabel="Absolute field-induced probability change",
            xticks=x,
            xticklabels=labels,
        )
        axes[1].tick_params(axis="x", rotation=45, labelsize=6)
        axes[1].legend(fontsize=7, ncol=3)
        path = output / "qualitative_field_changes"
        _finish(figure, path)
        products.append(path.with_suffix(".pdf"))
    return products
