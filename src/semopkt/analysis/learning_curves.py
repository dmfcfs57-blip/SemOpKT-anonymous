"""Power-law data-efficiency fits and equivalent sample-size estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _curve(sample_size: np.ndarray, asymptote: float, gap: float, exponent: float) -> np.ndarray:
    return asymptote - gap * np.power(sample_size, -exponent)


def fit_power_law(sample_size: np.ndarray, auc: np.ndarray) -> dict[str, float]:
    x = np.asarray(sample_size, dtype=np.float64)
    y = np.asarray(auc, dtype=np.float64)
    if len(np.unique(x)) < 4:
        raise ValueError("Power-law fitting requires at least four training sizes")
    parameters, _ = curve_fit(
        _curve,
        x,
        y,
        p0=(min(0.999, float(np.max(y) + 0.03)), max(0.01, float(np.ptp(y))), 0.5),
        bounds=([0.5, 0.0, 1.0e-4], [1.0, 2.0, 5.0]),
        maxfev=20000,
    )
    prediction = _curve(x, *parameters)
    residual = y - prediction
    return {
        "asymptote": float(parameters[0]),
        "gap": float(parameters[1]),
        "exponent": float(parameters[2]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def required_sample_size(parameters: dict[str, float], target_auc: float) -> float:
    asymptote = parameters["asymptote"]
    gap = parameters["gap"]
    exponent = parameters["exponent"]
    if target_auc >= asymptote or gap <= 0 or exponent <= 0:
        return float("inf")
    return float((gap / (asymptote - target_auc)) ** (1.0 / exponent))


def learning_curve_summary(
    metrics: pd.DataFrame,
    bootstrap_resamples: int = 1000,
    seed: int = 161803,
) -> pd.DataFrame:
    subset = metrics[
        (metrics["experiment"] == "E16")
        & (metrics["calibration"] == "uncalibrated")
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    means = (
        subset.groupby(["dataset", "model", "train_size"], as_index=False)["auc"]
        .mean()
        .sort_values("train_size")
    )
    clst_rows = means[(means["model"] == "CLST")]
    clst_at_64 = clst_rows[clst_rows["train_size"] == 64]
    clst_target = (
        clst_at_64.groupby("dataset")["auc"].first().to_dict()
        if not clst_at_64.empty
        else clst_rows.sort_values("train_size").groupby("dataset")["auc"].last().to_dict()
    )
    semopkt_target = (
        means[(means["model"] == "SemOpKT")]
        .sort_values("train_size")
        .groupby("dataset")["auc"]
        .last()
        .mul(0.95)
        .to_dict()
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (dataset, model), group in means.groupby(["dataset", "model"], sort=True):
        try:
            fit = fit_power_law(
                group["train_size"].to_numpy(dtype=float),
                group["auc"].to_numpy(dtype=float),
            )
        except (RuntimeError, ValueError):
            continue
        row: dict[str, Any] = {"dataset": dataset, "model": model, **fit}
        for target_name, targets in (
            ("clst_final", clst_target),
            ("semopkt_95_percent", semopkt_target),
        ):
            target = targets.get(dataset)
            row[f"{target_name}_auc"] = target
            row[f"{target_name}_required_students"] = (
                required_sample_size(fit, float(target)) if target is not None else np.nan
            )
        model_seed = subset[
            (subset["dataset"] == dataset) & (subset["model"] == model)
        ]
        seeds = sorted(model_seed["seed"].unique())
        draws: list[float] = []
        target = semopkt_target.get(dataset)
        if target is not None and len(seeds) > 1:
            for _ in range(bootstrap_resamples):
                sampled = rng.choice(seeds, size=len(seeds), replace=True)
                sampled_rows = pd.concat(
                    [model_seed[model_seed["seed"] == value] for value in sampled],
                    ignore_index=True,
                )
                sampled_means = sampled_rows.groupby("train_size", as_index=False)["auc"].mean()
                try:
                    sampled_fit = fit_power_law(
                        sampled_means["train_size"].to_numpy(dtype=float),
                        sampled_means["auc"].to_numpy(dtype=float),
                    )
                    draws.append(required_sample_size(sampled_fit, float(target)))
                except (RuntimeError, ValueError):
                    continue
        finite = np.asarray([value for value in draws if np.isfinite(value)], dtype=float)
        row["semopkt_95_required_ci_lower"] = (
            float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        )
        row["semopkt_95_required_ci_upper"] = (
            float(np.quantile(finite, 0.975)) if len(finite) else np.nan
        )
        row["bootstrap_valid_fits"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame.from_records(rows)
