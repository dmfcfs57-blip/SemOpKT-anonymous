"""Validation-only equal-budget hyperparameter search."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from semopkt.config import deep_merge
from semopkt.data.sequences import build_sequences
from semopkt.data.splits import apply_manifest
from semopkt.evaluation.metrics import compute_metrics
from semopkt.experiments.runner import ExperimentRunner
from semopkt.experiments.specs import RunSpec
from semopkt.models.registry import build_model
from semopkt.training.trainer import Trainer
from semopkt.utils.hashing import hash_json
from semopkt.utils.io import write_table, write_yaml
from semopkt.utils.random import seed_everything


def _candidate_grid(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "learning_rates",
        "hidden_sizes",
        "dropouts",
        "layers",
        "regularization",
    )
    values = [list(settings[key]) for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def validation_search(
    configuration: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    output_root: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    runner = ExperimentRunner(configuration, device=device)
    settings = runner.config["hyperparameter_search"]
    frame = runner._load_dataset(dataset)
    specification = RunSpec("tuning", dataset, "system", model_name, seed, train_size=64)
    manifest = runner._manifest(frame, specification)
    split = apply_manifest(frame, manifest, 64)
    model_config = runner._resolved_model_config({}, model_name)
    vocabulary, encoder_metadata = runner._vocabulary(
        frame, "default", seed, model_config=model_config
    )
    maximum_length = int(model_config["training"]["max_sequence_length"])
    train_sequences = build_sequences(
        split.train, vocabulary, maximum_length=maximum_length
    )
    validation_sequences = build_sequences(
        split.validation, vocabulary, maximum_length=maximum_length
    )
    candidates = _candidate_grid(settings)
    maximum_candidates = int(settings.get("maximum_candidates", len(candidates)))
    rng = np.random.default_rng(seed)
    if len(candidates) > maximum_candidates:
        selected_indices = sorted(
            rng.choice(len(candidates), size=maximum_candidates, replace=False).tolist()
        )
        candidates = [candidates[index] for index in selected_indices]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        seed_everything(seed, deterministic=True)
        candidate_model = deep_merge(model_config, {})
        candidate_baselines = deep_merge(runner.baseline_config, {})
        if model_name == "SemOpKT":
            candidate_model = deep_merge(
                candidate_model,
                {
                    "dimensions": {"hidden": int(candidate["hidden_sizes"])},
                    "architecture": {
                        "dropout": float(candidate["dropouts"]),
                        "layers": int(candidate["layers"]),
                    },
                    "regularization": {
                        "smoothness": float(candidate["regularization"]),
                        "stability": float(candidate["regularization"]),
                    },
                    "training": {
                        "learning_rate": float(candidate["learning_rates"])
                    },
                },
            )
            training_config = candidate_model["training"]
        else:
            model_settings = dict(candidate_baselines["models"][model_name])
            if "hidden_size" in model_settings:
                model_settings["hidden_size"] = int(candidate["hidden_sizes"])
            if "dropout" in model_settings:
                model_settings["dropout"] = float(candidate["dropouts"])
            if "layers" in model_settings:
                model_settings["layers"] = int(candidate["layers"])
            candidate_baselines["models"][model_name] = model_settings
            candidate_baselines["common_training"]["learning_rate"] = float(
                candidate["learning_rates"]
            )
            candidate_baselines["common_training"]["weight_decay"] = float(
                candidate["regularization"]
            )
            training_config = deep_merge(
                candidate_baselines["common_training"], model_settings
            )
        candidate_model["dimensions"]["text"] = int(vocabulary.embeddings.shape[1])
        model = build_model(
            model_name,
            candidate_model,
            candidate_baselines,
            vocabulary.embeddings,
            seed,
            concept_texts=vocabulary.texts,
        )
        key = hash_json(candidate)[:12]
        trainer = Trainer(model, training_config, output / key, device=device)
        result = trainer.fit(
            train_sequences,
            validation_sequences,
            validation_sequences,
            ece_bins=int(runner.config["statistics"]["ece_bins"]),
        )
        metrics = compute_metrics(result.validation_predictions)
        row = {
            "candidate_key": key,
            **candidate,
            "validation_nll": metrics["nll"],
            "validation_auc": metrics["auc"],
            "best_epoch": result.best_epoch,
            "trainable_parameters": model.trainable_parameter_count(),
            "encoder_cache_key": encoder_metadata["cache_key"],
            "manifest_hash": manifest.manifest_hash,
        }
        rows.append(row)
        if best is None or float(metrics["nll"]) < best[0]:
            best = (float(metrics["nll"]), row)
    if best is None:
        raise RuntimeError("Hyperparameter search produced no candidate")
    audit = pd.DataFrame.from_records(rows).sort_values(
        ["validation_nll", "candidate_key"], kind="mergesort"
    )
    write_table(audit, output / "validation_search.csv")
    selected = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "selection_metric": "validation_nll",
        "selected": best[1],
        "test_predictions_generated": False,
    }
    write_yaml(output / "selected.yaml", selected)
    return selected
