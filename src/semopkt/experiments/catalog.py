"""Complete task expansion for E0--E20 without hard-coded results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from semopkt.experiments.specs import RunSpec

SYSTEM_MODELS = (
    "BKT",
    "DKT",
    "DKVMN",
    "AKT",
    "simpleKT",
    "GKT",
    "csKT",
    "UKT",
    "SINKT",
    "EAKT",
    "MAHKT",
    "CLST",
    "SemOpKT",
)
OPEN_MODELS = ("DKT", "AKT", "GKT", "SINKT", "EAKT", "MAHKT", "CLST", "SemOpKT")
EARLY_MODELS = ("DKT", "DKVMN", "AKT", "simpleKT", "csKT", "MAML-KT", "CLST", "SemOpKT")
TRANSFER_MODELS = ("SINKT", "EAKT", "CLST", "DisenKT", "SemOpKT")


def _datasets(config: Mapping[str, Any]) -> list[str]:
    return sorted(config["datasets"].keys())


def _seeds(config: Mapping[str, Any]) -> list[int]:
    return [int(seed) for seed in config["seeds"]]


def _sizes(config: Mapping[str, Any]) -> list[int]:
    return [int(size) for size in config["train_sizes"]]


def _select_models(config: Mapping[str, Any], candidates: Sequence[str]) -> tuple[str, ...]:
    declared = set(config.get("models", candidates))
    return tuple(
        model
        for model in candidates
        if model in declared or (model.startswith("A") and model[1:].isdigit() and "SemOpKT" in declared)
    )


def _product_specs(
    experiment: str,
    datasets: Sequence[str],
    protocols: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    sizes: Sequence[int | None],
    ratios: Sequence[float | None] = (None,),
    condition: str = "default",
    overrides: Mapping[str, Any] | None = None,
) -> list[RunSpec]:
    return [
        RunSpec(
            experiment,
            dataset,
            protocol,
            model,
            seed,
            train_size=size,
            holdout_ratio=ratio,
            condition=condition,
            overrides=dict(overrides or {}),
        )
        for dataset in datasets
        for protocol in protocols
        for model in models
        for seed in seeds
        for size in sizes
        for ratio in ratios
    ]


def build_experiment_specs(config: Mapping[str, Any], experiment: str) -> list[RunSpec]:
    datasets = _datasets(config)
    seeds = _seeds(config)
    sizes = _sizes(config)
    if not sizes:
        raise ValueError("At least one training size is required")
    smallest_size = min(sizes)
    largest_size = max(sizes)
    ratios = [float(value) for value in config["holdout_ratios"]]
    if experiment == "E0":
        return _product_specs(experiment, datasets, ["system"], _select_models(config, ["BKT", "DKT", "DKVMN", "AKT", "CLST"]), seeds, sizes)
    if experiment == "E1":
        return _product_specs(experiment, datasets, ["system"], _select_models(config, SYSTEM_MODELS), seeds, sizes)
    if experiment == "E2":
        return _product_specs(
            experiment,
            datasets,
            ["full_data"],
            _select_models(config, ["DKT", "DKVMN", "AKT", "simpleKT", "csKT", "SINKT", "EAKT", "CLST", "SemOpKT"]),
            seeds,
            [None],
        )
    if experiment == "E3":
        return _product_specs(experiment, datasets, ["early_history"], _select_models(config, EARLY_MODELS), seeds, sizes)
    if experiment == "E4":
        return _product_specs(experiment, datasets, ["unseen_random"], _select_models(config, OPEN_MODELS), seeds, [largest_size], ratios)
    if experiment == "E5":
        return _product_specs(experiment, datasets, ["unseen_cluster"], _select_models(config, OPEN_MODELS), seeds, [largest_size], ratios)
    if experiment == "E6":
        models = _select_models(config, OPEN_MODELS)
        specs = _product_specs(experiment, datasets, ["double_random"], models, seeds, [largest_size], [0.20])
        specs.extend(_product_specs(experiment, datasets, ["double_cluster"], models, seeds, [largest_size], [0.20]))
        return specs
    if experiment == "E7":
        return _product_specs(
            experiment,
            datasets,
            ["online_adaptation"],
            _select_models(config, ["SINKT", "EAKT", "MAHKT", "CLST", "SemOpKT"]),
            seeds,
            [largest_size],
            [0.20],
        )
    if experiment == "E8":
        specs: list[RunSpec] = []
        transfer_models = _select_models(config, TRANSFER_MODELS)
        for target in datasets:
            for model in transfer_models:
                for seed in seeds:
                    for adaptation in sizes:
                        specs.append(
                            RunSpec(
                                experiment,
                                target,
                                "target_only",
                                model,
                                seed,
                                train_size=adaptation,
                                target_dataset=target,
                                adaptation_size=adaptation,
                            )
                        )
        for source, target in config.get("cross_domain_pairs", []):
            for model in transfer_models:
                for seed in seeds:
                    for condition in ("overlap_retained", "near_duplicates_removed"):
                        specs.append(
                            RunSpec(
                                experiment,
                                target,
                                "source_only_online",
                                model,
                                seed,
                                source_dataset=source,
                                target_dataset=target,
                                condition=condition,
                            )
                        )
                        for adaptation in sizes:
                            specs.append(
                                RunSpec(
                                    experiment,
                                    target,
                                    "source_target_adaptation",
                                    model,
                                    seed,
                                    source_dataset=source,
                                    target_dataset=target,
                                    adaptation_size=adaptation,
                                    condition=condition,
                                )
                            )
        for target in datasets:
            sources = "+".join(dataset for dataset in datasets if dataset != target)
            if not sources:
                continue
            for model in transfer_models:
                for seed in seeds:
                    for condition in ("overlap_retained", "near_duplicates_removed"):
                        for adaptation in sizes:
                            specs.append(
                                RunSpec(
                                    experiment,
                                    target,
                                    "multi_source_target_adaptation",
                                    model,
                                    seed,
                                    source_dataset=sources,
                                    target_dataset=target,
                                    adaptation_size=adaptation,
                                    condition=condition,
                                )
                            )
        return specs
    if experiment == "E9":
        variants = _select_models(config, tuple(f"A{index}" for index in range(15)))
        boundary_sizes = sorted({smallest_size, largest_size})
        specs = _product_specs(experiment, datasets, ["system"], variants, seeds, boundary_sizes)
        specs.extend(_product_specs(experiment, datasets, ["unseen_cluster"], variants, seeds, [largest_size], [0.20]))
        specs.extend(_product_specs(experiment, datasets, ["double_cluster"], variants, seeds, [largest_size], [0.20]))
        return specs
    if experiment == "E10":
        controls = (
            "original",
            "paraphrase",
            "normalized",
            "delete_20_percent",
            "masked",
            "shuffled_names",
            "random_vectors",
            "learnable_identifiers",
            "external_definitions",
        )
        specs: list[RunSpec] = []
        for control in controls:
            specs.extend(
                _product_specs(
                    experiment,
                    datasets,
                    ["semantic_system"],
                    ["SemOpKT"],
                    seeds,
                    [largest_size],
                    condition=control,
                )
            )
            for protocol in ("semantic_random", "semantic_cluster", "semantic_double"):
                specs.extend(
                    _product_specs(
                        experiment,
                        datasets,
                        [protocol],
                        ["SemOpKT"],
                        seeds,
                        [largest_size],
                        [0.20],
                        condition=control,
                    )
                )
        return specs
    if experiment == "E11":
        models = ["A8", "A5", "A6", "A7", "A9", "A10", "A0"]
        specs = _product_specs(
            experiment, datasets, ["operator_comparison"], models, seeds, [smallest_size]
        )
        specs.extend(
            _product_specs(
                experiment, datasets, ["operator_cluster"], models, seeds, [smallest_size], [0.20]
            )
        )
        specs.extend(
            _product_specs(
                experiment, datasets, ["operator_double"], models, seeds, [smallest_size], [0.20]
            )
        )
        return specs
    if experiment == "E12":
        specs: list[RunSpec] = []
        for visible in (0.25, 0.50, 0.75, 1.0):
            protocol = "vocabulary_expansion" if visible < 1.0 else "vocabulary_full"
            specs.extend(
                _product_specs(
                    experiment,
                    datasets,
                    [protocol],
                    _select_models(config, ["GKT", "SINKT", "MAHKT", "SemOpKT"]),
                    seeds,
                    [largest_size],
                    [1.0 - visible if visible < 1.0 else None],
                    condition=f"visible-{int(visible * 100)}",
                )
            )
        for inducing in config["inducing_sizes"]:
            specs.extend(
                _product_specs(
                    experiment,
                    datasets,
                    ["inducing_sensitivity"],
                    ["SemOpKT"],
                    seeds,
                    [smallest_size],
                    condition=f"M-{inducing}",
                    overrides={"architecture": {"inducing_points": int(inducing)}},
                )
            )
        return specs
    if experiment == "E13":
        return _product_specs(
            experiment,
            datasets,
            ["propagation"],
            _select_models(config, ["GKT", "MAHKT", "A9", "A10", "SemOpKT"]),
            seeds,
            [largest_size],
        )
    if experiment == "E14":
        perturbations = (
            "history_delete_05",
            "history_delete_10",
            "history_delete_20",
            "label_flip_02",
            "label_flip_05",
            "label_flip_10",
            "adjacent_swap_10",
            "adjacent_swap_20",
            "unrelated_insert_1",
            "unrelated_insert_2",
            "unrelated_insert_4",
            "name_delete_10",
            "name_delete_20",
            "name_delete_30",
            "synonym_replace_10",
            "synonym_replace_20",
        )
        specs: list[RunSpec] = []
        for condition in perturbations:
            specs.extend(
                _product_specs(
                    experiment,
                    datasets,
                    ["robustness"],
                    _select_models(config, ["AKT", "UKT", "SINKT", "CLST", "SemOpKT", "A13"]),
                    seeds,
                    [largest_size],
                    condition=condition,
                )
            )
        return specs
    if experiment == "E15":
        models = _select_models(config, ["AKT", "csKT", "UKT", "CLST", "SemOpKT"])
        specs = _product_specs(experiment, datasets, ["calibration_system"], models, seeds, sizes)
        specs.extend(_product_specs(experiment, datasets, ["calibration_cluster"], models, seeds, [largest_size], [0.20]))
        specs.extend(_product_specs(experiment, datasets, ["calibration_double"], models, seeds, [largest_size], [0.20]))
        return specs
    if experiment == "E16":
        return _product_specs(experiment, datasets, ["data_efficiency"], _select_models(config, ["simpleKT", "csKT", "SINKT", "CLST", "SemOpKT"]), seeds, [int(value) for value in config["data_efficiency_sizes"]])
    if experiment == "E17":
        specs: list[RunSpec] = []
        factors = {
            "rank": [8, 16, 32, 64],
            "inducing_points": [16, 32, 64, 128],
            "layers": [1, 2, 3, 4],
            "interpolation_temperature": [0.05, 0.1, 0.2, 0.5],
        }
        screening_datasets = [
            dataset for dataset in ("NIPS34", "Algebra05") if dataset in datasets
        ] or datasets
        for key, values in factors.items():
            for value in values:
                architecture_override = {key: value}
                if key == "interpolation_temperature":
                    architecture_override["learnable_temperature"] = False
                specs.extend(
                    _product_specs(
                        experiment,
                        screening_datasets,
                        ["sensitivity"],
                        ["SemOpKT"],
                        seeds,
                        [smallest_size],
                        condition=f"{key}-{value}",
                        overrides={"architecture": architecture_override},
                    )
                )
        for key in ("smoothness", "stability"):
            for value in (0.0, 1.0e-4, 1.0e-3, 1.0e-2):
                specs.extend(
                    _product_specs(
                        experiment,
                        screening_datasets,
                        ["sensitivity"],
                        ["SemOpKT"],
                        seeds,
                        [smallest_size],
                        condition=f"{key}-{value:g}",
                        overrides={"regularization": {key: value}},
                    )
                )
        encoders = (
            (
                "minilm-l6-frozen",
                "sentence-transformers/all-MiniLM-L6-v2",
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                384,
                "frozen",
            ),
            (
                "minilm-l12-frozen",
                "sentence-transformers/all-MiniLM-L12-v2",
                "a50ef00143b4d5391434df20ae11632588ac25be",
                384,
                "frozen",
            ),
            (
                "mpnet-frozen",
                "sentence-transformers/all-mpnet-base-v2",
                "e8c3b32edf5434bc2275fc9bab85f82640a19130",
                768,
                "frozen",
            ),
            (
                "mpnet-partial",
                "sentence-transformers/all-mpnet-base-v2",
                "e8c3b32edf5434bc2275fc9bab85f82640a19130",
                768,
                "partial",
            ),
            (
                "mpnet-full",
                "sentence-transformers/all-mpnet-base-v2",
                "e8c3b32edf5434bc2275fc9bab85f82640a19130",
                768,
                "full",
            ),
        )
        for condition, model_id, revision, dimension, finetuning in encoders:
            specs.extend(
                _product_specs(
                    experiment,
                    screening_datasets,
                    ["sensitivity"],
                    ["SemOpKT"],
                    seeds,
                    [smallest_size],
                    condition=condition,
                    overrides={
                        "text_encoder": {
                            "model_id": model_id,
                            "revision": revision,
                            "finetuning": finetuning,
                            "partial_unfrozen_layers": 2,
                        },
                        "dimensions": {"text": dimension},
                    },
                )
            )
        screening_reference = screening_datasets[0]
        confirmation = [
            RunSpec(
                experiment,
                dataset,
                "double_cluster",
                original.model,
                original.seed,
                train_size=smallest_size,
                holdout_ratio=0.20,
                condition=original.condition,
                overrides=original.overrides,
            )
            for original in specs
            if original.dataset == screening_reference
            for dataset in datasets
        ]
        specs.extend(confirmation)
        return specs
    if experiment == "E18":
        return _product_specs(experiment, datasets, ["efficiency"], _select_models(config, ["AKT", "simpleKT", "GKT", "SINKT", "MAHKT", "CLST", "SemOpKT"]), seeds, [largest_size])
    if experiment == "E19":
        return _product_specs(experiment, datasets, ["error_analysis"], _select_models(config, ["SINKT", "MAHKT", "CLST", "SemOpKT"]), seeds, [largest_size], [0.20])
    if experiment == "E20":
        models = _select_models(config, ["CLST", "SemOpKT"])
        specs = _product_specs(
            experiment,
            datasets,
            ["qualitative_system"],
            models,
            [seeds[0]],
            [largest_size],
        )
        specs.extend(
            _product_specs(
                experiment,
                datasets,
                ["qualitative_unseen"],
                models,
                [seeds[0]],
                [largest_size],
                [0.20],
            )
        )
        specs.extend(
            _product_specs(
                experiment,
                datasets,
                ["qualitative_double"],
                models,
                [seeds[0]],
                [largest_size],
                [0.20],
            )
        )
        return specs
    raise KeyError(f"Unknown experiment identifier: {experiment}")
