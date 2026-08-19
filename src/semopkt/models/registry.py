"""Model and ablation registry with deterministic coordinate controls."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from semopkt.models.baselines import (
    BKTModel,
    DiscreteStateKT,
    DKVMNModel,
    DisentangledKT,
    GraphKTModel,
    MetaRecurrentKT,
    RecurrentKT,
    SemanticAttributeKT,
    TransformerKT,
)
from semopkt.models.base import KTModel
from semopkt.models.semopkt import SemOpKT

_BASELINES = (
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
    "MAML-KT",
    "DisenKT",
)
_ABLATIONS = tuple(f"A{index}" for index in range(15))


def model_names() -> tuple[str, ...]:
    return (*_BASELINES, "SemOpKT", *_ABLATIONS)


def _baseline_settings(name: str, configuration: Mapping[str, Any]) -> dict[str, Any]:
    models = configuration.get("models", configuration)
    if name not in models:
        raise KeyError(f"No baseline configuration for {name}")
    return dict(models[name])


def _coordinate_control(embeddings: np.ndarray, mode: str, seed: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if mode == "semantic":
        return matrix
    if mode == "shuffled":
        return matrix[rng.permutation(len(matrix))]
    if mode in {"random", "identifier"}:
        random = rng.normal(size=matrix.shape).astype(np.float32)
        random /= np.clip(np.linalg.norm(random, axis=1, keepdims=True), 1.0e-8, None)
        return random
    raise ValueError(f"Unknown coordinate mode: {mode}")


def _semopkt_variant(
    name: str,
    config: Mapping[str, Any],
    concept_embeddings: np.ndarray,
    seed: int,
    concept_texts: Sequence[str] | None = None,
) -> KTModel:
    resolved = copy.deepcopy(dict(config))
    architecture = resolved["architecture"]
    regularization = resolved["regularization"]
    mode = "semantic"
    if name == "A1":
        mode = "identifier"
        architecture["trainable_concept_embeddings"] = True
    elif name == "A2":
        mode = "random"
    elif name == "A3":
        mode = "shuffled"
    elif name == "A8":
        architecture["update_mode"] = "pointwise"
    elif name == "A9":
        architecture["update_mode"] = "fixed_rbf"
    elif name == "A10":
        architecture["update_mode"] = "linear"
    elif name == "A11":
        architecture["response_conditioning"] = False
    elif name == "A12":
        regularization["smoothness"] = 0.0
    elif name == "A13":
        regularization["stability"] = 0.0
    elif name == "A14":
        resolved["text_encoder"]["finetuning"] = "full"
    transformed = _coordinate_control(concept_embeddings, mode, seed)
    return SemOpKT(resolved, transformed, concept_texts=concept_texts)


def _parameter_matched_model(
    builder: Any,
    target_parameters: int,
    minimum: int,
    maximum: int,
    step: int = 1,
) -> KTModel:
    """Find the closest monotone capacity setting and retain only the final model."""

    low = int(math.ceil(minimum / step))
    high = int(math.floor(maximum / step))
    evaluated: dict[int, int] = {}
    while low <= high:
        middle = (low + high) // 2
        value = middle * step
        candidate = builder(value)
        count = candidate.trainable_parameter_count()
        evaluated[value] = count
        if count < target_parameters:
            low = middle + 1
        elif count > target_parameters:
            high = middle - 1
        else:
            break
    for middle in (low - 1, low, high, high + 1):
        value = middle * step
        if value < minimum or value > maximum or value in evaluated:
            continue
        candidate = builder(value)
        evaluated[value] = candidate.trainable_parameter_count()
    selected = min(
        evaluated,
        key=lambda value: (abs(evaluated[value] - target_parameters), value),
    )
    model = builder(selected)
    model.parameter_budget_target = int(target_parameters)  # type: ignore[attr-defined]
    model.parameter_budget_value = int(selected)  # type: ignore[attr-defined]
    model.parameter_budget_relative_error = float(  # type: ignore[attr-defined]
        abs(model.trainable_parameter_count() - target_parameters)
        / max(1, target_parameters)
    )
    return model


def build_model(
    name: str,
    semopkt_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    concept_embeddings: np.ndarray,
    seed: int,
    concept_texts: Sequence[str] | None = None,
) -> KTModel:
    if name in {"SemOpKT", "A0", "A1", "A2", "A3", "A8", "A9", "A10", "A11", "A12", "A13", "A14"}:
        return _semopkt_variant(
            name, semopkt_config, concept_embeddings, seed, concept_texts=concept_texts
        )
    if name in {"A4", "A5", "A6", "A7"}:
        budget = SemOpKT(semopkt_config, concept_embeddings).trainable_parameter_count()
        if name == "A4":
            state_size = 32 if budget < 500_000 else 256
            return _parameter_matched_model(
                lambda capacity: DiscreteStateKT(
                    len(concept_embeddings),
                    state_size=state_size,
                    capacity_size=capacity,
                    dropout=0.2,
                ),
                budget,
                minimum=8,
                maximum=2500,
                step=8,
            )
        if name == "A5":
            return _parameter_matched_model(
                lambda hidden: RecurrentKT(
                    "A5-GRU",
                    concept_embeddings,
                    hidden,
                    1,
                    0.2,
                    semantic_input=True,
                ),
                budget,
                minimum=8,
                maximum=1024,
                step=1,
            )
        if name == "A6":
            transformer_heads = 1 if budget < 500_000 else 8
            return _parameter_matched_model(
                lambda hidden: TransformerKT(
                    "A6-Transformer",
                    concept_embeddings,
                    hidden,
                    3,
                    transformer_heads,
                    0.2,
                    True,
                ),
                budget,
                minimum=8,
                maximum=768,
                step=1 if transformer_heads == 1 else 8,
            )
        graph_hidden = 32 if budget < 500_000 else 256
        return _parameter_matched_model(
            lambda capacity: GraphKTModel(
                concept_embeddings,
                hidden_size=graph_hidden,
                graph_neighbors=5,
                dropout=0.2,
                capacity_size=capacity,
            ),
            budget,
            minimum=8,
            maximum=2500,
            step=8,
        )
    settings = _baseline_settings(name, baseline_config)
    hidden = int(settings.get("hidden_size", 128))
    dropout = float(settings.get("dropout", 0.2))
    if name == "BKT":
        return BKTModel(len(concept_embeddings))
    if name == "DKT":
        return RecurrentKT(name, concept_embeddings, hidden, int(settings.get("layers", 1)), dropout)
    if name == "DKVMN":
        return DKVMNModel(
            concept_embeddings,
            hidden,
            int(settings.get("memory_slots", 64)),
            int(settings.get("memory_value_size", hidden)),
            dropout,
        )
    if name in {"AKT", "simpleKT", "csKT"}:
        return TransformerKT(
            name,
            concept_embeddings,
            hidden,
            int(settings.get("layers", 2)),
            int(settings.get("heads", 8)),
            dropout,
            semantic_input=False,
            distance_bias=str(settings.get("distance_bias", "none")),
        )
    if name == "GKT":
        return GraphKTModel(
            concept_embeddings, hidden, int(settings.get("graph_neighbors", 5)), dropout
        )
    if name == "UKT":
        return RecurrentKT(name, concept_embeddings, hidden, 1, dropout, uncertainty=True)
    if name == "SINKT":
        return RecurrentKT(name, concept_embeddings, hidden, 1, dropout, semantic_input=True)
    if name in {"EAKT", "MAHKT"}:
        return SemanticAttributeKT(
            name,
            concept_embeddings,
            hidden,
            dropout,
            relation_heads=int(settings.get("relation_heads", 1)),
        )
    if name == "CLST":
        return TransformerKT(
            name,
            concept_embeddings,
            hidden,
            int(settings.get("layers", 4)),
            int(settings.get("heads", 8)),
            dropout,
            semantic_input=True,
        )
    if name == "MAML-KT":
        return MetaRecurrentKT(
            concept_embeddings,
            hidden_size=hidden,
            layers=1,
            dropout=dropout,
            inner_steps=int(settings.get("inner_steps", 3)),
            inner_learning_rate=float(settings.get("inner_learning_rate", 1.0e-2)),
        )
    if name == "DisenKT":
        return DisentangledKT(concept_embeddings, hidden, dropout)
    raise KeyError(f"Unknown model: {name}")
