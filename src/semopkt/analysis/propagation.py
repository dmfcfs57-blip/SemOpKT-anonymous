"""Held-out compatible empirical association and SemOpKT influence analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from semopkt.data.sequences import ConceptVocabulary, StudentSequence
from semopkt.models.semopkt import SemOpKT


def empirical_association_matrix(
    frame: pd.DataFrame,
    vocabulary: ConceptVocabulary,
    maximum_lag: int = 5,
    minimum_pair_observations: int = 20,
) -> tuple[np.ndarray, pd.DataFrame]:
    concept_count = len(vocabulary.texts)
    text_to_index = vocabulary.text_to_index
    difficulty = frame.groupby("kc_text_norm")["correct"].mean().to_dict()
    observations: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    ordered = frame.sort_values(["student_id", "position", "source_row_id"], kind="mergesort")
    for _, group in ordered.groupby("student_id", sort=False):
        rows = list(group.itertuples(index=False))
        cumulative_correct = 0.0
        for source_position, source in enumerate(rows):
            prior_accuracy = cumulative_correct / max(1, source_position)
            source_index = text_to_index[str(source.kc_text_norm)]
            for lag in range(1, maximum_lag + 1):
                target_position = source_position + lag
                if target_position >= len(rows):
                    break
                target = rows[target_position]
                target_index = text_to_index[str(target.kc_text_norm)]
                observations[(source_index, target_index)].append(
                    [
                        float(source.correct),
                        prior_accuracy,
                        float(difficulty[str(target.kc_text_norm)]),
                        float(target.position) / 50.0,
                        float(lag) / maximum_lag,
                        float(target.correct),
                    ]
                )
            cumulative_correct += float(source.correct)
    matrix = np.full((concept_count, concept_count), np.nan, dtype=np.float64)
    audit_rows: list[dict[str, float | int]] = []
    for (source, target), rows in observations.items():
        values = np.asarray(rows, dtype=np.float64)
        if len(values) < minimum_pair_observations or np.unique(values[:, -1]).size < 2 or np.unique(values[:, 0]).size < 2:
            continue
        features = values[:, :-1]
        labels = values[:, -1].astype(int)
        model = LogisticRegression(C=1.0e4, solver="lbfgs", max_iter=500)
        model.fit(features, labels)
        coefficient = float(model.coef_[0, 0])
        response_zero = features.copy()
        response_one = features.copy()
        response_zero[:, 0] = 0.0
        response_one[:, 0] = 1.0
        adjusted_difference = float(
            np.mean(
                model.predict_proba(response_one)[:, 1]
                - model.predict_proba(response_zero)[:, 1]
            )
        )
        matrix[source, target] = adjusted_difference
        audit_rows.append(
            {
                "source_index": source,
                "target_index": target,
                "observations": int(len(values)),
                "coefficient": coefficient,
                "adjusted_probability_difference": adjusted_difference,
                "source_positive_rate": float(values[:, 0].mean()),
                "target_positive_rate": float(values[:, -1].mean()),
            }
        )
    return matrix, pd.DataFrame(audit_rows)


@torch.inference_mode()
def semopkt_effective_influence(
    model: SemOpKT,
    sequences: Sequence[StudentSequence],
    vocabulary: ConceptVocabulary,
    maximum_states: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    concept_count = len(vocabulary.texts)
    sums = np.zeros((concept_count, concept_count), dtype=np.float64)
    counts = np.zeros(concept_count, dtype=np.int64)
    coordinates = model.concept_coordinates()
    anchors = torch.nn.functional.normalize(model.anchors, dim=-1, eps=model.epsilon)
    processed = 0
    for sequence in sequences:
        state = model.population_prior[None, :, :].clone()
        for source_index, observed_response in zip(
            sequence.concept_indices, sequence.labels, strict=True
        ):
            source = int(source_index)
            coordinate = coordinates[source][None, :]
            counterfactual_predictions: list[torch.Tensor] = []
            for response in (0, 1):
                updated = state.clone()
                response_tensor = torch.tensor([response], dtype=torch.float32, device=device)
                for layer in model.update_layers:
                    current_field = model.field_query(updated, coordinate)
                    updated, _ = layer(
                        updated, coordinate, current_field, response_tensor, anchors
                    )
                target_coordinates = coordinates[None, :, :]
                target_field = model.field_query(updated, target_coordinates)
                prediction_input = (
                    torch.cat([target_field, target_coordinates], dim=-1)
                    if model.direct_coordinate_path
                    else target_field
                )
                probabilities = torch.sigmoid(model.prediction_head(prediction_input).squeeze(-1))
                counterfactual_predictions.append(probabilities[0])
            difference = (counterfactual_predictions[1] - counterfactual_predictions[0]).cpu().numpy()
            sums[source] += difference
            counts[source] += 1
            actual = torch.tensor([int(observed_response)], dtype=torch.float32, device=device)
            updated = state
            for layer in model.update_layers:
                current_field = model.field_query(updated, coordinate)
                updated, _ = layer(updated, coordinate, current_field, actual, anchors)
            state = updated
            processed += 1
            if maximum_states is not None and processed >= maximum_states:
                break
        if maximum_states is not None and processed >= maximum_states:
            break
    influence = np.divide(
        sums,
        counts[:, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, None] > 0,
    )
    return influence, counts


def replay_effective_influence(
    predict: Callable[[Sequence[StudentSequence]], pd.DataFrame],
    sequences: Sequence[StudentSequence],
    concept_count: int,
    maximum_states: int | None = None,
    maximum_sequence_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-agnostic response counterfactual influence by strict history replay."""

    sums = np.zeros((concept_count, concept_count), dtype=np.float64)
    counts = np.zeros(concept_count, dtype=np.int64)
    processed = 0
    for sequence in sequences:
        for source_step in range(len(sequence.labels)):
            if source_step + 2 > maximum_sequence_length:
                break
            source = int(sequence.concept_indices[source_step])
            counterfactual: dict[int, np.ndarray] = {}
            for response in (0, 1):
                queries: list[StudentSequence] = []
                for target in range(concept_count):
                    length = source_step + 2
                    labels = np.concatenate(
                        [
                            sequence.labels[:source_step],
                            np.asarray([response, 0.0], dtype=np.float32),
                        ]
                    )
                    concepts = np.concatenate(
                        [
                            sequence.concept_indices[: source_step + 1],
                            np.asarray([target], dtype=np.int64),
                        ]
                    )
                    identifier = f"cf-{processed}-{response}-{target}"
                    queries.append(
                        StudentSequence(
                            dataset=sequence.dataset,
                            student_id=identifier,
                            question_ids=[
                                *sequence.question_ids[: source_step + 1],
                                f"query-{target}",
                            ],
                            kc_ids=[
                                *sequence.kc_ids[: source_step + 1],
                                f"query-{target}",
                            ],
                            concept_indices=concepts,
                            labels=labels,
                            positions=np.arange(1, length + 1, dtype=np.int64),
                            source_row_ids=[
                                *sequence.source_row_ids[: source_step + 1],
                                identifier,
                            ],
                            score_mask=np.asarray(
                                [False] * (length - 1) + [True], dtype=bool
                            ),
                        )
                    )
                predictions = predict(queries)
                if len(predictions) != concept_count:
                    raise ValueError("Counterfactual replay lost target queries")
                probability = {
                    str(row.student_id): float(row.probability)
                    for row in predictions.itertuples(index=False)
                }
                counterfactual[response] = np.asarray(
                    [probability[f"cf-{processed}-{response}-{target}"] for target in range(concept_count)]
                )
            sums[source] += counterfactual[1] - counterfactual[0]
            counts[source] += 1
            processed += 1
            if maximum_states is not None and processed >= maximum_states:
                break
        if maximum_states is not None and processed >= maximum_states:
            break
    influence = np.divide(
        sums,
        counts[:, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, None] > 0,
    )
    return influence, counts


def _ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="mergesort")[:k]
    ideal = np.argsort(-relevance, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[order] * discounts[: len(order)]))
    idcg = float(np.sum(relevance[ideal] * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else float("nan")


def propagation_alignment(
    empirical: np.ndarray,
    influence: np.ndarray,
    top_k: int = 5,
) -> dict[str, float | int]:
    valid = np.isfinite(empirical) & np.isfinite(influence)
    np.fill_diagonal(valid, False)
    if np.sum(valid) < 3:
        raise ValueError("Too few valid concept pairs for propagation alignment")
    empirical_abs = np.abs(empirical)
    influence_abs = np.abs(influence)
    correlation = spearmanr(empirical_abs[valid], influence_abs[valid]).statistic
    ndcg_values: list[float] = []
    overlap_values: list[float] = []
    for source in range(empirical.shape[0]):
        row_valid = valid[source]
        if np.sum(row_valid) < top_k:
            continue
        candidate = np.flatnonzero(row_valid)
        relevance = empirical_abs[source, candidate]
        scores = influence_abs[source, candidate]
        ndcg_values.append(_ndcg_at_k(relevance, scores, top_k))
        empirical_top = set(candidate[np.argsort(-relevance)[:top_k]].tolist())
        influence_top = set(candidate[np.argsort(-scores)[:top_k]].tolist())
        overlap_values.append(len(empirical_top & influence_top) / top_k)
    empirical_threshold = np.nanquantile(empirical_abs[valid], 0.75)
    related = valid & (empirical_abs >= empirical_threshold)
    random_like = valid & (empirical_abs <= np.nanquantile(empirical_abs[valid], 0.25))
    ratio = float(np.nanmean(influence_abs[related]) / np.nanmean(influence_abs[random_like]))
    sign = float(np.mean(np.sign(empirical[valid]) == np.sign(influence[valid])))
    return {
        "spearman": float(correlation),
        "ndcg_at_5": float(np.nanmean(ndcg_values)),
        "top5_overlap": float(np.nanmean(overlap_values)),
        "related_random_ratio": ratio,
        "sign_agreement": sign,
        "valid_pairs": int(np.sum(valid)),
        "source_rows": int(len(ndcg_values)),
    }


def cross_fitted_propagation(
    frame: pd.DataFrame,
    fold_assignments: Mapping[str, int],
    fit_influence: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    vocabulary: ConceptVocabulary,
    folds: int,
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    student_fold = frame["student_id"].astype(str).map(fold_assignments)
    if student_fold.isna().any():
        raise ValueError("Every student requires a cross-fitting fold assignment")
    for fold in range(folds):
        train = frame[student_fold != fold]
        heldout = frame[student_fold == fold]
        influence = fit_influence(train, heldout)
        empirical, _ = empirical_association_matrix(heldout, vocabulary)
        results.append({"fold": fold, **propagation_alignment(empirical, influence)})
    return results
