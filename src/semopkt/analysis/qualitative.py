"""Rule-based case selection that includes representative failures."""

from __future__ import annotations

import json
from typing import Sequence
import numpy as np
import pandas as pd
import torch

from semopkt.data.sequences import ConceptVocabulary, StudentSequence
from semopkt.models.semopkt import SemOpKT


@torch.inference_mode()
def trace_semopkt_targets(
    model: SemOpKT,
    sequences: list[StudentSequence],
    vocabulary: ConceptVocabulary,
) -> pd.DataFrame:
    """Record field changes at every pre-registered scored target."""

    model.eval()
    device = next(model.parameters()).device
    coordinates = model.concept_coordinates()
    anchors = torch.nn.functional.normalize(model.anchors, dim=-1, eps=model.epsilon)

    def probabilities(state: torch.Tensor) -> torch.Tensor:
        query = coordinates[None, :, :]
        field = model.field_query(state, query)
        inputs = torch.cat([field, query], dim=-1) if model.direct_coordinate_path else field
        return torch.sigmoid(model.prediction_head(inputs).squeeze(-1))[0]

    rows: list[dict[str, object]] = []
    for sequence in sequences:
        state = model.population_prior[None, :, :].clone()
        for step, (concept_value, label_value) in enumerate(
            zip(sequence.concept_indices, sequence.labels, strict=True)
        ):
            concept = int(concept_value)
            before = probabilities(state)
            coordinate = coordinates[concept][None, :]
            updated = state
            response = torch.tensor(
                [int(label_value)], dtype=torch.float32, device=device
            )
            for layer in model.update_layers:
                current_field = model.field_query(updated, coordinate)
                updated, _ = layer(
                    updated, coordinate, current_field, response, anchors
                )
            after = probabilities(updated)
            if bool(sequence.score_mask[step]):
                change = after - before
                similarity = coordinates @ coordinates[concept]
                nearest = torch.argsort(similarity, descending=True)
                nearest = nearest[nearest != concept][:5]
                far_count = max(1, int(np.ceil(len(coordinates) / 4)))
                farthest = torch.argsort(similarity)[:far_count]
                propagation = torch.argsort(torch.abs(change), descending=True)[:5]
                rows.append(
                    {
                        "dataset": sequence.dataset,
                        "student_id": sequence.student_id,
                        "source_row_id": sequence.source_row_ids[step],
                        "position": int(sequence.positions[step]),
                        "kc_id": sequence.kc_ids[step],
                        "correct": int(label_value),
                        "probability_before": float(before[concept].cpu()),
                        "probability_after": float(after[concept].cpu()),
                        "current_concept_change": float(change[concept].cpu()),
                        "related_mean_absolute_change": float(
                            torch.abs(change[nearest]).mean().cpu()
                        ),
                        "unrelated_mean_absolute_change": float(
                            torch.abs(change[farthest]).mean().cpu()
                        ),
                        "top5_propagation": json.dumps(
                            [
                                {
                                    "concept": vocabulary.texts[int(index)],
                                    "change": float(change[int(index)].cpu()),
                                }
                                for index in propagation
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "subsequent_label": int(sequence.labels[step + 1])
                        if step + 1 < len(sequence.labels)
                        else None,
                    }
                )
            state = updated
    return pd.DataFrame.from_records(rows)


def select_cases(
    semopkt: pd.DataFrame,
    baseline: pd.DataFrame,
    seed: int,
    rules: Sequence[str] | None = None,
) -> pd.DataFrame:
    keys = [
        "dataset",
        "student_id",
        "source_row_id",
        "question_id",
        "kc_id",
        "position",
        "correct",
    ]
    merged = semopkt[keys + ["probability", "seen_status"]].rename(
        columns={"probability": "semopkt_probability"}
    ).merge(
        baseline[keys + ["probability"]].rename(columns={"probability": "baseline_probability"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No paired predictions for qualitative selection")
    labels = merged["correct"].to_numpy(dtype=float)
    sem_error = np.abs(labels - merged["semopkt_probability"].to_numpy())
    baseline_error = np.abs(labels - merged["baseline_probability"].to_numpy())
    merged["error_gain"] = baseline_error - sem_error
    rng = np.random.default_rng(seed)
    student_gain = merged.groupby(
        ["dataset", "student_id"], sort=True
    )["error_gain"].mean()
    student_keys = list(student_gain.index)
    random_student = student_keys[int(rng.integers(len(student_keys)))]
    random_candidates = merged[
        (merged["dataset"] == random_student[0])
        & (merged["student_id"] == random_student[1])
    ]
    random_index = int(rng.choice(random_candidates.index.to_numpy()))
    median_gain = float(student_gain.median())
    median_student = (student_gain - median_gain).abs().idxmin()
    median_candidates = merged[
        (merged["dataset"] == median_student[0])
        & (merged["student_id"] == median_student[1])
    ]
    median_index = int(
        (median_candidates["error_gain"] - student_gain.loc[median_student])
        .abs()
        .idxmin()
    )
    failure_student = student_gain.idxmin()
    failure_candidates = merged[
        (merged["dataset"] == failure_student[0])
        & (merged["student_id"] == failure_student[1])
    ]
    failure_index = int(failure_candidates["error_gain"].idxmin())
    unseen = merged[merged["seen_status"] == "unseen"]
    unseen_index = int(unseen.index[0]) if not unseen.empty else failure_index
    double = merged[(merged["seen_status"] == "unseen") & (merged["position"] <= 5)]
    double_index = int(double.index[0]) if not double.empty else unseen_index
    selections = [
        ("random", random_index),
        ("median_gain", median_index),
        ("failure", failure_index),
        ("unseen_concept", unseen_index),
        ("double_cold_start", double_index),
    ]
    selected_rules = set(
        rules
        if rules is not None
        else [rule for rule, _ in selections]
    )
    unknown = sorted(selected_rules - {rule for rule, _ in selections})
    if unknown:
        raise ValueError(f"Unknown qualitative selection rules: {unknown}")
    selections = [item for item in selections if item[0] in selected_rules]
    rows = []
    for rule, index in selections:
        row = merged.loc[index].to_dict()
        row["selection_rule"] = rule
        rows.append(row)
    return pd.DataFrame(rows)
