"""Vocabulary expansion experiments separated into query, insertion, and refitting effects."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from semopkt.models.semopkt import SemOpKT


@torch.inference_mode()
def query_only_expansion_drift(
    model: SemOpKT,
    state: torch.Tensor,
    old_concept_indices: torch.Tensor,
    added_concept_indices: torch.Tensor,
) -> dict[str, float]:
    coordinates = model.concept_coordinates()
    before = torch.sigmoid(
        model.prediction_head(
            torch.cat(
                [model.field_query(state, coordinates[old_concept_indices][None, :, :]), coordinates[old_concept_indices][None, :, :]],
                dim=-1,
            )
            if model.direct_coordinate_path
            else model.field_query(state, coordinates[old_concept_indices][None, :, :])
        ).squeeze(-1)
    )
    # Querying added coordinates has no side effect; this call makes the interface test explicit.
    model.field_query(state, coordinates[added_concept_indices][None, :, :])
    after = torch.sigmoid(
        model.prediction_head(
            torch.cat(
                [model.field_query(state, coordinates[old_concept_indices][None, :, :]), coordinates[old_concept_indices][None, :, :]],
                dim=-1,
            )
            if model.direct_coordinate_path
            else model.field_query(state, coordinates[old_concept_indices][None, :, :])
        ).squeeze(-1)
    )
    drift = torch.abs(after - before).cpu().numpy().ravel()
    return {
        "mean_absolute_drift": float(np.mean(drift)),
        "median_absolute_drift": float(np.median(drift)),
        "p95_absolute_drift": float(np.quantile(drift, 0.95)),
        "maximum_absolute_drift": float(np.max(drift)),
    }


def retraining_expansion_summary(
    before: Mapping[str, float], after: Mapping[str, float], added: Mapping[str, float]
) -> dict[str, float]:
    return {
        "old_auc_before": float(before["auc"]),
        "old_auc_after": float(after["auc"]),
        "old_auc_change": float(after["auc"] - before["auc"]),
        "added_auc": float(added["auc"]),
        "old_nll_change": float(after["nll"] - before["nll"]),
        "added_nll": float(added["nll"]),
    }

