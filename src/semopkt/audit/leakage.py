"""Data-split and strict-online causality checks."""

from __future__ import annotations

from typing import Mapping

import pandas as pd
import torch

from semopkt.data.schema import interaction_contains_heldout
from semopkt.data.splits import SplitManifest, apply_manifest
from semopkt.models.base import KTModel


def audit_split_leakage(frame: pd.DataFrame, manifest: SplitManifest) -> dict[str, object]:
    split = apply_manifest(frame, manifest, train_size=None)
    train_students = set(split.train["student_id"].astype(str))
    validation_students = set(split.validation["student_id"].astype(str))
    test_students = set(split.test["student_id"].astype(str))
    heldout = set(manifest.heldout_kcs)
    train_heldout = int(
        split.train["kc_components"].map(
            lambda value: interaction_contains_heldout(value, heldout)
        ).sum()
    ) if heldout else 0
    validation_heldout = int(
        split.validation["kc_components"].map(
            lambda value: interaction_contains_heldout(value, heldout)
        ).sum()
    ) if heldout else 0
    user_overlap_allowed = bool(manifest.metadata.get("user_overlap_allowed", False))
    findings = {
        "train_validation_student_overlap": len(train_students & validation_students),
        "train_test_student_overlap": len(train_students & test_students),
        "validation_test_student_overlap": len(validation_students & test_students),
        "heldout_training_interactions": train_heldout,
        "heldout_validation_interactions": validation_heldout,
        "scored_test_interactions": int(split.test_target_mask.sum()),
        "user_overlap_allowed": user_overlap_allowed,
    }
    passed = (
        findings["train_validation_student_overlap"] == 0
        and train_heldout == 0
        and validation_heldout == 0
        and findings["scored_test_interactions"] > 0
        and (
            user_overlap_allowed
            or (
                findings["train_test_student_overlap"] == 0
                and findings["validation_test_student_overlap"] == 0
            )
        )
    )
    return {"passed": passed, **findings}


@torch.inference_mode()
def verify_strict_online_order(
    model: KTModel,
    batch: Mapping[str, object],
    tolerance: float = 1.0e-7,
) -> dict[str, object]:
    model.eval()
    original = model(batch).logits.detach().clone()  # type: ignore[arg-type]
    labels = batch["labels"]  # type: ignore[index]
    valid = batch["valid_mask"]  # type: ignore[index]
    if not isinstance(labels, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise TypeError("Batch tensors are required")
    maximum_current_difference = 0.0
    changed_future = False
    for step in range(labels.shape[1]):
        perturbed = dict(batch)
        changed_labels = labels.clone()
        changed_labels[:, step] = 1.0 - changed_labels[:, step]
        perturbed["labels"] = changed_labels
        altered = model(perturbed).logits.detach()  # type: ignore[arg-type]
        current = torch.abs(altered[:, step] - original[:, step])[valid[:, step]]
        if current.numel():
            maximum_current_difference = max(maximum_current_difference, float(current.max().cpu()))
        if step + 1 < labels.shape[1]:
            future = torch.abs(altered[:, step + 1 :] - original[:, step + 1 :])
            future_valid = valid[:, step + 1 :]
            changed_future = changed_future or bool(torch.any(future[future_valid] > tolerance))
    return {
        "passed": maximum_current_difference <= tolerance,
        "maximum_current_step_difference": maximum_current_difference,
        "future_predictions_can_change": changed_future,
        "tolerance": tolerance,
    }

