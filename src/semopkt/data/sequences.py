"""Padded student sequences with explicit scoring masks for strict online evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConceptVocabulary:
    texts: tuple[str, ...]
    text_to_index: Mapping[str, int]
    embeddings: np.ndarray

    @classmethod
    def build(
        cls, frame: pd.DataFrame, embedding_lookup: Mapping[str, np.ndarray]
    ) -> "ConceptVocabulary":
        texts = tuple(sorted(frame["kc_text_norm"].astype(str).unique().tolist()))
        missing = [text for text in texts if text not in embedding_lookup]
        if missing:
            raise KeyError(f"Missing embeddings for {len(missing)} descriptors")
        matrix = np.vstack([np.asarray(embedding_lookup[text], dtype=np.float32) for text in texts])
        return cls(texts, {text: index for index, text in enumerate(texts)}, matrix)


@dataclass
class StudentSequence:
    dataset: str
    student_id: str
    question_ids: list[str]
    kc_ids: list[str]
    concept_indices: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    source_row_ids: list[str]
    score_mask: np.ndarray


def remap_concept_indices(
    sequences: Sequence[StudentSequence], mapping: Mapping[int, int]
) -> list[StudentSequence]:
    """Copy sequences while replacing selected concept indices."""

    remapped: list[StudentSequence] = []
    for sequence in sequences:
        indices = sequence.concept_indices.copy()
        for source, target in mapping.items():
            indices[indices == int(source)] = int(target)
        remapped.append(
            StudentSequence(
                dataset=sequence.dataset,
                student_id=sequence.student_id,
                question_ids=list(sequence.question_ids),
                kc_ids=list(sequence.kc_ids),
                concept_indices=indices,
                labels=sequence.labels.copy(),
                positions=sequence.positions.copy(),
                source_row_ids=list(sequence.source_row_ids),
                score_mask=sequence.score_mask.copy(),
            )
        )
    return remapped


def build_sequences(
    frame: pd.DataFrame,
    vocabulary: ConceptVocabulary,
    target_mask: pd.Series | None = None,
    maximum_length: int = 50,
) -> list[StudentSequence]:
    sequences: list[StudentSequence] = []
    mask = (
        target_mask.reindex(frame.index, fill_value=False).astype(bool)
        if target_mask is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    ordered = frame.sort_values(["student_id", "position", "source_row_id"], kind="mergesort")
    for student_id, group in ordered.groupby("student_id", sort=False):
        group = group.head(maximum_length)
        sequences.append(
            StudentSequence(
                dataset=str(group["dataset"].iloc[0]),
                student_id=str(student_id),
                question_ids=group["question_id"].astype(str).tolist(),
                kc_ids=group["kc_id"].astype(str).tolist(),
                concept_indices=np.asarray(
                    [vocabulary.text_to_index[str(text)] for text in group["kc_text_norm"]],
                    dtype=np.int64,
                ),
                labels=group["correct"].to_numpy(dtype=np.float32),
                positions=group["position"].to_numpy(dtype=np.int64),
                source_row_ids=group["source_row_id"].astype(str).tolist(),
                score_mask=mask.loc[group.index].to_numpy(dtype=bool),
            )
        )
    return sequences


class TorchSequenceDataset:
    def __init__(self, sequences: Sequence[StudentSequence]):
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> StudentSequence:
        return self.sequences[index]


def collate_sequences(items: Sequence[StudentSequence]) -> dict[str, object]:
    import torch

    batch_size = len(items)
    maximum = max(len(item.labels) for item in items)
    concept_indices = torch.zeros((batch_size, maximum), dtype=torch.long)
    labels = torch.zeros((batch_size, maximum), dtype=torch.float32)
    positions = torch.zeros((batch_size, maximum), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    score_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item.labels)
        concept_indices[row, :length] = torch.from_numpy(item.concept_indices)
        labels[row, :length] = torch.from_numpy(item.labels.copy())
        positions[row, :length] = torch.from_numpy(item.positions.copy())
        valid_mask[row, :length] = True
        score_mask[row, :length] = torch.from_numpy(item.score_mask.copy())
    return {
        "dataset": [item.dataset for item in items],
        "student_id": [item.student_id for item in items],
        "question_ids": [item.question_ids for item in items],
        "kc_ids": [item.kc_ids for item in items],
        "source_row_ids": [item.source_row_ids for item in items],
        "concept_indices": concept_indices,
        "labels": labels,
        "positions": positions,
        "valid_mask": valid_mask,
        "score_mask": score_mask,
    }
