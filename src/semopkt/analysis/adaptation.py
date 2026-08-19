"""Online unseen-concept adaptation and old-concept stability analysis."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from semopkt.data.sequences import StudentSequence


def old_concept_stability_replay(
    sequences: Sequence[StudentSequence],
    heldout_indices: set[int],
    query_indices: Sequence[int],
    predict: Callable[[Sequence[StudentSequence]], pd.DataFrame],
    maximum_events: int = 1000,
    maximum_sequence_length: int = 50,
) -> pd.DataFrame:
    """Compare fixed old-concept queries immediately before and after an update."""

    rows: list[dict[str, float | int | str]] = []
    event = 0
    for sequence in sequences:
        prior_counts: dict[int, int] = {}
        for target in np.flatnonzero(sequence.score_mask):
            target_concept = int(sequence.concept_indices[target])
            if target_concept not in heldout_indices:
                continue
            if target + 2 > maximum_sequence_length:
                prior_counts[target_concept] = prior_counts.get(target_concept, 0) + 1
                continue
            before_sequences: list[StudentSequence] = []
            after_sequences: list[StudentSequence] = []
            for query in query_indices:
                before_length = target + 1
                after_length = target + 2
                before_id = f"stability-{event}-before-{query}"
                after_id = f"stability-{event}-after-{query}"
                before_sequences.append(
                    StudentSequence(
                        dataset=sequence.dataset,
                        student_id=before_id,
                        question_ids=[*sequence.question_ids[:target], f"query-{query}"],
                        kc_ids=[*sequence.kc_ids[:target], f"query-{query}"],
                        concept_indices=np.concatenate(
                            [
                                sequence.concept_indices[:target],
                                np.asarray([query], dtype=np.int64),
                            ]
                        ),
                        labels=np.concatenate(
                            [
                                sequence.labels[:target],
                                np.asarray([0.0], dtype=np.float32),
                            ]
                        ),
                        positions=np.arange(1, before_length + 1, dtype=np.int64),
                        source_row_ids=[
                            *sequence.source_row_ids[:target],
                            before_id,
                        ],
                        score_mask=np.asarray(
                            [False] * (before_length - 1) + [True], dtype=bool
                        ),
                    )
                )
                after_sequences.append(
                    StudentSequence(
                        dataset=sequence.dataset,
                        student_id=after_id,
                        question_ids=[
                            *sequence.question_ids[: target + 1],
                            f"query-{query}",
                        ],
                        kc_ids=[*sequence.kc_ids[: target + 1], f"query-{query}"],
                        concept_indices=np.concatenate(
                            [
                                sequence.concept_indices[: target + 1],
                                np.asarray([query], dtype=np.int64),
                            ]
                        ),
                        labels=np.concatenate(
                            [
                                sequence.labels[: target + 1],
                                np.asarray([0.0], dtype=np.float32),
                            ]
                        ),
                        positions=np.arange(1, after_length + 1, dtype=np.int64),
                        source_row_ids=[
                            *sequence.source_row_ids[: target + 1],
                            after_id,
                        ],
                        score_mask=np.asarray(
                            [False] * (after_length - 1) + [True], dtype=bool
                        ),
                    )
                )
            before = predict(before_sequences)
            after = predict(after_sequences)
            before_probability = dict(
                zip(before["kc_id"].astype(str), before["probability"], strict=True)
            )
            after_probability = dict(
                zip(after["kc_id"].astype(str), after["probability"], strict=True)
            )
            drift = [
                abs(after_probability[f"query-{query}"] - before_probability[f"query-{query}"])
                for query in query_indices
            ]
            rows.append(
                {
                    "dataset": sequence.dataset,
                    "student_id": sequence.student_id,
                    "source_row_id": sequence.source_row_ids[target],
                    "target_concept_index": target_concept,
                    "position": int(sequence.positions[target]),
                    "prior_target_observations": prior_counts.get(target_concept, 0),
                    "old_query_count": len(query_indices),
                    "old_concept_mean_absolute_drift": float(np.mean(drift)),
                }
            )
            prior_counts[target_concept] = prior_counts.get(target_concept, 0) + 1
            event += 1
            if event >= maximum_events:
                return pd.DataFrame.from_records(rows)
    return pd.DataFrame.from_records(rows)
