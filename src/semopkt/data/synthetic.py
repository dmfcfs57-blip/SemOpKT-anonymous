"""Deterministic synthetic interactions used only for software validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from semopkt.data.schema import normalize_text, stable_identifier
from semopkt.utils.io import write_table


def generate_synthetic_interactions(
    output: str | Path,
    students: int = 48,
    concepts: int = 12,
    sequence_length: int = 20,
    seed: int = 314159,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    concept_vectors = rng.normal(size=(concepts, 4))
    concept_vectors /= np.linalg.norm(concept_vectors, axis=1, keepdims=True)
    difficulties = rng.normal(0.0, 0.8, size=concepts)
    records: list[dict[str, object]] = []
    for student in range(students):
        ability = rng.normal()
        mastery = np.zeros(concepts, dtype=float)
        for position in range(1, sequence_length + 1):
            concept = int(rng.integers(concepts))
            related = concept_vectors @ concept_vectors[concept]
            logit = ability + mastery[concept] - difficulties[concept]
            probability = 1.0 / (1.0 + np.exp(-logit))
            correct = int(rng.random() < probability)
            text = normalize_text(f"synthetic concept {concept:02d}")
            records.append(
                {
                    "dataset": "Synthetic",
                    "student_id": f"student_{student:04d}",
                    "question_id": f"question_{concept:02d}_{position:02d}",
                    "kc_id": stable_identifier(text),
                    "kc_text_raw": f"Synthetic concept {concept:02d}",
                    "kc_text_norm": text,
                    "kc_components": json.dumps([text], separators=(",", ":")),
                    "correct": correct,
                    "timestamp": str(position),
                    "position": position,
                    "source_row_id": f"student_{student:04d}_row_{position:02d}",
                }
            )
            signed = 1.0 if correct else -0.5
            mastery += 0.08 * signed * np.maximum(related, 0.0)
    frame = pd.DataFrame.from_records(records)
    write_table(frame, output)
    return frame

