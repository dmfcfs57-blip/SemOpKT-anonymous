"""Target-preserving history perturbations for robustness experiment E14."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from semopkt.data.sequences import StudentSequence


def parse_perturbation_condition(condition: str) -> tuple[str, float | int]:
    prefixes: tuple[tuple[str, str, bool], ...] = (
        ("history_delete_", "history_delete", True),
        ("label_flip_", "label_flip", True),
        ("adjacent_swap_", "adjacent_swap", True),
        ("unrelated_insert_", "unrelated_insert", False),
        ("name_delete_", "name_delete", True),
        ("synonym_replace_", "synonym_replace", True),
    )
    for prefix, kind, percentage in prefixes:
        if condition.startswith(prefix):
            raw = int(condition[len(prefix) :])
            return kind, raw / 100.0 if percentage else raw
    raise ValueError(f"Unknown robustness condition: {condition}")


def delete_descriptor_tokens(
    texts: Sequence[str], fraction: float, seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    transformed: list[str] = []
    for text in texts:
        tokens = str(text).split()
        if not tokens:
            transformed.append("[missing concept]")
            continue
        count = max(1, min(len(tokens), int(round(len(tokens) * fraction))))
        removed = set(rng.choice(len(tokens), size=count, replace=False).tolist())
        retained = [token for index, token in enumerate(tokens) if index not in removed]
        transformed.append(" ".join(retained) if retained else "[missing concept]")
    return transformed


def replace_descriptor_synonyms(
    texts: Sequence[str], mapping: Mapping[str, str], fraction: float, seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    transformed: list[str] = []
    for text in texts:
        tokens = str(text).split()
        candidates = [index for index, token in enumerate(tokens) if token in mapping]
        if not candidates:
            transformed.append(str(text))
            continue
        count = max(1, min(len(candidates), int(round(len(tokens) * fraction))))
        selected = set(rng.choice(candidates, size=count, replace=False).tolist())
        transformed.append(
            " ".join(mapping[token] if index in selected else token for index, token in enumerate(tokens))
        )
    return transformed


def append_semantic_embeddings(model: torch.nn.Module, embeddings: np.ndarray) -> bool:
    """Append inference-only descriptor vectors to models that consume text."""

    matrix = torch.as_tensor(embeddings, dtype=torch.float32, device=next(model.parameters()).device)
    if hasattr(model, "concept_embeddings") and getattr(model, "descriptor_encoder", None) is None:
        current = getattr(model, "concept_embeddings")
        setattr(model, "concept_embeddings", torch.cat([current, matrix], dim=0))
        return True
    if bool(getattr(model, "semantic_input", False)) and hasattr(model, "text_embeddings"):
        current = getattr(model, "text_embeddings")
        setattr(model, "text_embeddings", torch.cat([current, matrix], dim=0))
        return True
    return False


def _subset(sequence: StudentSequence, indices: np.ndarray, target_original_index: int) -> StudentSequence:
    target_new = int(np.flatnonzero(indices == target_original_index)[0])
    score = np.zeros(len(indices), dtype=bool)
    score[target_new] = True
    return StudentSequence(
        dataset=sequence.dataset,
        student_id=sequence.student_id,
        question_ids=[sequence.question_ids[index] for index in indices],
        kc_ids=[sequence.kc_ids[index] for index in indices],
        concept_indices=sequence.concept_indices[indices].copy(),
        labels=sequence.labels[indices].copy(),
        positions=sequence.positions[indices].copy(),
        source_row_ids=[sequence.source_row_ids[index] for index in indices],
        score_mask=score,
    )


def target_replay_sequences(
    sequences: Sequence[StudentSequence],
    perturbation: str,
    strength: float | int,
    seed: int,
    unrelated_concepts: np.ndarray | None = None,
    semantic_embeddings: np.ndarray | None = None,
    insertion_relation: str = "unrelated",
) -> list[StudentSequence]:
    rng = np.random.default_rng(seed)
    replays: list[StudentSequence] = []
    for sequence in sequences:
        for target in np.flatnonzero(sequence.score_mask):
            prefix = np.arange(target + 1, dtype=int)
            history = prefix[:-1]
            if perturbation == "history_delete" and len(history):
                delete_count = min(len(history), int(round(len(history) * float(strength))))
                deleted = set(rng.choice(history, size=delete_count, replace=False).tolist()) if delete_count else set()
                prefix = np.asarray([index for index in prefix if index not in deleted], dtype=int)
            replay = _subset(sequence, prefix, target)
            if perturbation == "label_flip" and len(replay.labels) > 1:
                history_count = len(replay.labels) - 1
                flip_count = min(history_count, int(round(history_count * float(strength))))
                if flip_count:
                    selected = rng.choice(np.arange(history_count), size=flip_count, replace=False)
                    replay.labels[selected] = 1.0 - replay.labels[selected]
            elif perturbation == "adjacent_swap" and len(replay.labels) > 2:
                history_count = len(replay.labels) - 1
                candidates = np.arange(history_count - 1)
                if semantic_embeddings is not None:
                    left = semantic_embeddings[replay.concept_indices[candidates]]
                    right = semantic_embeddings[replay.concept_indices[candidates + 1]]
                    left = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1.0e-8, None)
                    right = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1.0e-8, None)
                    distances = 1.0 - np.sum(left * right, axis=1)
                    candidates = candidates[np.argsort(-distances, kind="mergesort")]
                else:
                    rng.shuffle(candidates)
                swaps = max(1, int(round(history_count * float(strength))))
                used: set[int] = set()
                for index in candidates:
                    index = int(index)
                    if index in used or index + 1 in used:
                        continue
                    for array in (replay.concept_indices, replay.labels):
                        array[index], array[index + 1] = array[index + 1].copy(), array[index].copy()
                    for values in (replay.question_ids, replay.kc_ids, replay.source_row_ids):
                        values[index], values[index + 1] = values[index + 1], values[index]
                    used.update({index, index + 1})
                    if len(used) // 2 >= swaps:
                        break
            elif perturbation == "unrelated_insert":
                candidates = unrelated_concepts
                if semantic_embeddings is not None:
                    target_index = int(replay.concept_indices[-1])
                    matrix = np.asarray(semantic_embeddings, dtype=np.float64)
                    normalized = matrix / np.clip(
                        np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-8, None
                    )
                    distances = 1.0 - normalized @ normalized[target_index]
                    order = np.argsort(distances, kind="mergesort")
                    order = order[order != target_index]
                    quartile = max(1, int(np.ceil(len(order) / 4)))
                    if insertion_relation == "related":
                        candidates = order[:quartile]
                    elif insertion_relation == "unrelated":
                        candidates = order[-quartile:]
                    else:
                        raise ValueError(f"Unknown insertion relation: {insertion_relation}")
                if candidates is None or len(candidates) == 0:
                    raise ValueError("History insertion requires candidate concept indices")
                insertions = int(strength)
                for insertion in range(insertions):
                    location = int(rng.integers(0, max(1, len(replay.labels))))
                    concept = int(rng.choice(candidates))
                    replay.concept_indices = np.insert(replay.concept_indices, location, concept)
                    replay.labels = np.insert(replay.labels, location, int(rng.integers(0, 2))).astype(np.float32)
                    replay.positions = np.insert(replay.positions, location, 0)
                    replay.question_ids.insert(location, f"robustness_insert_{insertion}")
                    replay.kc_ids.insert(location, f"robustness_kc_{concept}")
                    replay.source_row_ids.insert(location, f"robustness_insert_{insertion}")
                    replay.score_mask = np.insert(replay.score_mask, location, False)
            replays.append(replay)
    return replays


def history_concept_remap_sequences(
    sequences: Sequence[StudentSequence], mapping: Mapping[int, int]
) -> list[StudentSequence]:
    """Replay every scored target while remapping history descriptors only."""

    replays: list[StudentSequence] = []
    for sequence in sequences:
        for target in np.flatnonzero(sequence.score_mask):
            replay = _subset(sequence, np.arange(target + 1, dtype=int), target)
            history = replay.concept_indices[:-1]
            for source, replacement in mapping.items():
                history[history == int(source)] = int(replacement)
            replays.append(replay)
    return replays
