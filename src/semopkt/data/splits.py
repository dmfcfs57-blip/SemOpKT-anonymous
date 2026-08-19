"""Immutable student/concept manifests and leakage-safe frame construction."""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from semopkt.constants import PREPROCESS_VERSION
from semopkt.data.schema import interaction_contains_heldout, parse_component_cell
from semopkt.utils.hashing import hash_dataframe, hash_json
from semopkt.utils.io import read_json, write_json


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    experiment: str
    protocol: str
    seed: int
    train_student_order: tuple[str, ...]
    validation_students: tuple[str, ...]
    test_students: tuple[str, ...]
    heldout_kcs: tuple[str, ...] = ()
    train_sizes: tuple[int, ...] = (8, 16, 32, 64)
    holdout_ratio: float | None = None
    preprocess_version: str = PREPROCESS_VERSION
    source_table_hash: str = ""
    train_interaction_hash: str = ""
    validation_interaction_hash: str = ""
    test_interaction_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        for key in (
            "train_student_order",
            "validation_students",
            "test_students",
            "heldout_kcs",
            "train_sizes",
        ):
            value[key] = list(value[key])
        if not include_hash:
            value.pop("manifest_hash", None)
        return value

    def with_hash(self) -> "SplitManifest":
        digest = hash_json(self.to_dict(include_hash=False))
        return dataclasses.replace(self, manifest_hash=digest)

    def save(self, path: str | Path) -> None:
        manifest = self if self.manifest_hash else self.with_hash()
        write_json(path, manifest.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        value = read_json(path)
        for key in (
            "train_student_order",
            "validation_students",
            "test_students",
            "heldout_kcs",
            "train_sizes",
        ):
            value[key] = tuple(value.get(key, []))
        manifest = cls(**value)
        expected = manifest.with_hash().manifest_hash
        if manifest.manifest_hash != expected:
            raise ValueError(f"Manifest hash mismatch: {path}")
        return manifest

    def nested_train_students(self, size: int | None) -> set[str]:
        if size is None:
            return set(self.train_student_order)
        if size not in self.train_sizes:
            raise ValueError(f"Training size {size} is not declared in manifest {self.train_sizes}")
        if size > len(self.train_student_order):
            raise ValueError(
                f"Training size {size} exceeds available students {len(self.train_student_order)}"
            )
        return set(self.train_student_order[:size])


@dataclass
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    test_target_mask: pd.Series


def _partition_students(
    students: Sequence[str], seed: int, test_fraction: float, validation_fraction: float
) -> tuple[list[str], list[str], list[str]]:
    ordered = np.array(sorted(map(str, students)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    test_count = max(1, int(round(len(ordered) * test_fraction)))
    test = ordered[:test_count].tolist()
    development = ordered[test_count:]
    validation_count = max(1, int(round(len(development) * validation_fraction)))
    validation = development[:validation_count].tolist()
    train = development[validation_count:].tolist()
    if not train:
        raise ValueError("Student split leaves no training student")
    return train, validation, test


def _component_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[str, str, int]] = []
    for row in frame.itertuples(index=False):
        for component in parse_component_cell(row.kc_components):
            rows.append((str(row.student_id), component, int(row.correct)))
    exploded = pd.DataFrame(rows, columns=["student_id", "component", "correct"])
    grouped = exploded.groupby("component", sort=True)
    return grouped.agg(
        students=("student_id", "nunique"),
        interactions=("correct", "size"),
        positives=("correct", "sum"),
    ).assign(negatives=lambda value: value["interactions"] - value["positives"])


def eligible_concepts(frame: pd.DataFrame, criteria: Mapping[str, int]) -> pd.DataFrame:
    stats = _component_statistics(frame)
    mask = (
        (stats["students"] >= int(criteria["minimum_students"]))
        & (stats["interactions"] >= int(criteria["minimum_interactions"]))
        & (stats["positives"] >= int(criteria["minimum_positive"]))
        & (stats["negatives"] >= int(criteria["minimum_negative"]))
    )
    return stats.loc[mask].copy()


def _hash_text_embedding(text: str, dimension: int = 64, salt: str = "split") -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float64)
    tokens = text.split() or [text]
    for token in tokens:
        digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=16).digest()
        for offset in range(0, len(digest), 2):
            index = int.from_bytes(digest[offset : offset + 2], "little") % dimension
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _choose_random_holdout(
    concepts: Sequence[str], ratio: float, seed: int
) -> tuple[str, ...]:
    values = np.array(sorted(concepts), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    count = max(1, int(round(len(values) * ratio)))
    return tuple(sorted(map(str, values[:count])))


def _choose_cluster_holdout(
    concepts: Sequence[str],
    ratio: float,
    seed: int,
    embeddings: Mapping[str, np.ndarray] | None,
    n_init: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    ordered = sorted(map(str, concepts))
    matrix = np.vstack(
        [
            np.asarray(embeddings[concept], dtype=np.float64)
            if embeddings is not None and concept in embeddings
            else _hash_text_embedding(concept)
            for concept in ordered
        ]
    )
    cluster_count = min(len(ordered), max(2, max(8, int(math.ceil(math.sqrt(len(ordered)))))))
    model = KMeans(n_clusters=cluster_count, init="k-means++", n_init=n_init, random_state=seed)
    labels = model.fit_predict(matrix)
    rng = np.random.default_rng(seed + 17)
    cluster_order = np.arange(cluster_count)
    rng.shuffle(cluster_order)
    target_count = max(1, int(round(len(ordered) * ratio)))
    chosen: list[int] = []
    current = 0
    for cluster_id in cluster_order:
        size = int(np.sum(labels == cluster_id))
        if not chosen or abs((current + size) - target_count) <= abs(current - target_count):
            chosen.append(int(cluster_id))
            current += size
        if current >= target_count:
            break
    heldout = tuple(
        concept for concept, label in zip(ordered, labels, strict=True) if int(label) in chosen
    )
    metadata = {
        "cluster_count": cluster_count,
        "chosen_clusters": chosen,
        "cluster_sizes": {
            str(cluster): int(np.sum(labels == cluster)) for cluster in range(cluster_count)
        },
        "embedding_source": "configured-independent-encoder" if embeddings else "deterministic-hash",
    }
    return tuple(sorted(heldout)), metadata


def _normalized_edit_distance(left: str, right: str) -> float:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(1, len(left), len(right))


def _heldout_text_audit(
    heldout: Sequence[str],
    available: Sequence[str],
    embeddings: Mapping[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    training = sorted(set(map(str, available)) - set(map(str, heldout)))
    if not training:
        return []
    rows: list[dict[str, Any]] = []
    for concept in sorted(map(str, heldout)):
        character_scores = [
            difflib.SequenceMatcher(None, concept, candidate).ratio()
            for candidate in training
        ]
        nearest_character = int(np.argmax(character_scores))
        row: dict[str, Any] = {
            "heldout_concept": concept,
            "nearest_character_concept": training[nearest_character],
            "character_similarity": float(character_scores[nearest_character]),
            "normalized_edit_distance": float(
                _normalized_edit_distance(concept, training[nearest_character])
            ),
        }
        if embeddings is not None and concept in embeddings:
            target = np.asarray(embeddings[concept], dtype=np.float64)
            target /= max(1.0e-12, float(np.linalg.norm(target)))
            candidates = np.vstack(
                [np.asarray(embeddings[value], dtype=np.float64) for value in training]
            )
            candidates /= np.clip(
                np.linalg.norm(candidates, axis=1, keepdims=True), 1.0e-12, None
            )
            similarities = candidates @ target
            nearest_semantic = int(np.argmax(similarities))
            row.update(
                {
                    "nearest_semantic_concept": training[nearest_semantic],
                    "nearest_semantic_cosine": float(similarities[nearest_semantic]),
                    "nearest_semantic_distance": float(
                        1.0 - similarities[nearest_semantic]
                    ),
                }
            )
        rows.append(row)
    return rows


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hash_json([])
    columns = ["source_row_id", "student_id", "kc_id", "correct", "position"]
    return hash_dataframe(frame, columns=columns)


def _heldout_mask(frame: pd.DataFrame, heldout: set[str]) -> pd.Series:
    if not heldout:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["kc_components"].map(lambda value: interaction_contains_heldout(value, heldout))


def generate_split_manifest(
    frame: pd.DataFrame,
    experiment: str,
    protocol: str,
    seed: int,
    train_sizes: Sequence[int] = (8, 16, 32, 64),
    test_fraction: float = 0.20,
    validation_fraction: float = 0.15,
    holdout_ratio: float | None = None,
    unseen_criteria: Mapping[str, int] | None = None,
    cluster_embeddings: Mapping[str, np.ndarray] | None = None,
    cluster_n_init: int = 20,
) -> SplitManifest:
    if frame["dataset"].nunique() != 1:
        raise ValueError("A split manifest binds exactly one dataset")
    dataset = str(frame["dataset"].iloc[0])
    train, validation, disjoint_test = _partition_students(
        frame["student_id"].astype(str).unique(), seed, test_fraction, validation_fraction
    )
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(train)
    heldout: tuple[str, ...] = ()
    metadata: dict[str, Any] = {"user_overlap_allowed": protocol.startswith("unseen_")}
    if protocol in {"unseen_random", "unseen_cluster", "double_random", "double_cluster"}:
        if holdout_ratio is None or unseen_criteria is None:
            raise ValueError(f"Protocol {protocol} requires holdout ratio and eligibility criteria")
        stats = eligible_concepts(frame, unseen_criteria)
        if len(stats) < 2:
            raise ValueError("Fewer than two eligible concepts; unseen-concept split is undefined")
        if protocol.endswith("random"):
            heldout = _choose_random_holdout(stats.index.tolist(), holdout_ratio, seed + 2)
        else:
            heldout, cluster_metadata = _choose_cluster_holdout(
                stats.index.tolist(), holdout_ratio, seed + 2, cluster_embeddings, cluster_n_init
            )
            metadata.update(cluster_metadata)
        metadata["eligible_concepts"] = int(len(stats))
        metadata["heldout_statistics"] = stats.loc[list(heldout)].to_dict(orient="index")
        metadata["heldout_text_overlap_audit"] = _heldout_text_audit(
            heldout,
            stats.index.tolist(),
            cluster_embeddings,
        )
    if protocol.startswith("unseen_"):
        test_students = sorted(frame["student_id"].astype(str).unique().tolist())
    else:
        test_students = disjoint_test
    provisional = SplitManifest(
        dataset=dataset,
        experiment=experiment,
        protocol=protocol,
        seed=int(seed),
        train_student_order=tuple(train),
        validation_students=tuple(sorted(validation)),
        test_students=tuple(sorted(test_students)),
        heldout_kcs=heldout,
        train_sizes=tuple(int(size) for size in train_sizes if int(size) <= len(train)),
        holdout_ratio=holdout_ratio,
        source_table_hash=hash_dataframe(frame),
        metadata=metadata,
    )
    frames = apply_manifest(frame, provisional, train_size=None)
    manifest = dataclasses.replace(
        provisional,
        train_interaction_hash=_frame_hash(frames.train),
        validation_interaction_hash=_frame_hash(frames.validation),
        test_interaction_hash=_frame_hash(frames.test.loc[frames.test_target_mask]),
    ).with_hash()
    validate_manifest(frame, manifest)
    return manifest


def _first_heldout_encounter_mask(frame: pd.DataFrame, heldout: set[str]) -> pd.Series:
    target = _heldout_mask(frame, heldout)
    result = pd.Series(False, index=frame.index, dtype=bool)
    seen: dict[str, set[str]] = {}
    ordered = frame.sort_values(["student_id", "position", "source_row_id"], kind="mergesort")
    for index, row in ordered.iterrows():
        student = str(row["student_id"])
        student_seen = seen.setdefault(student, set())
        components = set(parse_component_cell(row["kc_components"])) & heldout
        if bool(target.loc[index]) and any(component not in student_seen for component in components):
            result.loc[index] = True
        student_seen.update(components)
    return result


def apply_manifest(
    frame: pd.DataFrame, manifest: SplitManifest, train_size: int | None
) -> SplitFrames:
    heldout = set(manifest.heldout_kcs)
    train_students = manifest.nested_train_students(train_size)
    train = frame[frame["student_id"].astype(str).isin(train_students)].copy()
    validation = frame[
        frame["student_id"].astype(str).isin(set(manifest.validation_students))
    ].copy()
    test = frame[frame["student_id"].astype(str).isin(set(manifest.test_students))].copy()
    if heldout:
        train = train.loc[~_heldout_mask(train, heldout)].copy()
        validation = validation.loc[~_heldout_mask(validation, heldout)].copy()
    if manifest.protocol.startswith("unseen_"):
        # Concept novelty is isolated; students may overlap with global fitting.
        test = frame.copy()
    if heldout:
        target_mask = _first_heldout_encounter_mask(test, heldout)
    else:
        target_mask = test["position"].astype(int) >= 2
    return SplitFrames(
        train=train.sort_values(["student_id", "position"], kind="mergesort"),
        validation=validation.sort_values(["student_id", "position"], kind="mergesort"),
        test=test.sort_values(["student_id", "position"], kind="mergesort"),
        test_target_mask=target_mask.reindex(test.index, fill_value=False),
    )


def validate_manifest(frame: pd.DataFrame, manifest: SplitManifest) -> None:
    train_students = set(manifest.train_student_order)
    validation_students = set(manifest.validation_students)
    test_students = set(manifest.test_students)
    if train_students & validation_students:
        raise ValueError("Training and validation students overlap")
    if not manifest.metadata.get("user_overlap_allowed", False):
        if train_students & test_students or validation_students & test_students:
            raise ValueError("Disjoint-user protocol contains student overlap")
    frames = apply_manifest(frame, manifest, train_size=None)
    heldout = set(manifest.heldout_kcs)
    if heldout:
        if _heldout_mask(frames.train, heldout).any():
            raise ValueError("Held-out concept leaks into training interactions")
        if _heldout_mask(frames.validation, heldout).any():
            raise ValueError("Held-out concept leaks into validation interactions")
        if not frames.test_target_mask.any():
            raise ValueError("Held-out manifest has no evaluable target interaction")
    if manifest.source_table_hash and manifest.source_table_hash != hash_dataframe(frame):
        raise ValueError("Source table hash does not match manifest")
