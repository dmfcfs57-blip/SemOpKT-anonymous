"""Hash-bound execution engine for local and cross-domain experiment specifications."""

from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from semopkt.analysis.perturbations import (
    append_semantic_embeddings,
    delete_descriptor_tokens,
    history_concept_remap_sequences,
    parse_perturbation_condition,
    replace_descriptor_synonyms,
    target_replay_sequences,
)
from semopkt.analysis.error_analysis import (
    classify_error_cases,
    grouped_metrics,
    high_confidence_errors,
    prediction_covariates,
)
from semopkt.analysis.efficiency import measure_inference, measure_semopkt_query_scaling
from semopkt.analysis.expansion import query_only_expansion_drift
from semopkt.analysis.adaptation import old_concept_stability_replay
from semopkt.analysis.propagation import (
    empirical_association_matrix,
    propagation_alignment,
    replay_effective_influence,
    semopkt_effective_influence,
)
from semopkt.analysis.qualitative import trace_semopkt_targets
from semopkt.config import canonical_config, config_hash, deep_merge, load_config
from semopkt.data.schema import interaction_contains_heldout
from semopkt.data.sequences import (
    ConceptVocabulary,
    build_sequences,
    collate_sequences,
    remap_concept_indices,
)
from semopkt.data.splits import (
    SplitManifest,
    apply_manifest,
    generate_split_manifest,
)
from semopkt.evaluation.calibration import TemperatureScaler
from semopkt.evaluation.metrics import compute_metrics, metrics_by_history_window
from semopkt.experiments.specs import RunSpec
from semopkt.models.registry import build_model
from semopkt.models.semopkt import SemOpKT
from semopkt.semantics.encoder import build_text_encoder, encode_with_cache
from semopkt.training.trainer import Trainer
from semopkt.utils.hashing import bind_hashes, hash_dataframe, hash_file, hash_tree
from semopkt.utils.io import read_json, read_table, write_json, write_table, write_yaml
from semopkt.utils.provenance import environment_record
from semopkt.utils.random import seed_everything


_CLOSED_VOCABULARY_MODELS = {
    "BKT",
    "DKT",
    "DKVMN",
    "AKT",
    "simpleKT",
    "csKT",
    "UKT",
    "MAML-KT",
    "A1",
    "A4",
}


def _resolve(reference: str | Path, base: Path) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _canonical_protocol(protocol: str) -> str:
    mapping = {
        "system": "system",
        "target_only": "system",
        "full_data": "system",
        "early_history": "system",
        "operator_comparison": "system",
        "propagation": "system",
        "robustness": "system",
        "calibration_system": "system",
        "data_efficiency": "system",
        "sensitivity": "system",
        "efficiency": "system",
        "semantic_control": "system",
        "semantic_system": "system",
        "semantic_random": "unseen_random",
        "semantic_cluster": "unseen_cluster",
        "semantic_double": "double_cluster",
        "operator_cluster": "unseen_cluster",
        "operator_double": "double_cluster",
        "unseen_random": "unseen_random",
        "online_adaptation": "unseen_random",
        "vocabulary_expansion": "double_random",
        "vocabulary_full": "system",
        "unseen_cluster": "unseen_cluster",
        "calibration_cluster": "unseen_cluster",
        "error_analysis": "unseen_cluster",
        "double_random": "double_random",
        "double_cluster": "double_cluster",
        "calibration_double": "double_cluster",
        "qualitative": "double_cluster",
        "qualitative_system": "system",
        "qualitative_unseen": "unseen_cluster",
        "qualitative_double": "double_cluster",
        "inducing_sensitivity": "system",
    }
    if protocol not in mapping:
        raise KeyError(f"No local manifest protocol for {protocol}")
    return mapping[protocol]


def _manifest_name(spec: RunSpec, canonical_protocol: str) -> str:
    ratio = f"_r{int(round(spec.holdout_ratio * 100))}" if spec.holdout_ratio else ""
    return f"{spec.dataset}_{canonical_protocol}{ratio}_seed{spec.seed}.json"


def _all_target_mask(frame: pd.DataFrame, heldout: set[str]) -> pd.Series:
    if not heldout:
        return frame["position"].astype(int) >= 2
    return frame["kc_components"].map(
        lambda value: interaction_contains_heldout(value, heldout)
    )


def _heldout_descriptor_indices(
    frame: pd.DataFrame, vocabulary: ConceptVocabulary, heldout: set[str]
) -> set[int]:
    if not heldout:
        return set()
    mask = frame["kc_components"].map(
        lambda value: interaction_contains_heldout(value, heldout)
    )
    texts = set(frame.loc[mask, "kc_text_norm"].astype(str))
    return {vocabulary.text_to_index[text] for text in texts}


def _support_validation_students(
    ordered_students: Sequence[str], support_fraction: float
) -> tuple[set[str], set[str]]:
    students = list(map(str, ordered_students))
    if len(students) < 2:
        raise ValueError("Few-shot adaptation requires at least two target students")
    support_count = min(
        len(students) - 1,
        max(1, int(round(len(students) * support_fraction))),
    )
    return set(students[:support_count]), set(students[support_count:])


def _artifact_hashes(directory: Path) -> dict[str, str]:
    excluded = {"run.json", "complete.json"}
    return {
        path.relative_to(directory).as_posix(): hash_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _load_text_resource(repository: Path, dataset: str, kind: str) -> dict[str, str]:
    path = repository / "data" / "text" / f"{dataset}_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"The {kind} condition requires the anonymous text resource {path.relative_to(repository)}"
        )
    frame = pd.read_csv(path)
    required = {"kc_text_norm", "replacement_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Text resource {path.name} must contain {sorted(required)}")
    return dict(zip(frame["kc_text_norm"].astype(str), frame["replacement_text"].astype(str), strict=True))


def _load_synonym_resource(repository: Path, dataset: str) -> dict[str, str]:
    path = repository / "data" / "text" / f"{dataset}_synonyms.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"The synonym robustness condition requires {path.relative_to(repository)}"
        )
    frame = pd.read_csv(path)
    required = {"source_token", "replacement_token"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Synonym resource {path.name} must contain {sorted(required)}")
    source = frame["source_token"].astype(str).str.strip().str.casefold()
    replacement = frame["replacement_token"].astype(str).str.strip().str.casefold()
    if source.duplicated().any() or (source == "").any() or (replacement == "").any():
        raise ValueError(f"Synonym resource {path.name} has blank or duplicate source tokens")
    return dict(zip(source, replacement, strict=True))


def _transform_embeddings(
    matrix: np.ndarray, condition: str, seed: int
) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    if condition in {"default", "original", "normalized", "paraphrase", "external_definitions"}:
        return result
    if condition == "shuffled_names":
        return result[rng.permutation(len(result))]
    if condition in {"random_vectors", "learnable_identifiers"}:
        result = rng.normal(size=result.shape).astype(np.float32)
        result /= np.clip(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-8, None)
        return result
    if condition == "masked":
        mean = result.mean(axis=0, keepdims=True)
        return np.repeat(mean, len(result), axis=0)
    return result


class ExperimentRunner:
    def __init__(self, configuration: Mapping[str, Any] | str | Path, device: str | None = None):
        self.config = load_config(configuration) if isinstance(configuration, (str, Path)) else dict(configuration)
        self.config_path = Path(self.config["_config_path"]).resolve()
        self.config_base = self.config_path.parent
        self.repository = self.config_base.parent.parent
        self.device = device
        self.model_config = load_config(_resolve(self.config["model_config"], self.config_base))
        self.baseline_config = load_config(_resolve(self.config["baseline_config"], self.config_base))
        self.output_root = _resolve(self.config["output_root"], self.config_base)
        self.manifest_root = _resolve(self.config["manifest_root"], self.config_base)
        self.embedding_root = _resolve(self.config["embedding_root"], self.config_base)
        self.source_tree_hash = hash_tree(self.repository)

    def _dataset_path(self, dataset: str) -> Path:
        return _resolve(self.config["datasets"][dataset]["processed"], self.config_base)

    def _load_dataset(self, dataset: str) -> pd.DataFrame:
        frame = read_table(self._dataset_path(dataset))
        if frame.empty or set(frame["dataset"].astype(str).unique()) != {dataset}:
            raise ValueError(f"Processed table does not bind dataset {dataset}")
        return frame

    def _encoder_and_lookup(
        self,
        frame: pd.DataFrame,
        split: bool = False,
        additional_texts: Sequence[str] = (),
        model_config: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        key = "split_encoder" if split else "text_encoder"
        source_config = self.model_config if split or model_config is None else model_config
        encoder_config = copy.deepcopy(source_config[key])
        if encoder_config.get("backend") == "hash":
            encoder_config["dimension"] = int(source_config["dimensions"]["text"])
            if split:
                encoder_config["dimension"] = max(16, int(self.model_config["dimensions"]["text"]))
        encoder = build_text_encoder(encoder_config, device=self.device)
        texts = frame["kc_text_norm"].astype(str).tolist() + list(additional_texts)
        namespace = f"{frame['dataset'].iloc[0]}-{'split' if split else 'predictive'}"
        return encode_with_cache(texts, encoder, self.embedding_root, namespace)

    def _component_embeddings(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        components: set[str] = set()
        for value in frame["kc_components"]:
            parsed = json.loads(value) if isinstance(value, str) and value.startswith("[") else [str(value)]
            components.update(map(str, parsed))
        lookup, _ = self._encoder_and_lookup(frame, split=True, additional_texts=sorted(components))
        return {component: lookup[component] for component in components}

    def _manifest(self, frame: pd.DataFrame, spec: RunSpec) -> SplitManifest:
        canonical = _canonical_protocol(spec.protocol)
        path = self.manifest_root / spec.dataset / _manifest_name(spec, canonical)
        if path.exists():
            manifest = SplitManifest.load(path)
            if manifest.source_table_hash != hash_dataframe(frame):
                raise ValueError(f"Existing manifest is bound to another processed table: {path}")
            return manifest
        unseen = self.config["unseen_concept"]
        criteria = {
            "minimum_students": int(unseen["minimum_students"]),
            "minimum_interactions": int(unseen["minimum_interactions"]),
            "minimum_positive": int(unseen["minimum_positive"]),
            "minimum_negative": int(unseen["minimum_negative"]),
        }
        cluster_embeddings = (
            self._component_embeddings(frame) if canonical.endswith("cluster") else None
        )
        manifest = generate_split_manifest(
            frame,
            experiment=spec.experiment,
            protocol=canonical,
            seed=spec.seed,
            train_sizes=tuple(
                sorted(
                    set(
                        int(value)
                        for value in (
                            list(self.config["train_sizes"])
                            + list(self.config.get("data_efficiency_sizes", []))
                        )
                    )
                )
            ),
            test_fraction=float(self.config["student_split"]["test_fraction"]),
            validation_fraction=float(
                self.config["student_split"]["validation_fraction_of_development"]
            ),
            holdout_ratio=spec.holdout_ratio,
            unseen_criteria=criteria if spec.holdout_ratio is not None else None,
            cluster_embeddings=cluster_embeddings,
            cluster_n_init=int(unseen["cluster_initializations"]),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(path)
        return manifest

    def _vocabulary(
        self,
        frame: pd.DataFrame,
        condition: str,
        seed: int,
        model_config: Mapping[str, Any] | None = None,
    ) -> tuple[ConceptVocabulary, dict[str, Any]]:
        original_texts = tuple(sorted(frame["kc_text_norm"].astype(str).unique().tolist()))
        replacements = {text: text for text in original_texts}
        if condition in {"paraphrase", "normalized", "external_definitions"}:
            kind = {
                "paraphrase": "paraphrases",
                "normalized": "normalizations",
                "external_definitions": "definitions",
            }[condition]
            mapping = _load_text_resource(self.repository, str(frame["dataset"].iloc[0]), kind)
            missing_resources = sorted(set(original_texts) - set(mapping))
            if missing_resources:
                raise ValueError(
                    f"The {kind} resource is missing {len(missing_resources)} descriptors"
                )
            if condition == "external_definitions":
                replacements = {
                    text: f"{text} {mapping[text]}" if text in mapping else text
                    for text in original_texts
                }
            else:
                replacements = {text: mapping.get(text, text) for text in original_texts}
        elif condition == "delete_20_percent":
            rng = np.random.default_rng(seed)
            for text in original_texts:
                tokens = text.split()
                if len(tokens) > 1:
                    count = max(1, int(round(0.20 * len(tokens))))
                    removed = set(rng.choice(len(tokens), size=min(count, len(tokens) - 1), replace=False).tolist())
                    replacements[text] = " ".join(
                        token for index, token in enumerate(tokens) if index not in removed
                    )
        elif condition == "masked":
            replacements = {text: "[masked concept]" for text in original_texts}
        lookup, metadata = self._encoder_and_lookup(
            frame,
            additional_texts=sorted(set(replacements.values())),
            model_config=model_config,
        )
        matrix = np.vstack([lookup[replacements[text]] for text in original_texts]).astype(np.float32)
        transformed = _transform_embeddings(matrix, condition, seed)
        return ConceptVocabulary(
            original_texts,
            {text: index for index, text in enumerate(original_texts)},
            transformed,
        ), metadata

    def _resolved_model_config(
        self, overrides: Mapping[str, Any], model_name: str
    ) -> dict[str, Any]:
        resolved = canonical_config(deep_merge(self.model_config, overrides))
        resolved["name"] = model_name
        resolved["dimensions"]["text"] = int(resolved["dimensions"]["text"])
        return resolved

    def _run_directory(self, spec: RunSpec) -> Path:
        return self.output_root / spec.experiment / spec.dataset / spec.model / spec.key

    def _validated_completion(
        self, completion: Path, spec: RunSpec
    ) -> dict[str, Any] | None:
        if not completion.exists():
            return None
        record = json.loads(completion.read_text(encoding="utf-8"))
        if (
            record.get("status") != "complete"
            or record.get("specification") != spec.to_dict()
            or record.get("source_tree_hash") != self.source_tree_hash
        ):
            return None
        if spec.source_dataset and spec.target_dataset:
            expected_sources = record.get("source_table_hashes", {})
            for source in spec.source_dataset.split("+"):
                if expected_sources.get(source) != hash_dataframe(
                    self._load_dataset(source)
                ):
                    return None
            if record.get("target_table_hash") != hash_dataframe(
                self._load_dataset(spec.target_dataset)
            ):
                return None
            manifest_hashes = record.get("source_manifest_hashes", {})
            for source in spec.source_dataset.split("+"):
                path = (
                    self.manifest_root
                    / source
                    / _manifest_name(
                        RunSpec(
                            spec.experiment,
                            source,
                            "system",
                            spec.model,
                            spec.seed,
                        ),
                        "system",
                    )
                )
                if not path.exists() or SplitManifest.load(path).manifest_hash != manifest_hashes.get(source):
                    return None
            target_path = (
                self.manifest_root
                / spec.target_dataset
                / _manifest_name(
                    RunSpec(
                        spec.experiment,
                        spec.target_dataset,
                        "system",
                        spec.model,
                        spec.seed,
                    ),
                    "system",
                )
            )
            if (
                not target_path.exists()
                or SplitManifest.load(target_path).manifest_hash
                != record.get("target_manifest_hash")
            ):
                return None
        else:
            frame = self._load_dataset(spec.dataset)
            if record.get("source_table_hash") != hash_dataframe(frame):
                return None
            canonical = _canonical_protocol(spec.protocol)
            manifest_path = (
                self.manifest_root
                / spec.dataset
                / _manifest_name(spec, canonical)
            )
            if (
                not manifest_path.exists()
                or SplitManifest.load(manifest_path).manifest_hash
                != record.get("manifest_hash")
            ):
                return None
        encoder = record.get("encoder", {})
        cache_key = encoder.get("cache_key")
        matrix_hash = encoder.get("matrix_sha256")
        if cache_key:
            matrix_path = self.embedding_root / f"{cache_key}.npz"
            metadata_path = self.embedding_root / f"{cache_key}.json"
            if (
                not matrix_path.exists()
                or not metadata_path.exists()
                or matrix_hash != hash_file(matrix_path)
                or read_json(metadata_path).get("matrix_sha256") != matrix_hash
            ):
                return None
        perturbation_key = record.get("robustness", {}).get(
            "perturbation_encoder_cache_key"
        ) if record.get("robustness") else None
        if perturbation_key:
            perturbation_path = self.embedding_root / f"{perturbation_key}.npz"
            perturbation_metadata = self.embedding_root / f"{perturbation_key}.json"
            if not perturbation_path.exists() or not perturbation_metadata.exists():
                return None
            expected = read_json(perturbation_metadata).get("matrix_sha256")
            if expected != hash_file(perturbation_path):
                return None
        if record.get("artifact_hashes") != _artifact_hashes(completion.parent):
            return None
        return record

    def _robustness_predictions(
        self,
        spec: RunSpec,
        frame: pd.DataFrame,
        vocabulary: ConceptVocabulary,
        model_config: Mapping[str, Any],
        test_sequences: Sequence[Any],
        trainer: Trainer,
        clean: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        kind, strength = parse_perturbation_condition(spec.condition)
        metadata: dict[str, Any] = {
            "kind": kind,
            "strength": strength,
            "target_labels_perturbed": False,
            "mask_seed": spec.seed,
        }
        related: pd.DataFrame | None = None
        if kind in {"name_delete", "synonym_replace"}:
            if bool(getattr(trainer.model, "uses_semantic_descriptors", False)):
                if kind == "name_delete":
                    transformed = delete_descriptor_tokens(
                        vocabulary.texts, float(strength), spec.seed
                    )
                else:
                    synonyms = _load_synonym_resource(
                        self.repository, str(frame["dataset"].iloc[0])
                    )
                    transformed = replace_descriptor_synonyms(
                        vocabulary.texts, synonyms, float(strength), spec.seed
                    )
                lookup, perturbation_encoder = self._encoder_and_lookup(
                    frame,
                    additional_texts=transformed,
                    model_config=model_config,
                )
                matrix = np.vstack([lookup[text] for text in transformed]).astype(np.float32)
                offset = len(vocabulary.texts)
                if not append_semantic_embeddings(trainer.model, matrix):
                    raise TypeError(
                        f"Model {spec.model} declares semantic input but cannot extend descriptors"
                    )
                replay = history_concept_remap_sequences(
                    test_sequences,
                    {index: offset + index for index in range(len(vocabulary.texts))},
                )
                metadata["perturbation_encoder_cache_key"] = perturbation_encoder["cache_key"]
                metadata["changed_descriptors"] = int(
                    sum(original != changed for original, changed in zip(vocabulary.texts, transformed, strict=True))
                )
            else:
                replay = history_concept_remap_sequences(test_sequences, {})
                metadata["changed_descriptors"] = 0
                metadata["closed_vocabulary_invariant"] = True
        else:
            replay = target_replay_sequences(
                test_sequences,
                kind,
                strength,
                spec.seed,
                semantic_embeddings=vocabulary.embeddings,
                insertion_relation="unrelated",
            )
            if kind == "unrelated_insert":
                related_replay = target_replay_sequences(
                    test_sequences,
                    kind,
                    strength,
                    spec.seed,
                    semantic_embeddings=vocabulary.embeddings,
                    insertion_relation="related",
                )
                related = trainer.predict(related_replay, score_only=True)
        perturbed = trainer.predict(replay, score_only=True)
        keys = ["dataset", "student_id", "source_row_id", "correct", "position"]
        clean_columns = keys + ["logit", "probability"]
        clean_lookup = clean[clean_columns].rename(
            columns={"logit": "clean_logit", "probability": "clean_probability"}
        )
        perturbed = perturbed.merge(
            clean_lookup,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(perturbed) != len(clean):
            raise ValueError("Robustness replay did not preserve the complete target set")
        perturbed["probability_drift"] = (
            perturbed["probability"] - perturbed["clean_probability"]
        ).abs()
        metadata["mean_probability_drift"] = float(perturbed["probability_drift"].mean())
        metadata["clean_metrics"] = compute_metrics(clean)
        metadata["perturbed_metrics"] = compute_metrics(perturbed)
        metadata["auc_change"] = float(
            metadata["perturbed_metrics"]["auc"] - metadata["clean_metrics"]["auc"]
        )
        if related is not None:
            related_lookup = related[keys + ["probability"]].rename(
                columns={"probability": "related_probability"}
            )
            perturbed = perturbed.merge(
                related_lookup,
                on=keys,
                how="left",
                validate="one_to_one",
            )
            related_drift = (
                perturbed["related_probability"] - perturbed["clean_probability"]
            ).abs()
            unrelated_drift = perturbed["probability_drift"]
            denominator = float(unrelated_drift.mean())
            metadata["related_mean_probability_drift"] = float(related_drift.mean())
            metadata["selective_sensitivity_ratio"] = (
                float(related_drift.mean()) / denominator
                if denominator > 0
                else float("nan")
            )
        return perturbed, metadata

    def _write_propagation_analysis(
        self,
        run_directory: Path,
        training_frame: pd.DataFrame,
        training_sequences: Sequence[Any],
        vocabulary: ConceptVocabulary,
        trainer: Trainer,
    ) -> dict[str, Any]:
        settings = self.config.get("propagation", {})
        empirical, audit = empirical_association_matrix(
            training_frame,
            vocabulary,
            maximum_lag=int(settings.get("maximum_lag", 5)),
            minimum_pair_observations=int(
                settings.get("minimum_pair_observations", 20)
            ),
        )
        maximum_states = int(settings.get("maximum_model_states", 2000))
        if isinstance(trainer.model, SemOpKT):
            influence, counts = semopkt_effective_influence(
                trainer.model,
                training_sequences,
                vocabulary,
                maximum_states=maximum_states,
            )
            estimator = "direct_state_counterfactual"
        else:
            influence, counts = replay_effective_influence(
                lambda sequences: trainer.predict(sequences, score_only=True),
                training_sequences,
                len(vocabulary.texts),
                maximum_states=maximum_states,
                maximum_sequence_length=int(
                    self.model_config["training"]["max_sequence_length"]
                ),
            )
            estimator = "strict_history_replay_counterfactual"
        alignment = propagation_alignment(empirical, influence)
        np.savez_compressed(
            run_directory / "propagation_matrices.npz",
            empirical=empirical,
            influence=influence,
            source_counts=counts,
        )
        write_table(audit, run_directory / "empirical_association_audit.csv")
        record = {
            **alignment,
            "influence_estimator": estimator,
            "maximum_model_states": maximum_states,
            "empirical_edges": int(len(audit)),
        }
        write_json(run_directory / "propagation_metrics.json", record)
        return record

    def run(self, spec: RunSpec, resume: bool = True) -> dict[str, Any]:
        if spec.protocol in {
            "source_only_online",
            "source_target_adaptation",
            "multi_source_target_adaptation",
        }:
            return self._run_cross_domain(spec, resume=resume)
        run_directory = self._run_directory(spec)
        completion = run_directory / "complete.json"
        if resume:
            completed = self._validated_completion(completion, spec)
            if completed is not None:
                return completed
        run_directory.mkdir(parents=True, exist_ok=True)
        seed_everything(spec.seed, deterministic=True)
        started = time.time()
        try:
            frame = self._load_dataset(spec.dataset)
            manifest = self._manifest(frame, spec)
            split = apply_manifest(frame, manifest, spec.train_size)
            if spec.protocol == "target_only":
                if spec.train_size is None:
                    raise ValueError("Target-only transfer requires an adaptation size")
                selected = list(manifest.train_student_order[: spec.train_size])
                support, internal_validation = _support_validation_students(
                    selected,
                    float(self.config.get("cross_domain", {}).get("support_fraction", 0.80)),
                )
                selected_frame = frame[
                    frame["student_id"].astype(str).isin(set(selected))
                ]
                split.train = selected_frame[
                    selected_frame["student_id"].astype(str).isin(support)
                ].copy()
                split.validation = selected_frame[
                    selected_frame["student_id"].astype(str).isin(internal_validation)
                ].copy()
            if spec.protocol == "online_adaptation":
                split.test_target_mask = _all_target_mask(split.test, set(manifest.heldout_kcs))
            if spec.protocol in {"vocabulary_expansion", "vocabulary_full"}:
                split.test_target_mask = split.test["position"].astype(int) >= 2
            model_config = self._resolved_model_config(spec.overrides, spec.model)
            if spec.model == "A14":
                model_config["text_encoder"]["finetuning"] = "full"
            if spec.experiment == "E10" and spec.condition == "learnable_identifiers":
                model_config["architecture"]["trainable_concept_embeddings"] = True
            vocabulary, encoder_metadata = self._vocabulary(
                frame, spec.condition, spec.seed, model_config=model_config
            )
            maximum_length = int(self.model_config["training"]["max_sequence_length"])
            train_sequences = build_sequences(split.train, vocabulary, maximum_length=maximum_length)
            validation_sequences = build_sequences(
                split.validation, vocabulary, maximum_length=maximum_length
            )
            test_sequences = build_sequences(
                split.test,
                vocabulary,
                target_mask=split.test_target_mask,
                maximum_length=maximum_length,
            )
            heldout = set(manifest.heldout_kcs)
            heldout_indices = _heldout_descriptor_indices(frame, vocabulary, heldout)
            seen_indices = sorted(set(range(len(vocabulary.texts))) - heldout_indices)
            model_embeddings = vocabulary.embeddings
            model_texts = vocabulary.texts
            oov_index: int | None = None
            if heldout_indices and spec.model in _CLOSED_VOCABULARY_MODELS:
                oov_index = len(model_embeddings)
                model_embeddings = np.vstack(
                    [
                        model_embeddings,
                        np.zeros((1, model_embeddings.shape[1]), dtype=np.float32),
                    ]
                )
                model_texts = (*model_texts, "[oov kc]")
                mapping = {index: oov_index for index in heldout_indices}
                train_sequences = remap_concept_indices(train_sequences, mapping)
                validation_sequences = remap_concept_indices(validation_sequences, mapping)
                test_sequences = remap_concept_indices(test_sequences, mapping)
            model_config["dimensions"]["text"] = int(model_embeddings.shape[1])
            model = build_model(
                spec.model,
                model_config,
                self.baseline_config,
                model_embeddings,
                spec.seed,
                concept_texts=model_texts,
            )
            if hasattr(model, "configure_training_concepts"):
                model.configure_training_concepts(seen_indices)  # type: ignore[attr-defined]
            training_config = (
                model_config["training"]
                if spec.model in {"SemOpKT", *[f"A{index}" for index in range(15)]}
                else deep_merge(
                    self.baseline_config["common_training"],
                    self.baseline_config["models"].get(spec.model, {}),
                )
            )
            trainer = Trainer(model, training_config, run_directory, device=self.device)
            result = trainer.fit(
                train_sequences,
                validation_sequences,
                test_sequences,
                ece_bins=int(self.config["statistics"]["ece_bins"]),
            )
            predictions_before_graph_insertion = result.test_predictions.copy()
            if heldout_indices and spec.model in {"GKT", "A7"}:
                model.activate_unseen_concepts()  # type: ignore[attr-defined]
                result.test_predictions = trainer.predict(test_sequences, score_only=True)
            if spec.experiment == "E20" and isinstance(model, SemOpKT):
                write_table(
                    trace_semopkt_targets(model, test_sequences, vocabulary),
                    run_directory / "field_traces.csv",
                )
            propagation_metadata: dict[str, Any] | None = None
            if spec.experiment == "E13":
                propagation_metadata = self._write_propagation_analysis(
                    run_directory,
                    split.train,
                    train_sequences,
                    vocabulary,
                    trainer,
                )
            adaptation_metadata: dict[str, Any] | None = None
            if spec.experiment == "E7":
                settings = self.config.get("online_adaptation", {})
                frequencies = split.train["kc_text_norm"].astype(str).value_counts()
                old_queries = [
                    vocabulary.text_to_index[text]
                    for text in frequencies.index
                    if vocabulary.text_to_index[text] in set(seen_indices)
                ][: int(settings.get("old_query_concepts", 16))]
                stability = old_concept_stability_replay(
                    test_sequences,
                    heldout_indices,
                    old_queries,
                    lambda sequences: trainer.predict(sequences, score_only=True),
                    maximum_events=int(
                        settings.get("maximum_stability_events", 1000)
                    ),
                    maximum_sequence_length=maximum_length,
                )
                if stability.empty:
                    raise ValueError("Online-adaptation stability analysis produced no events")
                write_table(stability, run_directory / "old_concept_stability.csv")
                stability_summary = (
                    stability.groupby("prior_target_observations", as_index=False)
                    .agg(
                        events=("source_row_id", "size"),
                        old_concept_mean_absolute_drift=(
                            "old_concept_mean_absolute_drift",
                            "mean",
                        ),
                    )
                    .sort_values("prior_target_observations")
                )
                write_table(
                    stability_summary,
                    run_directory / "old_concept_stability_summary.csv",
                )
                adaptation_metadata = {
                    "stability_events": int(len(stability)),
                    "old_query_concepts": len(old_queries),
                    "mean_old_concept_drift": float(
                        stability["old_concept_mean_absolute_drift"].mean()
                    ),
                }
            robustness_metadata: dict[str, Any] | None = None
            if spec.experiment == "E14":
                clean_predictions = result.test_predictions.copy()
                write_table(clean_predictions, run_directory / "predictions_clean.csv")
                result.test_predictions, robustness_metadata = self._robustness_predictions(
                    spec,
                    frame,
                    vocabulary,
                    model_config,
                    test_sequences,
                    trainer,
                    clean_predictions,
                )
            uncalibrated = prediction_covariates(
                split.train,
                split.test,
                result.test_predictions.copy(),
                vocabulary,
            )
            uncalibrated["calibration_status"] = "uncalibrated"
            calibrated = uncalibrated.copy()
            temperature: float | None = None
            calibration_allowed = not spec.protocol.startswith("source_")
            if calibration_allowed:
                scaler = TemperatureScaler.fit(
                    result.validation_predictions["logit"].to_numpy(),
                    result.validation_predictions["correct"].to_numpy(),
                )
                calibrated["probability"] = scaler.probabilities(calibrated["logit"].to_numpy())
                calibrated["calibration_status"] = "validation_temperature"
                temperature = scaler.temperature
            target_lookup = split.test[["source_row_id", "kc_components"]].drop_duplicates("source_row_id")
            seen_map = {
                str(row.source_row_id): (
                    "unseen" if interaction_contains_heldout(row.kc_components, heldout) else "seen"
                )
                for row in target_lookup.itertuples(index=False)
            }
            for predictions in (uncalibrated, calibrated):
                predictions.insert(0, "experiment", spec.experiment)
                predictions.insert(1, "protocol", spec.protocol)
                predictions.insert(2, "model", spec.model)
                predictions.insert(3, "seed", spec.seed)
                predictions.insert(4, "condition", spec.condition)
                predictions["train_size"] = spec.train_size
                predictions["holdout_ratio"] = spec.holdout_ratio
                predictions["source_dataset"] = spec.source_dataset
                predictions["target_dataset"] = spec.target_dataset
                predictions["adaptation_size"] = spec.adaptation_size
                predictions["seen_status"] = predictions["source_row_id"].map(seen_map).fillna("mixed")
            metrics_uncalibrated = compute_metrics(
                uncalibrated, ece_bins=int(self.config["statistics"]["ece_bins"])
            )
            metrics_calibrated = compute_metrics(
                calibrated, ece_bins=int(self.config["statistics"]["ece_bins"])
            )
            targets_per_student = calibrated.groupby("student_id").size()
            target_audit = {
                "effective_students": int(calibrated["student_id"].nunique()),
                "effective_concepts": int(calibrated["kc_id"].nunique()),
                "targets": int(len(calibrated)),
                "positive_targets": int(calibrated["correct"].sum()),
                "negative_targets": int(
                    len(calibrated) - calibrated["correct"].sum()
                ),
                "targets_per_student_minimum": int(targets_per_student.min()),
                "targets_per_student_median": float(targets_per_student.median()),
                "targets_per_student_p95": float(
                    np.quantile(targets_per_student.to_numpy(dtype=float), 0.95)
                ),
                "primary_auc_reporting_allowed": bool(
                    not spec.protocol.startswith("double_") or len(calibrated) >= 200
                ),
            }
            write_json(run_directory / "target_audit.json", target_audit)
            expansion_metadata: dict[str, Any] | None = None
            if spec.experiment == "E12" and spec.protocol in {
                "vocabulary_expansion",
                "vocabulary_full",
            }:
                seen_predictions = uncalibrated[uncalibrated["seen_status"] == "seen"]
                added_predictions = uncalibrated[uncalibrated["seen_status"] == "unseen"]
                expansion_metadata = {
                    "old_concept_metrics": compute_metrics(seen_predictions)
                    if not seen_predictions.empty
                    else None,
                    "added_concept_metrics": compute_metrics(added_predictions)
                    if not added_predictions.empty
                    else None,
                    "querying_added_descriptors_requires_global_refitting": False,
                    "evaluation_users_disjoint_from_global_fit": True,
                }
                if isinstance(model, SemOpKT) and heldout_indices and seen_indices:
                    expansion_batch = trainer.prepare_batch(
                        collate_sequences([test_sequences[0]])
                    )
                    with torch.inference_mode():
                        expansion_state = model(expansion_batch).auxiliary[
                            "final_state"
                        ][:1]
                    device = next(model.parameters()).device
                    query_drift = query_only_expansion_drift(
                        model,
                        expansion_state,
                        torch.as_tensor(
                            seen_indices, dtype=torch.long, device=device
                        ),
                        torch.as_tensor(
                            sorted(heldout_indices),
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    expansion_metadata["query_only_drift"] = query_drift
                    expansion_metadata["requires_graph_insertion"] = False
                elif spec.model == "GKT" and heldout_indices:
                    before = predictions_before_graph_insertion[
                        ["source_row_id", "probability"]
                    ].rename(columns={"probability": "probability_before_insertion"})
                    drift = uncalibrated.merge(
                        before,
                        on="source_row_id",
                        how="inner",
                        validate="one_to_one",
                    )
                    drift = drift[drift["seen_status"] == "seen"]
                    expansion_metadata[
                        "graph_insertion_old_concept_mean_absolute_drift"
                    ] = float(
                        (
                            drift["probability"]
                            - drift["probability_before_insertion"]
                        ).abs().mean()
                    )
                    expansion_metadata["requires_graph_insertion"] = True
                else:
                    expansion_metadata["requires_graph_insertion"] = False
                write_json(run_directory / "expansion_metrics.json", expansion_metadata)
            efficiency_metadata: dict[str, Any] | None = None
            if test_sequences:
                timing = self.config["hardware_timing"]
                requested_length = int(timing["sequence_length"])
                timing_sequence = max(
                    test_sequences,
                    key=lambda sequence: min(len(sequence.labels), requested_length),
                )
                batch = trainer.prepare_batch(collate_sequences([timing_sequence]))
                sequence_timing = measure_inference(
                    model,
                    batch,
                    warmup=int(timing["warmup"]),
                    repetitions=int(timing["repetitions"]),
                )
                timing_external_parameters = (
                    int(encoder_metadata["encoder"].get("total_parameters", 0))
                    if getattr(model, "uses_semantic_descriptors", False)
                    and getattr(model, "descriptor_encoder", None) is None
                    else 0
                )
                sequence_timing["model_parameters"] = model.total_parameter_count()
                sequence_timing["external_encoder_parameters"] = timing_external_parameters
                sequence_timing["total_parameters"] = (
                    model.total_parameter_count() + timing_external_parameters
                )
                query_scaling = (
                    measure_semopkt_query_scaling(
                        model,
                        [int(value) for value in self.config["query_sizes"]],
                        warmup=int(timing["warmup"]),
                        repetitions=int(timing["repetitions"]),
                    )
                    if spec.experiment == "E18" and isinstance(model, SemOpKT)
                    else []
                )
                efficiency_metadata = {
                    "sequence_timing": sequence_timing,
                    "query_scaling": query_scaling,
                    "mean_epoch_seconds": float(
                        result.train_seconds / max(1, len(result.history))
                    ),
                    "total_train_seconds": result.train_seconds,
                    "measured_sequence_length": len(timing_sequence.labels),
                    "cached_text_embeddings": True,
                    "first_time_text_encoding_included": False,
                }
                write_json(run_directory / "efficiency.json", efficiency_metadata)
            write_table(uncalibrated, run_directory / "predictions_uncalibrated.csv")
            write_table(calibrated, run_directory / "predictions.csv")
            write_table(result.validation_predictions, run_directory / "validation_predictions.csv")
            write_table(
                grouped_metrics(
                    calibrated,
                    {"student_id": calibrated["student_id"]},
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                ),
                run_directory / "student_metrics.csv",
            )
            write_table(
                grouped_metrics(
                    calibrated,
                    {"kc_id": calibrated["kc_id"]},
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                ),
                run_directory / "concept_metrics.csv",
            )
            if spec.experiment == "E3":
                write_table(
                    metrics_by_history_window(
                        calibrated, ece_bins=int(self.config["statistics"]["ece_bins"])
                    ),
                    run_directory / "history_window_metrics.csv",
                )
            if spec.experiment == "E7":
                prior_attempts = calibrated["prior_same_kc"].to_numpy(dtype=int)
                attempts = pd.Series(
                    np.select(
                        [
                            prior_attempts == 0,
                            prior_attempts == 1,
                            prior_attempts == 2,
                            prior_attempts >= 4,
                        ],
                        ["0", "1", "2", "4+"],
                        default=None,
                    ),
                    index=calibrated.index,
                    dtype="object",
                )
                write_table(
                    grouped_metrics(
                        calibrated,
                        {"prior_same_kc": attempts},
                        ece_bins=int(self.config["statistics"]["ece_bins"]),
                    ),
                    run_directory / "online_adaptation_metrics.csv",
                )
            if spec.experiment == "E19":
                group_columns = (
                    "semantic_distance_group",
                    "frequency_group",
                    "history_length_group",
                    "history_accuracy_group",
                    "target_label_group",
                    "first_encounter_group",
                )
                write_table(
                    grouped_metrics(
                        calibrated,
                        {column: calibrated[column] for column in group_columns},
                        ece_bins=int(self.config["statistics"]["ece_bins"]),
                    ),
                    run_directory / "group_metrics.csv",
                )
                write_table(
                    high_confidence_errors(calibrated),
                    run_directory / "high_confidence_errors.csv",
                )
                write_table(
                    classify_error_cases(calibrated),
                    run_directory / "classified_errors.csv",
                )
            resolved_run = {
                "specification": spec.to_dict(),
                "model": model_config,
                "training": training_config,
                "manifest_hash": manifest.manifest_hash,
                "encoder": encoder_metadata,
                "closed_vocabulary_oov": {
                    "enabled": oov_index is not None,
                    "mapped_descriptor_count": len(heldout_indices) if oov_index is not None else 0,
                    "oov_index": oov_index,
                },
                "parameter_matching": {
                    "target_parameters": getattr(
                        model, "parameter_budget_target", None
                    ),
                    "selected_capacity": getattr(
                        model, "parameter_budget_value", None
                    ),
                    "relative_error": getattr(
                        model, "parameter_budget_relative_error", None
                    ),
                },
            }
            write_yaml(run_directory / "resolved_config.yaml", resolved_run)
            bound_inputs = {
                "configuration": config_hash(resolved_run),
                "manifest": manifest.manifest_hash,
                "source_tree": self.source_tree_hash,
                "encoder": str(encoder_metadata["cache_key"]),
            }
            if robustness_metadata and robustness_metadata.get(
                "perturbation_encoder_cache_key"
            ):
                bound_inputs["perturbation_encoder"] = str(
                    robustness_metadata["perturbation_encoder_cache_key"]
                )
            bound_hash = bind_hashes(bound_inputs)
            record = {
                "status": "complete",
                "run_key": spec.key,
                "bound_hash": bound_hash,
                "specification": spec.to_dict(),
                "manifest_hash": manifest.manifest_hash,
                "source_table_hash": hash_dataframe(frame),
                "source_tree_hash": self.source_tree_hash,
                "encoder": encoder_metadata,
                "robustness": robustness_metadata,
                "propagation": propagation_metadata,
                "expansion": expansion_metadata,
                "efficiency": efficiency_metadata,
                "online_adaptation": adaptation_metadata,
                "target_audit": target_audit,
                "counts": {
                    "train_students": len(train_sequences),
                    "validation_students": len(validation_sequences),
                    "test_students": len(test_sequences),
                    "train_interactions": int(len(split.train)),
                    "validation_interactions": int(len(split.validation)),
                    "scored_test_interactions": int(len(calibrated)),
                },
                "best_epoch": result.best_epoch,
                "checkpoint_sha256": result.checkpoint_sha256,
                "train_seconds": result.train_seconds,
                "peak_memory_bytes": result.peak_memory_bytes,
                "model_parameters": model.total_parameter_count(),
                "external_encoder_parameters": (
                    int(encoder_metadata["encoder"].get("total_parameters", 0))
                    if getattr(model, "uses_semantic_descriptors", False)
                    and getattr(model, "descriptor_encoder", None) is None
                    else 0
                ),
                "total_parameters": model.total_parameter_count()
                + (
                    int(encoder_metadata["encoder"].get("total_parameters", 0))
                    if getattr(model, "uses_semantic_descriptors", False)
                    and getattr(model, "descriptor_encoder", None) is None
                    else 0
                ),
                "trainable_parameters": model.trainable_parameter_count(),
                "parameter_matching": resolved_run["parameter_matching"],
                "temperature": temperature,
                "metrics_uncalibrated": metrics_uncalibrated,
                "metrics_calibrated": metrics_calibrated,
                "environment": environment_record(self.repository),
                "started_unix": started,
                "completed_unix": time.time(),
            }
            record["artifact_hashes"] = _artifact_hashes(run_directory)
            write_json(run_directory / "run.json", record)
            write_json(completion, record)
            return record
        except Exception as error:
            failure = {
                "status": "failed",
                "run_key": spec.key,
                "specification": spec.to_dict(),
                "exception": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "failed_unix": time.time(),
            }
            write_json(run_directory / "run.json", failure)
            raise

    def _remove_cross_domain_duplicates(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        threshold = float(
            self.config.get("cross_domain", {}).get(
                "near_duplicate_cosine_threshold", 0.92
            )
        )
        combined = pd.concat([source, target], ignore_index=True)
        lookup, metadata = self._encoder_and_lookup(combined, split=True)
        source_texts = sorted(source["kc_text_norm"].astype(str).unique())
        target_texts = sorted(target["kc_text_norm"].astype(str).unique())
        source_matrix = np.vstack([lookup[text] for text in source_texts])
        target_matrix = np.vstack([lookup[text] for text in target_texts])
        similarities = source_matrix @ target_matrix.T
        maximum = similarities.max(axis=1)
        removed = {
            text
            for text, similarity in zip(source_texts, maximum, strict=True)
            if float(similarity) >= threshold or text in set(target_texts)
        }
        filtered = source[~source["kc_text_norm"].astype(str).isin(removed)].copy()
        if filtered.empty:
            raise ValueError("Cross-domain de-duplication removed the complete source domain")
        return filtered, {
            "threshold": threshold,
            "removed_concepts": sorted(removed),
            "removed_concept_count": len(removed),
            "removed_interactions": int(len(source) - len(filtered)),
            "split_encoder_cache_key": metadata["cache_key"],
        }

    @staticmethod
    def _namespace_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
        result = frame.copy()
        prefix = f"{dataset}::"
        result["student_id"] = prefix + result["student_id"].astype(str)
        result["source_row_id"] = prefix + result["source_row_id"].astype(str)
        return result

    def _run_cross_domain(self, spec: RunSpec, resume: bool) -> dict[str, Any]:
        if spec.source_dataset is None or spec.target_dataset is None:
            raise ValueError("Cross-domain specification requires source and target datasets")
        run_directory = self._run_directory(spec)
        completion = run_directory / "complete.json"
        if resume:
            completed = self._validated_completion(completion, spec)
            if completed is not None:
                return completed
        run_directory.mkdir(parents=True, exist_ok=True)
        started = time.time()
        seed_everything(spec.seed, deterministic=True)
        try:
            source_names = spec.source_dataset.split("+")
            target = self._load_dataset(spec.target_dataset).copy()
            target_spec = RunSpec(
                spec.experiment,
                spec.target_dataset,
                "system",
                spec.model,
                spec.seed,
            )
            target_manifest = self._manifest(target, target_spec)
            target_split = apply_manifest(target, target_manifest, None)
            source_manifests: dict[str, SplitManifest] = {}
            source_table_hashes: dict[str, str] = {}
            source_training_frames: list[pd.DataFrame] = []
            source_validation_frames: list[pd.DataFrame] = []
            vocabulary_source_frames: list[pd.DataFrame] = []
            duplicate_audit: dict[str, Any] = {}
            for source_name in source_names:
                source = self._load_dataset(source_name).copy()
                source_table_hashes[source_name] = hash_dataframe(source)
                source_spec = RunSpec(
                    spec.experiment,
                    source_name,
                    "system",
                    spec.model,
                    spec.seed,
                )
                source_manifest = self._manifest(source, source_spec)
                source_manifests[source_name] = source_manifest
                source_split = apply_manifest(source, source_manifest, None)
                if spec.condition == "near_duplicates_removed":
                    filtered_source, audit = self._remove_cross_domain_duplicates(
                        source, target
                    )
                    retained = set(filtered_source["source_row_id"].astype(str))
                    source_split.train = source_split.train[
                        source_split.train["source_row_id"].astype(str).isin(retained)
                    ].copy()
                    source_split.validation = source_split.validation[
                        source_split.validation["source_row_id"].astype(str).isin(retained)
                    ].copy()
                    source = filtered_source
                    duplicate_audit[source_name] = audit
                source_training_frames.append(
                    self._namespace_frame(source_split.train, source_name)
                )
                source_validation_frames.append(
                    self._namespace_frame(source_split.validation, source_name)
                )
                vocabulary_source_frames.append(source)
            if any(frame.empty for frame in source_training_frames + source_validation_frames):
                raise ValueError("A source split became empty after cross-domain filtering")

            target_test = self._namespace_frame(target_split.test, spec.target_dataset)
            support_frame: pd.DataFrame | None = None
            adaptation_validation_frame: pd.DataFrame | None = None
            if spec.protocol != "source_only_online":
                if spec.adaptation_size is None:
                    raise ValueError("Source-to-target adaptation requires a target size")
                selected = list(
                    target_manifest.train_student_order[: spec.adaptation_size]
                )
                support_students, validation_students = _support_validation_students(
                    selected,
                    float(
                        self.config.get("cross_domain", {}).get(
                            "support_fraction", 0.80
                        )
                    ),
                )
                selected_frame = target[
                    target["student_id"].astype(str).isin(set(selected))
                ]
                support_frame = self._namespace_frame(
                    selected_frame[
                        selected_frame["student_id"].astype(str).isin(
                            support_students
                        )
                    ],
                    spec.target_dataset,
                )
                adaptation_validation_frame = self._namespace_frame(
                    selected_frame[
                        selected_frame["student_id"].astype(str).isin(
                            validation_students
                        )
                    ],
                    spec.target_dataset,
                )

            combined = pd.concat(
                [*vocabulary_source_frames, target], ignore_index=True
            )
            model_config = self._resolved_model_config(spec.overrides, spec.model)
            vocabulary, encoder_metadata = self._vocabulary(
                combined, "default", spec.seed, model_config=model_config
            )
            maximum_length = int(model_config["training"]["max_sequence_length"])
            source_train = pd.concat(source_training_frames, ignore_index=True)
            source_validation = pd.concat(source_validation_frames, ignore_index=True)
            source_train_sequences = build_sequences(
                source_train, vocabulary, maximum_length=maximum_length
            )
            source_validation_sequences = build_sequences(
                source_validation, vocabulary, maximum_length=maximum_length
            )
            model_config["dimensions"]["text"] = int(vocabulary.embeddings.shape[1])
            model = build_model(
                spec.model,
                model_config,
                self.baseline_config,
                vocabulary.embeddings,
                spec.seed,
                concept_texts=vocabulary.texts,
            )
            source_descriptor_indices = sorted(
                {
                    vocabulary.text_to_index[text]
                    for text in source_train["kc_text_norm"].astype(str).unique()
                }
            )
            if hasattr(model, "configure_training_concepts"):
                model.configure_training_concepts(  # type: ignore[attr-defined]
                    source_descriptor_indices
                )
            training_config = (
                model_config["training"]
                if spec.model == "SemOpKT"
                else deep_merge(
                    self.baseline_config["common_training"],
                    self.baseline_config["models"].get(spec.model, {}),
                )
            )
            pretrainer = Trainer(
                model,
                training_config,
                run_directory / "source_pretrain",
                device=self.device,
            )
            pretrain_result = pretrainer.fit(
                source_train_sequences,
                source_validation_sequences,
                source_validation_sequences,
                ece_bins=int(self.config["statistics"]["ece_bins"]),
            )
            if hasattr(model, "activate_unseen_concepts"):
                model.activate_unseen_concepts()  # type: ignore[attr-defined]
            target_sequences = build_sequences(
                target_test,
                vocabulary,
                target_mask=target_test["position"].astype(int) >= 2,
                maximum_length=maximum_length,
            )
            result = pretrain_result
            final_trainer = pretrainer
            fit_frame = source_train
            target_temperature_fitted = False
            if support_frame is not None and adaptation_validation_frame is not None:
                support_sequences = build_sequences(
                    support_frame, vocabulary, maximum_length=maximum_length
                )
                adaptation_validation_sequences = build_sequences(
                    adaptation_validation_frame,
                    vocabulary,
                    maximum_length=maximum_length,
                )
                adaptation_config = deep_merge(
                    training_config,
                    self.config.get("cross_domain", {}).get(
                        "adaptation_training", {}
                    ),
                )
                final_trainer = Trainer(
                    model, adaptation_config, run_directory, device=self.device
                )
                result = final_trainer.fit(
                    support_sequences,
                    adaptation_validation_sequences,
                    target_sequences,
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                )
                fit_frame = pd.concat(
                    [source_train, support_frame], ignore_index=True
                )
                target_temperature_fitted = True
            else:
                result.test_predictions = pretrainer.predict(
                    target_sequences, score_only=True
                )
                result.test_metrics = compute_metrics(
                    result.test_predictions,
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                )

            uncalibrated = prediction_covariates(
                fit_frame,
                target_test,
                result.test_predictions.copy(),
                vocabulary,
            )
            uncalibrated["calibration_status"] = "uncalibrated_target"
            calibrated = uncalibrated.copy()
            temperature: float | None = None
            if target_temperature_fitted:
                scaler = TemperatureScaler.fit(
                    result.validation_predictions["logit"].to_numpy(),
                    result.validation_predictions["correct"].to_numpy(),
                )
                calibrated["probability"] = scaler.probabilities(
                    calibrated["logit"].to_numpy()
                )
                calibrated["calibration_status"] = "target_support_temperature"
                temperature = scaler.temperature
            for predictions in (uncalibrated, calibrated):
                predictions.insert(0, "experiment", spec.experiment)
                predictions.insert(1, "protocol", spec.protocol)
                predictions.insert(2, "model", spec.model)
                predictions.insert(3, "seed", spec.seed)
                predictions.insert(4, "condition", spec.condition)
                predictions["train_size"] = spec.train_size
                predictions["holdout_ratio"] = spec.holdout_ratio
                predictions["source_dataset"] = spec.source_dataset
                predictions["target_dataset"] = spec.target_dataset
                predictions["adaptation_size"] = spec.adaptation_size
                predictions["seen_status"] = "target_domain"
            timing = self.config["hardware_timing"]
            timing_sequence = max(target_sequences, key=lambda sequence: len(sequence.labels))
            inference_timing = measure_inference(
                model,
                final_trainer.prepare_batch(collate_sequences([timing_sequence])),
                warmup=int(timing["warmup"]),
                repetitions=int(timing["repetitions"]),
            )
            inference_external_parameters = (
                int(encoder_metadata["encoder"].get("total_parameters", 0))
                if getattr(model, "uses_semantic_descriptors", False)
                and getattr(model, "descriptor_encoder", None) is None
                else 0
            )
            inference_timing["model_parameters"] = model.total_parameter_count()
            inference_timing["external_encoder_parameters"] = inference_external_parameters
            inference_timing["total_parameters"] = (
                model.total_parameter_count() + inference_external_parameters
            )
            write_table(
                uncalibrated, run_directory / "predictions_uncalibrated.csv"
            )
            write_table(calibrated, run_directory / "predictions.csv")
            write_table(
                result.validation_predictions,
                run_directory / "validation_predictions.csv",
            )
            write_table(
                grouped_metrics(
                    calibrated,
                    {"student_id": calibrated["student_id"]},
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                ),
                run_directory / "student_metrics.csv",
            )
            write_table(
                grouped_metrics(
                    calibrated,
                    {"kc_id": calibrated["kc_id"]},
                    ece_bins=int(self.config["statistics"]["ece_bins"]),
                ),
                run_directory / "concept_metrics.csv",
            )
            write_json(run_directory / "cross_domain_dedup.json", duplicate_audit)
            resolved_run = {
                "specification": spec.to_dict(),
                "model": model_config,
                "training": training_config,
                "source_manifests": {
                    name: manifest.manifest_hash
                    for name, manifest in source_manifests.items()
                },
                "target_manifest": target_manifest.manifest_hash,
                "encoder": encoder_metadata,
                "duplicate_audit": duplicate_audit,
                "vocabulary_boundary": {
                    "source_training_descriptors": len(source_descriptor_indices),
                    "target_descriptors_available_for_query": int(
                        target["kc_text_norm"].astype(str).nunique()
                    ),
                    "field_or_graph_initialized_from_source_only": True,
                },
            }
            write_yaml(run_directory / "resolved_config.yaml", resolved_run)
            bound_hash = bind_hashes(
                {
                    "configuration": config_hash(resolved_run),
                    "source_tree": self.source_tree_hash,
                    "encoder": str(encoder_metadata["cache_key"]),
                    **{
                        f"source_manifest_{name}": manifest.manifest_hash
                        for name, manifest in source_manifests.items()
                    },
                    "target_manifest": target_manifest.manifest_hash,
                }
            )
            external_encoder_parameters = (
                int(encoder_metadata["encoder"].get("total_parameters", 0))
                if getattr(model, "uses_semantic_descriptors", False)
                and getattr(model, "descriptor_encoder", None) is None
                else 0
            )
            record = {
                "status": "complete",
                "run_key": spec.key,
                "bound_hash": bound_hash,
                "specification": spec.to_dict(),
                "source_manifest_hashes": {
                    name: manifest.manifest_hash
                    for name, manifest in source_manifests.items()
                },
                "target_manifest_hash": target_manifest.manifest_hash,
                "source_table_hashes": source_table_hashes,
                "target_table_hash": hash_dataframe(target),
                "encoder": encoder_metadata,
                "duplicate_audit": duplicate_audit,
                "vocabulary_boundary": resolved_run["vocabulary_boundary"],
                "inference_timing": inference_timing,
                "best_epoch": result.best_epoch,
                "source_pretrain_best_epoch": pretrain_result.best_epoch,
                "metrics_uncalibrated": compute_metrics(uncalibrated),
                "target_temperature_fitted": target_temperature_fitted,
                "target_response_access": {
                    "during_source_pretraining": False,
                    "during_source_model_selection": False,
                    "during_target_support_adaptation": bool(support_frame is not None),
                    "test_labels_used_for_fitting_or_selection": False,
                    "test_predictions_generated_after_model_selection": True,
                },
                "temperature": temperature,
                "model_parameters": model.total_parameter_count(),
                "external_encoder_parameters": external_encoder_parameters,
                "total_parameters": model.total_parameter_count()
                + external_encoder_parameters,
                "trainable_parameters": model.trainable_parameter_count(),
                "checkpoint_sha256": result.checkpoint_sha256,
                "source_pretrain_checkpoint_sha256": pretrain_result.checkpoint_sha256,
                "train_seconds": result.train_seconds + pretrain_result.train_seconds
                if result is not pretrain_result
                else result.train_seconds,
                "peak_memory_bytes": max(
                    result.peak_memory_bytes, pretrain_result.peak_memory_bytes
                ),
                "source_tree_hash": self.source_tree_hash,
                "environment": environment_record(self.repository),
                "started_unix": started,
                "completed_unix": time.time(),
            }
            if target_temperature_fitted:
                record["metrics_calibrated"] = compute_metrics(calibrated)
            record["artifact_hashes"] = _artifact_hashes(run_directory)
            write_json(run_directory / "run.json", record)
            write_json(completion, record)
            return record
        except Exception as error:
            failure = {
                "status": "failed",
                "run_key": spec.key,
                "specification": spec.to_dict(),
                "exception": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "failed_unix": time.time(),
            }
            write_json(run_directory / "run.json", failure)
            raise
