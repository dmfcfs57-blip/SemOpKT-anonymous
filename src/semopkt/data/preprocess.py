"""Dataset-agnostic preprocessing with explicit aliases and audit records."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from semopkt.config import load_config, require
from semopkt.constants import PREPROCESS_VERSION, STANDARD_COLUMNS
from semopkt.data.schema import (
    canonical_descriptor,
    normalize_text,
    resolve_column,
    split_components,
    stable_identifier,
    validate_standard_frame,
)
from semopkt.utils.hashing import hash_dataframe, hash_file
from semopkt.utils.io import write_json, write_table


def _discover_raw_file(raw_root: Path, globs: list[str]) -> Path:
    for pattern in globs:
        candidates = sorted(
            set(path.resolve() for path in raw_root.glob(pattern) if path.is_file())
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Raw-file pattern {pattern!r} is ambiguous below {raw_root}: "
                f"{[path.name for path in candidates]}"
            )
    raise FileNotFoundError(f"No raw file found below {raw_root} for patterns {globs}")


def _metadata_file(raw_root: Path, relative: str) -> Path:
    direct = (raw_root / relative).resolve()
    if direct.is_file():
        return direct
    candidates = sorted(path.resolve() for path in raw_root.rglob(Path(relative).name))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one metadata file {relative!r} below {raw_root}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_raw(path: Path, separator: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).reset_index(drop=True)
    return pd.read_csv(path, sep=separator, low_memory=False).reset_index(drop=True)


def _parse_identifier_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        parsed = list(value)
    elif pd.isna(value):
        return []
    else:
        try:
            candidate = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            candidate = [part for part in str(value).replace(";", ",").split(",")]
        parsed = list(candidate) if isinstance(candidate, (list, tuple, set)) else [candidate]
    return [str(item).strip() for item in parsed if str(item).strip()]


def _read_nips_with_metadata(
    primary_path: Path,
    raw_root: Path,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    answer_path = _metadata_file(raw_root, str(settings["answer_metadata"]))
    question_path = _metadata_file(raw_root, str(settings["question_metadata"]))
    subject_path = _metadata_file(raw_root, str(settings["subject_metadata"]))
    primary = pd.read_csv(primary_path, low_memory=False).reset_index(drop=True)
    answers = pd.read_csv(answer_path, low_memory=False)
    questions = pd.read_csv(question_path, low_memory=False)
    subjects = pd.read_csv(subject_path, low_memory=False)
    answer_key = str(settings.get("answer_key", "AnswerId"))
    question_key = str(settings.get("question_key", "QuestionId"))
    timestamp_column = str(settings.get("timestamp_column", "DateAnswered"))
    subject_id_column = str(settings.get("subject_id_column", "SubjectId"))
    subject_name_column = str(settings.get("subject_name_column", "Name"))
    subject_level_column = str(settings.get("subject_level_column", "Level"))
    target_level = int(settings.get("target_subject_level", 3))
    for table, key, name in (
        (primary, answer_key, "primary answer"),
        (answers, answer_key, "answer metadata"),
        (primary, question_key, "primary question"),
        (questions, question_key, "question metadata"),
    ):
        if key not in table:
            raise KeyError(f"Missing {key} in {name} table")
    required_subject = {
        subject_id_column,
        subject_name_column,
        subject_level_column,
    }
    if not required_subject.issubset(subjects.columns):
        raise KeyError(
            f"Subject metadata is missing {sorted(required_subject - set(subjects.columns))}"
        )
    if timestamp_column not in answers or subject_id_column not in questions:
        raise KeyError("NIPS metadata lacks timestamp or question-subject columns")
    for table, key in ((primary, answer_key), (answers, answer_key)):
        table[key] = table[key].astype("string")
    for table, key in ((primary, question_key), (questions, question_key)):
        table[key] = table[key].astype("string")
    if answers[answer_key].duplicated().any() or questions[question_key].duplicated().any():
        raise ValueError("NIPS answer/question metadata keys must be unique")
    level_mask = pd.to_numeric(subjects[subject_level_column], errors="coerce").eq(
        target_level
    )
    level_subjects = subjects.loc[level_mask].copy()
    level_subjects[subject_id_column] = level_subjects[subject_id_column].astype(
        "string"
    )
    if level_subjects[subject_id_column].duplicated().any():
        raise ValueError("NIPS level-specific subject identifiers must be unique")
    subject_names = dict(
        zip(
            level_subjects[subject_id_column].astype(str),
            level_subjects[subject_name_column].astype(str),
            strict=True,
        )
    )

    def level_identifiers(value: object) -> list[str]:
        return [
            identifier
            for identifier in _parse_identifier_list(value)
            if identifier in subject_names
        ]

    question_subjects = questions[[question_key, subject_id_column]].copy()
    question_subjects["SubjectIdLevel3List"] = question_subjects[
        subject_id_column
    ].map(level_identifiers)
    question_subjects["SubjectIdLevel3"] = question_subjects[
        "SubjectIdLevel3List"
    ].map(lambda values: "~~".join(values) if values else pd.NA)
    question_subjects["SubjectName"] = question_subjects[
        "SubjectIdLevel3List"
    ].map(
        lambda values: "~~".join(subject_names[value] for value in values)
        if values
        else pd.NA
    )
    merged = primary.merge(
        answers[[answer_key, timestamp_column]],
        on=answer_key,
        how="left",
        validate="many_to_one",
    ).merge(
        question_subjects[[question_key, "SubjectIdLevel3", "SubjectName"]],
        on=question_key,
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(primary):
        raise ValueError("NIPS metadata merge changed the interaction count")
    return (
        merged.reset_index(drop=True),
        [primary_path, answer_path, question_path, subject_path],
        {
            "metadata_join": "answer_timestamp+question_level3_subjects",
            "target_subject_level": target_level,
            "level3_subjects": int(len(subject_names)),
        },
    )


def _optional_column(
    columns: pd.Index, aliases: list[str], field: str
) -> str | None:
    try:
        return resolve_column(columns, aliases, field)
    except KeyError:
        return None


def _question_values(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    components = config.get("question_id_components")
    if not components:
        column = resolve_column(
            raw.columns, config["column_aliases"]["question_id"], "question_id"
        )
        return raw[column].astype("string")
    resolved = [
        resolve_column(raw.columns, list(aliases), f"question_id_components[{index}]")
        for index, aliases in enumerate(components)
    ]
    values = raw[resolved[0]].astype("string")
    for column in resolved[1:]:
        values = values.str.cat(raw[column].astype("string"), sep=" :: ")
    return values


def _binary_label(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    text = normalize_text(value)
    mapping = {
        "1": 1.0,
        "1.0": 1.0,
        "true": 1.0,
        "correct": 1.0,
        "yes": 1.0,
        "0": 0.0,
        "0.0": 0.0,
        "false": 0.0,
        "incorrect": 0.0,
        "no": 0.0,
    }
    return mapping.get(text, np.nan)


def _statistics(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.groupby("student_id", sort=False).size()
    concept_counts = frame.groupby("student_id", sort=False)["kc_id"].nunique()
    return {
        "students": int(frame["student_id"].nunique()),
        "interactions": int(len(frame)),
        "questions": int(frame["question_id"].nunique()),
        "concepts": int(frame["kc_id"].nunique()),
        "median_interactions_per_student": float(counts.median()),
        "median_concepts_per_student": float(concept_counts.median()),
        "positive_rate": float(frame["correct"].mean()),
    }


def preprocess_dataset(
    config: Mapping[str, Any] | str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    allow_statistic_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_config = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    require(data_config, "dataset", "raw_globs", "column_aliases")
    root = Path(raw_root).resolve()
    raw_path = _discover_raw_file(root, list(data_config["raw_globs"]))
    if data_config.get("nips_metadata"):
        raw, raw_input_paths, source_transform = _read_nips_with_metadata(
            raw_path, root, data_config["nips_metadata"]
        )
    else:
        raw = _read_raw(raw_path, str(data_config.get("separator", ",")))
        raw_input_paths = [raw_path]
        source_transform = {"metadata_join": "none"}
    aliases = data_config["column_aliases"]
    required = {
        key: resolve_column(raw.columns, aliases[key], key)
        for key in ("student_id", "kc_id", "correct")
    }
    question_values = _question_values(raw, data_config)
    text_column = resolve_column(raw.columns, aliases.get("kc_text", aliases["kc_id"]), "kc_text")
    timestamp_column = _optional_column(
        raw.columns, list(aliases.get("timestamp", [])), "timestamp"
    )
    exact_duplicate_mask = raw.duplicated(keep="first")
    working = raw.loc[~exact_duplicate_mask].copy()
    working_questions = question_values.loc[working.index]
    if timestamp_column is None:
        timestamps = pd.Series(
            working.index.astype(str), index=working.index, dtype="string"
        )
    else:
        timestamps = working[timestamp_column].astype("string")
    source = pd.DataFrame(
        {
            "dataset": str(data_config["dataset"]),
            "student_id": working[required["student_id"]].astype("string"),
            "question_id": working_questions,
            "kc_source": working[required["kc_id"]],
            "kc_text_source": working[text_column],
            "correct": working[required["correct"]].map(_binary_label),
            "timestamp": timestamps,
            "source_row_id": [f"row_{index}" for index in working.index],
            "_source_order": working.index.to_numpy(dtype=np.int64),
        }
    )
    input_rows = len(raw)
    source = source.dropna(
        subset=["student_id", "question_id", "kc_source", "kc_text_source", "correct"]
    )
    exact_duplicates_removed = int(exact_duplicate_mask.sum())
    missing_rows_removed = input_rows - exact_duplicates_removed - len(source)
    source["correct"] = source["correct"].astype("int8")
    delimiters = list(data_config.get("multi_kc_delimiters", ["~~"]))
    joiner = str(data_config.get("multi_kc_joiner", " [SEP] "))
    source["kc_components"] = source["kc_text_source"].map(
        lambda value: split_components(value, delimiters)
    )
    source = source[source["kc_components"].map(bool)].copy()
    source["kc_text_raw"] = source["kc_text_source"].astype(str).str.strip()
    source["kc_text_norm"] = source["kc_components"].map(
        lambda parts: canonical_descriptor(parts, joiner)
    )
    source["kc_id"] = source["kc_components"].map(
        lambda parts: stable_identifier(canonical_descriptor(parts, joiner))
    )
    source["kc_components"] = source["kc_components"].map(
        lambda parts: json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    )
    parsed_time = (
        pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns, UTC]")
        if timestamp_column is None
        else pd.to_datetime(source["timestamp"], errors="coerce", utc=True)
    )
    source["_time_missing"] = parsed_time.isna().astype(int)
    source["_time_order"] = parsed_time.astype("int64")
    source.loc[parsed_time.isna(), "_time_order"] = source.loc[parsed_time.isna(), "_source_order"]
    source = source.sort_values(
        ["student_id", "_time_missing", "_time_order", "_source_order"], kind="mergesort"
    )
    minimum = int(data_config.get("minimum_student_interactions", 6))
    counts = source.groupby("student_id", sort=False).size()
    retained_students = counts[counts >= minimum].index
    short_students_removed = int((counts < minimum).sum())
    before_student_filter = len(source)
    source = source[source["student_id"].isin(retained_students)].copy()
    short_student_interactions_removed = before_student_filter - len(source)
    maximum = int(data_config.get("maximum_student_interactions", 50))
    before_truncation = len(source)
    source = source.groupby("student_id", sort=False, group_keys=False).head(maximum).copy()
    interactions_truncated = before_truncation - len(source)
    source["position"] = source.groupby("student_id", sort=False).cumcount() + 1
    frame = source.loc[:, list(STANDARD_COLUMNS)].reset_index(drop=True)
    validate_standard_frame(frame)
    stats = _statistics(frame)
    expected = dict(data_config.get("expected_raw_statistics", {}))
    raw_student_counts = raw.groupby(required["student_id"], dropna=True).size()
    raw_components = pd.DataFrame(
        {
            "student_id": raw[required["student_id"]].astype("string"),
            "component": raw[text_column].map(
                lambda value: split_components(value, delimiters)
                if not pd.isna(value)
                else []
            ),
        }
    ).explode("component")
    raw_components = raw_components.dropna(subset=["student_id", "component"])
    raw_concept_counts = raw_components.groupby("student_id")["component"].nunique()
    raw_stats = {
        "students": int(raw[required["student_id"]].nunique(dropna=True)),
        "interactions": int(len(raw)),
        "questions": int(question_values.nunique(dropna=True)),
        "concepts": int(raw_components["component"].nunique(dropna=True)),
        "median_interactions_per_student": float(raw_student_counts.median()),
        "median_concepts_per_student": float(raw_concept_counts.median()),
    }
    mismatches = {
        key: {"expected": value, "observed": raw_stats.get(key)}
        for key, value in expected.items()
        if key in raw_stats and float(value) != float(raw_stats[key])
    }
    if mismatches and not allow_statistic_mismatch:
        raise ValueError(f"Raw dataset statistics differ from the configured reference: {mismatches}")
    audit = {
        "dataset": data_config["dataset"],
        "preprocess_version": PREPROCESS_VERSION,
        "raw_file_name": raw_path.name,
        "raw_file_sha256": hash_file(raw_path),
        "raw_input_files": {
            path.relative_to(root).as_posix(): hash_file(path)
            for path in raw_input_paths
        },
        "source_transform": source_transform,
        "raw_statistics": raw_stats,
        "processed_statistics": stats,
        "statistic_mismatches": mismatches,
        "processed_table_sha256": hash_dataframe(frame),
        "normalization": "Unicode NFKC, casefold, whitespace collapse; component order retained",
        "question_identifier": (
            "composite:"
            + "+".join(
                resolve_column(raw.columns, list(component), "question_id_component")
                for component in data_config["question_id_components"]
            )
            if data_config.get("question_id_components")
            else "single:" + resolve_column(raw.columns, aliases["question_id"], "question_id")
        ),
        "ordering": (
            f"timestamp:{timestamp_column} with source-row fallback"
            if timestamp_column is not None
            else "source-row order"
        ),
        "filtering": {
            "missing_rows_removed": int(missing_rows_removed),
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "short_students_removed": short_students_removed,
            "short_student_interactions_removed": int(short_student_interactions_removed),
            "interactions_truncated_after_limit": int(interactions_truncated),
        },
    }
    output = Path(output_path)
    write_table(frame, output)
    write_json(output.with_suffix(output.suffix + ".audit.json"), audit)
    return frame, audit
