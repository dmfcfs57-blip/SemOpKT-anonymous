from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from semopkt.data.schema import normalize_text, split_components
from semopkt.data.preprocess import preprocess_dataset
from semopkt.data.splits import apply_manifest, generate_split_manifest
from semopkt.data.synthetic import generate_synthetic_interactions
from semopkt.evaluation.calibration import TemperatureScaler
from semopkt.evaluation.metrics import compute_metrics
from semopkt.utils.io import read_table


class CoreTests(unittest.TestCase):
    def test_text_normalization_and_components(self) -> None:
        self.assertEqual(normalize_text("  Fraction\u3000Addition  "), "fraction addition")
        self.assertEqual(split_components("A~~B~~A", ["~~"]), ["a", "b"])

    def test_system_split_is_nested_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interactions.csv"
            frame = generate_synthetic_interactions(path, students=30, concepts=8, sequence_length=10)
            loaded = read_table(path)
            self.assertEqual(len(frame), len(loaded))
            manifest = generate_split_manifest(
                loaded,
                experiment="E1",
                protocol="system",
                seed=202601,
                train_sizes=(4, 8),
            )
            split4 = apply_manifest(loaded, manifest, 4)
            split8 = apply_manifest(loaded, manifest, 8)
            students4 = set(split4.train["student_id"])
            students8 = set(split8.train["student_id"])
            self.assertTrue(students4 < students8)
            self.assertFalse(students8 & set(split8.test["student_id"]))
            self.assertFalse(students8 & set(split8.validation["student_id"]))
            self.assertTrue(split8.test_target_mask.any())

    def test_metrics_and_temperature(self) -> None:
        frame = pd.DataFrame(
            {
                "correct": [0, 0, 1, 1, 0, 1],
                "probability": [0.1, 0.3, 0.8, 0.9, 0.4, 0.7],
                "student_id": ["a", "a", "a", "b", "b", "b"],
                "kc_id": ["x", "y", "x", "y", "x", "y"],
            }
        )
        metrics = compute_metrics(frame, ece_bins=3)
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertLess(metrics["nll"], 0.5)
        logits = np.log(frame["probability"] / (1.0 - frame["probability"]))
        scaler = TemperatureScaler.fit(logits.to_numpy(), frame["correct"].to_numpy())
        self.assertGreater(scaler.temperature, 0.0)

    def test_preprocessing_removes_exact_duplicates_but_keeps_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for student in ("a", "b"):
                for position in range(6):
                    rows.append(
                        {
                            "student": student,
                            "question": f"q{position}",
                            "concept": f"c{position % 2}",
                            "label": position % 2,
                            "time": f"2025-01-01 00:00:{position:02d}",
                        }
                    )
            rows.append(dict(rows[0]))
            rows.append({**rows[1], "time": "2025-01-01 00:01:00"})
            pd.DataFrame(rows).to_csv(root / "raw.csv", index=False)
            config = {
                "dataset": "Tiny",
                "raw_globs": ["*.csv"],
                "separator": ",",
                "column_aliases": {
                    "student_id": ["student"],
                    "question_id": ["question"],
                    "kc_id": ["concept"],
                    "kc_text": ["concept"],
                    "correct": ["label"],
                    "timestamp": ["time"],
                },
                "minimum_student_interactions": 6,
                "maximum_student_interactions": 50,
            }
            frame, audit = preprocess_dataset(config, root, root / "processed.csv")
            self.assertEqual(audit["filtering"]["exact_duplicates_removed"], 1)
            self.assertEqual(len(frame), 13)
            self.assertEqual(len(frame[frame["student_id"] == "a"]), 7)

    def test_preprocessing_without_timestamp_uses_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for student in ("a", "b"):
                for position in range(6):
                    rows.append(
                        {
                            "student": student,
                            "question": "repeated-question" if position < 2 else f"q{position}",
                            "concept": "repeated-concept" if position < 2 else f"c{position}",
                            "label": 1 if position < 2 else position % 2,
                            "attempt_id": f"{student}-{position}",
                        }
                    )
            pd.DataFrame(rows).to_csv(root / "raw.csv", index=False)
            config = {
                "dataset": "NoTime",
                "raw_globs": ["*.csv"],
                "separator": ",",
                "column_aliases": {
                    "student_id": ["student"],
                    "question_id": ["question"],
                    "kc_id": ["concept"],
                    "kc_text": ["concept"],
                    "correct": ["label"],
                    "timestamp": ["missing_timestamp"],
                },
                "minimum_student_interactions": 6,
                "maximum_student_interactions": 50,
            }
            frame, audit = preprocess_dataset(config, root, root / "processed.csv")
            self.assertEqual(len(frame), 12)
            self.assertEqual(audit["ordering"], "source-row order")
            self.assertEqual(
                frame.loc[frame["student_id"] == "a", "source_row_id"].tolist(),
                [f"row_{index}" for index in range(6)],
            )

    def test_nips_metadata_join_uses_level_three_subject_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary_rows = []
            answer_rows = []
            for student in ("a", "b"):
                for position in range(6):
                    answer_id = f"{student}-{position}"
                    primary_rows.append(
                        {
                            "UserId": student,
                            "QuestionId": f"q{position % 2}",
                            "AnswerId": answer_id,
                            "IsCorrect": position % 2,
                        }
                    )
                    answer_rows.append(
                        {
                            "AnswerId": answer_id,
                            "DateAnswered": f"2025-01-01 00:00:{position:02d}",
                        }
                    )
            pd.DataFrame(primary_rows).to_csv(root / "train.csv", index=False)
            pd.DataFrame(answer_rows).to_csv(root / "answers.csv", index=False)
            pd.DataFrame(
                {
                    "QuestionId": ["q0", "q1"],
                    "SubjectId": ["[1, 3]", "[2]"],
                }
            ).to_csv(root / "questions.csv", index=False)
            pd.DataFrame(
                {
                    "SubjectId": [1, 2, 3],
                    "Name": ["fractions", "equations", "mathematics"],
                    "Level": [3, 3, 2],
                }
            ).to_csv(root / "subjects.csv", index=False)
            config = {
                "dataset": "NIPS-Tiny",
                "raw_globs": ["train.csv"],
                "separator": ",",
                "column_aliases": {
                    "student_id": ["UserId"],
                    "question_id": ["QuestionId"],
                    "kc_id": ["SubjectIdLevel3"],
                    "kc_text": ["SubjectName"],
                    "correct": ["IsCorrect"],
                    "timestamp": ["DateAnswered"],
                },
                "nips_metadata": {
                    "answer_metadata": "answers.csv",
                    "question_metadata": "questions.csv",
                    "subject_metadata": "subjects.csv",
                    "answer_key": "AnswerId",
                    "question_key": "QuestionId",
                    "timestamp_column": "DateAnswered",
                    "subject_id_column": "SubjectId",
                    "subject_name_column": "Name",
                    "subject_level_column": "Level",
                    "target_subject_level": 3,
                },
                "minimum_student_interactions": 6,
                "maximum_student_interactions": 50,
            }
            frame, audit = preprocess_dataset(config, root, root / "processed.csv")
            self.assertEqual(len(frame), 12)
            self.assertEqual(set(frame["kc_text_norm"]), {"fractions", "equations"})
            self.assertEqual(audit["raw_statistics"]["concepts"], 2)
            self.assertEqual(len(audit["raw_input_files"]), 4)


if __name__ == "__main__":
    unittest.main()
