from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from semopkt.analysis.aggregate import completed_run_directories
from semopkt.analysis.postprocess import _e12_protocol_expansion_drift
from semopkt.analysis.tables import _latex
from semopkt.utils.hashing import hash_file
from semopkt.utils.io import write_json, write_table


class ReportingTests(unittest.TestCase):
    def test_latex_escaping_is_single_pass(self) -> None:
        rendered = _latex(pd.DataFrame({"name_with_underscore": [r"a\b_{c}"]}))
        self.assertIn(r"name\_with\_underscore", rendered)
        self.assertIn(r"a\textbackslash{}b\_\{c\}", rendered)
        self.assertNotIn(r"textbackslash\{\}", rendered)

    def test_completed_run_reader_rejects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            artifact = run / "predictions.csv"
            write_table(pd.DataFrame({"value": [1]}), artifact)
            record = {
                "status": "complete",
                "artifact_hashes": {"predictions.csv": hash_file(artifact)},
            }
            write_json(run / "run.json", record)
            write_json(run / "complete.json", record)
            self.assertEqual(completed_run_directories(root), [run])
            write_table(pd.DataFrame({"value": [2]}), artifact)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                completed_run_directories(root)

    def test_e12_protocol_drift_pairs_old_targets_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_directory = root / "full"
            reduced_directory = root / "reduced"
            common = {
                "experiment": "E12",
                "dataset": "Synthetic",
                "model": "SemOpKT",
                "seed": 202601,
                "train_size": 64,
            }
            write_json(
                full_directory / "complete.json",
                {
                    "status": "complete",
                    "specification": {
                        **common,
                        "protocol": "vocabulary_full",
                        "condition": "visible-100",
                        "holdout_ratio": None,
                    },
                },
            )
            write_json(
                reduced_directory / "complete.json",
                {
                    "status": "complete",
                    "specification": {
                        **common,
                        "protocol": "vocabulary_expansion",
                        "condition": "visible-50",
                        "holdout_ratio": 0.5,
                    },
                },
            )
            rows = []
            for index, label in enumerate((0, 1, 0, 1), start=1):
                rows.append(
                    {
                        "dataset": "Synthetic",
                        "student_id": f"student-{index // 3}",
                        "source_row_id": f"row-{index}",
                        "question_id": f"question-{index}",
                        "kc_id": "old" if index <= 3 else "added",
                        "correct": label,
                        "position": index + 1,
                        "probability": 0.2 + index * 0.1,
                        "seen_status": "seen" if index <= 3 else "unseen",
                    }
                )
            reduced = pd.DataFrame.from_records(rows)
            full = reduced.copy()
            full["probability"] = full["probability"] + 0.05
            full["seen_status"] = "seen"
            write_table(reduced, reduced_directory / "predictions_uncalibrated.csv")
            write_table(full, full_directory / "predictions_uncalibrated.csv")

            result = _e12_protocol_expansion_drift(
                [full_directory, reduced_directory]
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(int(result.loc[0, "paired_old_targets"]), 3)
            self.assertEqual(int(result.loc[0, "added_targets"]), 1)
            self.assertAlmostEqual(
                float(result.loc[0, "protocol_mean_absolute_old_prediction_drift"]),
                0.05,
            )


if __name__ == "__main__":
    unittest.main()
