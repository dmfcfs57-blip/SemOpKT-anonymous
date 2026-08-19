from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from semopkt.analysis.postprocess import _e20_cases
from semopkt.utils.io import write_json, write_table


class QualitativeTests(unittest.TestCase):
    def test_e20_case_selection_retains_row_identity_and_field_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semopkt_directory = root / "SemOpKT"
            baseline_directory = root / "CLST"
            specification = {
                "experiment": "E20",
                "dataset": "Synthetic",
                "protocol": "qualitative",
                "seed": 202601,
                "train_size": 64,
                "holdout_ratio": 0.2,
            }
            write_json(
                semopkt_directory / "complete.json",
                {
                    "status": "complete",
                    "specification": {**specification, "model": "SemOpKT"},
                },
            )
            write_json(
                baseline_directory / "complete.json",
                {
                    "status": "complete",
                    "specification": {**specification, "model": "CLST"},
                },
            )
            rows = []
            for index in range(6):
                rows.append(
                    {
                        "dataset": "Synthetic",
                        "student_id": f"student-{index // 3}",
                        "source_row_id": f"row-{index}",
                        "question_id": f"question-{index}",
                        "kc_id": f"kc-{index % 3}",
                        "position": index % 3 + 1,
                        "correct": index % 2,
                        "probability": 0.20 + 0.10 * index,
                        "seen_status": "unseen" if index < 2 else "seen",
                    }
                )
            semopkt = pd.DataFrame.from_records(rows)
            baseline = semopkt.copy()
            baseline["probability"] = 1.0 - baseline["probability"]
            write_table(semopkt, semopkt_directory / "predictions.csv")
            write_table(baseline, baseline_directory / "predictions.csv")
            traces = semopkt[
                [
                    "dataset",
                    "student_id",
                    "source_row_id",
                    "position",
                    "kc_id",
                    "correct",
                ]
            ].copy()
            traces["probability_before"] = semopkt["probability"]
            traces["probability_after"] = semopkt["probability"] + 0.01
            write_table(traces, semopkt_directory / "field_traces.csv")

            selected = _e20_cases([semopkt_directory, baseline_directory])

            self.assertEqual(len(selected), 5)
            self.assertTrue(selected["source_row_id"].notna().all())
            self.assertTrue(selected["probability_before"].notna().all())
            self.assertEqual(selected["selection_rule"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
