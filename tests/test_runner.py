from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from semopkt.data.synthetic import generate_synthetic_interactions
from semopkt.experiments.runner import ExperimentRunner
from semopkt.experiments.specs import RunSpec
from semopkt.utils.io import write_table, write_yaml


class RunnerTests(unittest.TestCase):
    def test_source_only_transfer_does_not_fit_target_temperature(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_directory = root / "configs" / "experiment"
            source_path = root / "data" / "source.csv"
            target_path = root / "data" / "target.csv"
            source = generate_synthetic_interactions(
                source_path, students=24, concepts=8, sequence_length=10, seed=31
            )
            target = generate_synthetic_interactions(
                target_path, students=24, concepts=8, sequence_length=10, seed=37
            )
            source["dataset"] = "Source"
            target["dataset"] = "Target"
            write_table(source, source_path)
            write_table(target, target_path)
            configuration = {
                "project": "anonymous-test",
                "model_config": str(project / "configs/model/smoke_semopkt.yaml"),
                "baseline_config": str(project / "configs/baseline/baselines.yaml"),
                "datasets": {
                    "Source": {"processed": str(source_path)},
                    "Target": {"processed": str(target_path)},
                },
                "output_root": str(root / "runs"),
                "manifest_root": str(root / "data/manifests"),
                "embedding_root": str(root / "data/embeddings"),
                "seeds": [202601],
                "train_sizes": [4, 8],
                "data_efficiency_sizes": [4, 8],
                "holdout_ratios": [0.20],
                "inducing_sizes": [8],
                "query_sizes": [8],
                "student_split": {
                    "test_fraction": 0.20,
                    "validation_fraction_of_development": 0.15,
                },
                "unseen_concept": {
                    "minimum_students": 2,
                    "minimum_interactions": 4,
                    "minimum_positive": 1,
                    "minimum_negative": 1,
                    "cluster_initializations": 2,
                },
                "models": ["SemOpKT"],
                "experiments": ["E8"],
                "statistics": {"ece_bins": 5},
                "cross_domain": {"support_fraction": 0.80},
                "hardware_timing": {
                    "warmup": 1,
                    "repetitions": 2,
                    "sequence_length": 10,
                },
            }
            config_path = config_directory / "test.yaml"
            write_yaml(config_path, configuration)
            specification = RunSpec(
                "E8",
                "Target",
                "source_only_online",
                "SemOpKT",
                202601,
                source_dataset="Source",
                target_dataset="Target",
                condition="overlap_retained",
            )
            runner = ExperimentRunner(config_path, device="cpu")
            resolved = runner._resolved_model_config({}, "SemOpKT")
            self.assertEqual(resolved["name"], "SemOpKT")
            self.assertNotIn("_config_path", resolved)
            record = runner.run(
                specification, resume=False
            )
            self.assertFalse(record["target_temperature_fitted"])
            self.assertFalse(
                record["target_response_access"]["during_source_pretraining"]
            )
            self.assertFalse(
                record["target_response_access"][
                    "test_labels_used_for_fitting_or_selection"
                ]
            )
            self.assertNotIn("metrics_calibrated", record)
            self.assertGreater(record["metrics_uncalibrated"]["count"], 0)
            self.assertTrue(
                record["vocabulary_boundary"][
                    "field_or_graph_initialized_from_source_only"
                ]
            )
            resumed = ExperimentRunner(config_path, device="cpu").run(
                specification, resume=True
            )
            self.assertEqual(resumed["bound_hash"], record["bound_hash"])
            adaptation = RunSpec(
                "E8",
                "Target",
                "source_target_adaptation",
                "SemOpKT",
                202601,
                source_dataset="Source",
                target_dataset="Target",
                adaptation_size=8,
                condition="overlap_retained",
            )
            adapted = ExperimentRunner(config_path, device="cpu").run(
                adaptation, resume=False
            )
            self.assertTrue(adapted["target_temperature_fitted"])
            self.assertIn("metrics_calibrated", adapted)


if __name__ == "__main__":
    unittest.main()
