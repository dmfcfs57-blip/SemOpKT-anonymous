from __future__ import annotations

import unittest
from pathlib import Path

from semopkt.config import load_config
from semopkt.experiments.catalog import build_experiment_specs
from semopkt.experiments.runner import _canonical_protocol


class CatalogTests(unittest.TestCase):
    def test_all_experiment_identifiers_expand(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiment/paper.yaml")
        for index in range(21):
            experiment = f"E{index}"
            with self.subTest(experiment=experiment):
                specifications = build_experiment_specs(config, experiment)
                self.assertTrue(specifications)
                self.assertTrue(all(spec.experiment == experiment for spec in specifications))

    def test_smoke_catalog_respects_model_allowlist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiment/smoke.yaml")
        models = {spec.model for spec in build_experiment_specs(config, "E1")}
        self.assertEqual(models, {"DKT", "SemOpKT"})

    def test_vocabulary_expansion_uses_disjoint_test_students(self) -> None:
        self.assertEqual(_canonical_protocol("vocabulary_expansion"), "double_random")

    def test_single_dataset_catalog_omits_undefined_transfers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/experiment/smoke.yaml")
        specifications = build_experiment_specs(config, "E8")
        self.assertTrue(specifications)
        self.assertEqual({spec.protocol for spec in specifications}, {"target_only"})
        self.assertTrue(all(spec.source_dataset is None for spec in specifications))


if __name__ == "__main__":
    unittest.main()
