from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from semopkt.audit.leakage import verify_strict_online_order
from semopkt.config import load_config
from semopkt.models.baselines import (
    BKTModel,
    DKVMNModel,
    DisentangledKT,
    GraphKTModel,
    MetaRecurrentKT,
    RecurrentKT,
    TransformerKT,
)
from semopkt.models.semopkt import SemOpKT
from semopkt.models.registry import build_model


def batch() -> dict[str, object]:
    return {
        "dataset": ["Synthetic", "Synthetic"],
        "student_id": ["a", "b"],
        "question_ids": [["q1", "q2", "q3"], ["q1", "q2", "q3"]],
        "kc_ids": [["k1", "k2", "k3"], ["k1", "k2", "k3"]],
        "source_row_ids": [["a1", "a2", "a3"], ["b1", "b2", "b3"]],
        "concept_indices": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "labels": torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        "positions": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        "valid_mask": torch.ones((2, 3), dtype=torch.bool),
        "score_mask": torch.ones((2, 3), dtype=torch.bool),
    }


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.embeddings = np.random.default_rng(7).normal(size=(5, 32)).astype(np.float32)

    def test_semopkt_shape_and_online_order(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/model/smoke_semopkt.yaml")
        model = SemOpKT(config, self.embeddings)
        model.build_smoothness_graph()
        output = model(batch())
        self.assertEqual(tuple(output.logits.shape), (2, 3))
        audit = verify_strict_online_order(model, batch(), tolerance=1.0e-6)
        self.assertTrue(audit["passed"])

    def test_manuscript_configuration_parameter_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/model/semopkt.yaml")
        embeddings = np.random.default_rng(11).normal(size=(123, 768)).astype(np.float32)
        model = SemOpKT(config, embeddings)
        self.assertEqual(model.trainable_parameter_count(), 7_441_107)

    def test_baseline_shapes(self) -> None:
        models = [
            BKTModel(5),
            RecurrentKT("DKT", self.embeddings, hidden_size=16, dropout=0.0),
            TransformerKT("AKT", self.embeddings, hidden_size=16, layers=1, heads=2, dropout=0.0),
            DKVMNModel(self.embeddings, hidden_size=16, memory_slots=4, value_size=16, dropout=0.0),
            GraphKTModel(self.embeddings, hidden_size=16, graph_neighbors=2, dropout=0.0),
            MetaRecurrentKT(self.embeddings, hidden_size=16, dropout=0.0),
            DisentangledKT(self.embeddings, hidden_size=16, dropout=0.0),
        ]
        for model in models:
            with self.subTest(model=model.model_name):
                output = model(batch())
                self.assertEqual(tuple(output.logits.shape), (2, 3))
                audit = verify_strict_online_order(model, batch(), tolerance=1.0e-6)
                self.assertTrue(audit["passed"])

    def test_structural_ablations_match_full_parameter_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/model/smoke_semopkt.yaml")
        baselines = load_config(root / "configs/baseline/baselines.yaml")
        full = SemOpKT(config, self.embeddings).trainable_parameter_count()
        for name in ("A4", "A5", "A6", "A7"):
            with self.subTest(model=name):
                model = build_model(
                    name,
                    config,
                    baselines,
                    self.embeddings,
                    seed=7,
                )
                relative_error = abs(model.trainable_parameter_count() - full) / full
                self.assertLessEqual(relative_error, 0.05)
                self.assertEqual(tuple(model(batch()).logits.shape), (2, 3))
                self.assertTrue(
                    verify_strict_online_order(
                        model, batch(), tolerance=1.0e-6
                    )["passed"]
                )


if __name__ == "__main__":
    unittest.main()
