from __future__ import annotations

import unittest

import numpy as np

from semopkt.analysis.perturbations import (
    history_concept_remap_sequences,
    target_replay_sequences,
)
from semopkt.data.sequences import StudentSequence


class PerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = StudentSequence(
            dataset="Tiny",
            student_id="u1",
            question_ids=["q1", "q2", "q3", "q4"],
            kc_ids=["k0", "k1", "k2", "k3"],
            concept_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            labels=np.asarray([0, 1, 0, 1], dtype=np.float32),
            positions=np.asarray([1, 2, 3, 4], dtype=np.int64),
            source_row_ids=["r1", "r2", "r3", "r4"],
            score_mask=np.asarray([False, True, False, True]),
        )

    def test_history_deletion_preserves_targets(self) -> None:
        replay = target_replay_sequences(
            [self.sequence], "history_delete", 0.5, seed=9
        )
        self.assertEqual([item.source_row_ids[-1] for item in replay], ["r2", "r4"])
        self.assertEqual([int(item.labels[-1]) for item in replay], [1, 1])
        self.assertTrue(all(item.score_mask[-1] for item in replay))

    def test_name_remap_never_changes_target_index(self) -> None:
        replay = history_concept_remap_sequences(
            [self.sequence], {0: 10, 1: 11, 2: 12, 3: 13}
        )
        self.assertEqual([int(item.concept_indices[-1]) for item in replay], [1, 3])
        self.assertEqual(int(replay[-1].concept_indices[0]), 10)


if __name__ == "__main__":
    unittest.main()
