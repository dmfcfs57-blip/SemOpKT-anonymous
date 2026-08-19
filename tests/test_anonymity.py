from __future__ import annotations

import unittest
from pathlib import Path

from semopkt.audit.anonymity import audit_anonymity


class AnonymousReleaseTests(unittest.TestCase):
    def test_repository_contains_no_identity_material(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = audit_anonymity(root)
        self.assertTrue(report["passed"], report["findings"])


if __name__ == "__main__":
    unittest.main()

