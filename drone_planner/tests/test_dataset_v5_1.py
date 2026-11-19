"""Tests for the v5.1 training-only dataset revision.

v5.1 must add a train-only unsupported-shape rejection family while keeping the
v5 validation and test splits byte-identical, so router results remain
comparable on the same evaluation tasks.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from drone_planner.dataset_v5 import write_dataset_v5
from drone_planner.dataset_v5_1 import write_dataset_v5_1


def read_split(path: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


class DatasetV51Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        cls.base = root / "v5"
        cls.revision = root / "v5_1"
        write_dataset_v5(cls.base, scale=20, seed=42)
        cls.manifest = write_dataset_v5_1(cls.revision, base_dir=cls.base, scale=20, seed=42)
        cls.base_manifest = json.loads((cls.base / "manifest.json").read_text(encoding="utf-8"))
        cls.splits = {name: read_split(cls.revision, name) for name in ("train", "validation", "test")}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_validation_and_test_are_byte_identical_to_v5(self):
        for name in ("validation", "test"):
            self.assertEqual((self.revision / f"{name}.jsonl").read_bytes(), (self.base / f"{name}.jsonl").read_bytes(), name)
            self.assertEqual(self.manifest["sha256"][name], self.base_manifest["sha256"][name], name)

    def test_split_sizes_and_rejection_counts(self):
        self.assertEqual(self.manifest["splits"], {"train": 8000, "validation": 1000, "test": 1000})
        self.assertEqual(self.manifest["rejections"], {"train": 320, "validation": 100, "test": 200})
        self.assertEqual(len(self.splits["train"]), 8000)

    def test_train_has_the_new_unsupported_capability_families(self):
        train_rejections = [row for row in self.splits["train"] if row.get("expect_rejection")]
        families = Counter(row["risk_family"] for row in train_rejections)
        for expected in ("unsupported_shape", "unsupported_conic", "missing_zone_constraint"):
            self.assertIn(expected, families)
        self.assertEqual(len(families), 12)
        self.assertEqual(sum(families.values()), 320)
        self.assertTrue(all(count in (26, 27) for count in families.values()), families)

    def test_unsupported_shape_does_not_leak_the_test_ellipse_family(self):
        train_rejections = [row for row in self.splits["train"] if row.get("expect_rejection")]
        unsupported = [row["instruction"] for row in train_rejections if row["risk_family"].startswith("unsupported_")]
        self.assertTrue(unsupported)
        self.assertFalse(any("椭圆" in instruction for instruction in unsupported))
        test_families = {row["risk_family"] for row in self.splits["test"] if row.get("expect_rejection")}
        self.assertIn("unsupported_ellipse", test_families)
        self.assertNotIn("unsupported_shape", test_families)
        self.assertNotIn("unsupported_conic", test_families)
        self.assertNotIn("missing_zone_constraint", test_families)

    def test_train_and_test_families_remain_isolated(self):
        train_families = {row["risk_family"] for row in self.splits["train"] if row.get("expect_rejection")}
        validation_families = {row["risk_family"] for row in self.splits["validation"] if row.get("expect_rejection")}
        test_families = {row["risk_family"] for row in self.splits["test"] if row.get("expect_rejection")}
        self.assertFalse(train_families & validation_families)
        self.assertFalse(train_families & test_families)
        self.assertFalse(validation_families & test_families)

    def test_manifest_records_revision_and_base(self):
        self.assertEqual(self.manifest["dataset_version"], "v5.1")
        self.assertEqual(self.manifest["base_version"], "v5")
        self.assertEqual(self.manifest["base_sha256"]["test"], self.base_manifest["sha256"]["test"])


if __name__ == "__main__":
    unittest.main()
