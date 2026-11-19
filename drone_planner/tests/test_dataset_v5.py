"""Tests for the v5 10k-scale dataset generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from drone_planner.dataset_v5 import generate_records_v5, split_records_v5, write_dataset_v5
from drone_planner.validation import validate_payload


class DatasetV5Tests(unittest.TestCase):
    def test_base_scale_matches_v4_sizes(self):
        splits = split_records_v5(generate_records_v5(scale=1), seed=42)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 400, "validation": 50, "test": 50})

    def test_ten_k_scale_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_dataset_v5(output, scale=20, seed=42)
            splits = {name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()] for name in ("train", "validation", "test")}
        self.assertEqual(manifest["splits"], {"train": 8000, "validation": 1000, "test": 1000})
        self.assertEqual(manifest["evaluation"], {"valid_tasks": 800, "expected_rejections": 200})
        self.assertEqual(manifest["rejections"], {"train": 320, "validation": 100, "test": 200})
        self.assertEqual(len(splits["test"]), 1000)

    def test_rejections_are_unique_and_family_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset_v5(output, scale=20, seed=42)
            splits = {name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()] for name in ("train", "validation", "test")}
        families = {name: {row["risk_family"] for row in rows if row.get("expect_rejection")} for name, rows in splits.items()}
        self.assertFalse(families["train"] & families["validation"])
        self.assertFalse(families["train"] & families["test"])
        self.assertFalse(families["validation"] & families["test"])
        for name, rows in splits.items():
            rejection_instructions = [row["instruction"] for row in rows if row.get("expect_rejection")]
            self.assertEqual(len(rejection_instructions), len(set(rejection_instructions)), name)
            # Random float generation can collide for a tiny number of normal tasks.
            instructions = [row["instruction"] for row in rows]
            self.assertLessEqual(len(instructions) - len(set(instructions)), 10, name)

    def test_actions_and_category_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset_v5(output, scale=20, seed=42)
            test = [json.loads(line) for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()]
            train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        test_actions = Counter(row.get("action") for row in test if row.get("expect_rejection"))
        self.assertEqual(test_actions, {"reject": 160, "clarify": 40})
        train_actions = Counter(row.get("action") for row in train if row.get("expect_rejection"))
        self.assertGreater(train_actions["clarify"], 0)
        normal_categories = Counter(row["category"] for row in test if not row.get("expect_rejection"))
        self.assertEqual(normal_categories, {"point": 208, "multi": 240, "circle": 96, "rectangle": 96, "multi_drone": 160})

    def test_normal_plans_follow_the_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset_v5(output, scale=20, seed=42)
            test = [json.loads(line) for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in test:
            if row.get("expect_rejection"):
                continue
            self.assertTrue(validate_payload(row["plan"]).valid, row["id"])


if __name__ == "__main__":
    unittest.main()
