"""Tests for router training data mapping and balancing (no model load)."""

from __future__ import annotations

import json
import unittest

from router.data import build_balanced_train, record_to_call, row_to_sample


def normal(identifier: str = "n") -> dict:
    return {"id": identifier, "instruction": f"让 drone-1 飞到 ({identifier}, 0.0, 0.35) 后返回起点", "expect_rejection": False}


def rejection(identifier: str, action: str) -> dict:
    return {
        "id": identifier,
        "instruction": f"无效任务 {identifier}",
        "expect_rejection": True,
        "action": action,
        "expected_response": {"status": "needs_clarification", "reason": "缺少信息"},
    }


class RouterDataTests(unittest.TestCase):
    def test_normal_record_maps_to_plan_task(self):
        call = record_to_call(normal("n1"))
        self.assertEqual(call["name"], "plan_task")
        self.assertEqual(call["arguments"], {"instruction": normal("n1")["instruction"]})

    def test_reject_record_maps_to_reject_task(self):
        call = record_to_call(rejection("r1", "reject"))
        self.assertEqual(call["name"], "reject_task")
        self.assertIn("reason", call["arguments"])

    def test_clarify_record_maps_to_ask_clarification(self):
        call = record_to_call(rejection("c1", "clarify"))
        self.assertEqual(call["name"], "ask_clarification")
        self.assertIn("reason", call["arguments"])

    def test_missing_action_defaults_to_reject(self):
        row = rejection("r2", "reject")
        row.pop("action")
        self.assertEqual(record_to_call(row)["name"], "reject_task")

    def test_row_to_sample_has_parsable_target(self):
        sample = row_to_sample(rejection("r3", "clarify"))
        target = json.loads(sample["target"])
        self.assertEqual(target["name"], "ask_clarification")
        self.assertEqual(sample["tool"], "ask_clarification")

    def test_balanced_train_counts(self):
        samples = [row_to_sample(normal(f"n{index}")) for index in range(100)]
        samples += [row_to_sample(rejection(f"r{index}", "reject")) for index in range(10)]
        samples += [row_to_sample(rejection(f"c{index}", "clarify")) for index in range(10)]
        balanced = build_balanced_train(samples, seed=42, normal_limit=30)
        counts: dict[str, int] = {}
        for row in balanced:
            counts[row["tool"]] = counts.get(row["tool"], 0) + 1
        self.assertEqual(counts, {"plan_task": 30, "reject_task": 20, "ask_clarification": 60})


if __name__ == "__main__":
    unittest.main()
