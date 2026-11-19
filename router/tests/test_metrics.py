"""Tests for router metric computation (no model load).

These lock down the metric denominators: recall for a tool must be computed on
that tool's support, not on all rows, otherwise the class distribution dilutes
the number.
"""

from __future__ import annotations

import json
import unittest

from router.evaluate import measure, select_rows


def row(identifier: str, expected: str) -> dict:
    if expected == "plan_task":
        return {"id": identifier, "instruction": f"正常任务 {identifier}", "expect_rejection": False}
    return {
        "id": identifier,
        "instruction": f"拒绝任务 {identifier}",
        "expect_rejection": True,
        "action": "clarify" if expected == "ask_clarification" else "reject",
    }


def call(name: str) -> str:
    return json.dumps({"name": name, "arguments": {"reason": "test"}})


class RouterMetricTests(unittest.TestCase):
    def test_recall_uses_the_tool_support_not_all_rows(self):
        rows = [row("n1", "plan_task"), row("r1", "reject_task"), row("c1", "ask_clarification")]
        calls = [call("plan_task"), call("plan_task"), call("ask_clarification")]
        _, summary = measure(rows, calls, "test")
        self.assertEqual(summary["reject_recall"], 0.0)
        self.assertEqual(summary["clarify_recall"], 1.0)
        self.assertEqual(summary["unsafe_plan_rate"], 0.5)
        self.assertEqual(summary["normal_plan_rate"], 1.0)
        self.assertAlmostEqual(summary["tool_accuracy"], 2 / 3)

    def test_all_correct_metrics_are_one(self):
        rows = [row("n1", "plan_task"), row("r1", "reject_task"), row("c1", "ask_clarification")]
        calls = [call("plan_task"), call("reject_task"), call("ask_clarification")]
        predictions, summary = measure(rows, calls, "test")
        self.assertEqual(summary["tool_accuracy"], 1.0)
        self.assertEqual(summary["reject_recall"], 1.0)
        self.assertEqual(summary["clarify_recall"], 1.0)
        self.assertEqual(summary["unsafe_plan_rate"], 0.0)
        self.assertEqual(summary["normal_plan_rate"], 1.0)
        self.assertTrue(all(item["valid_json"] for item in predictions))

    def test_malformed_calls_are_counted_as_invalid(self):
        rows = [row("n1", "plan_task")]
        _, summary = measure(rows, ["not json"], "test")
        self.assertEqual(summary["json_valid_rate"], 0.0)
        self.assertEqual(summary["tool_accuracy"], 0.0)
        self.assertEqual(summary["unsafe_plan_rate"], 0.0)

    def test_per_tool_support_matches_expected_classes(self):
        rows = [row("n1", "plan_task"), row("n2", "plan_task"), row("r1", "reject_task"), row("c1", "ask_clarification")]
        calls = [call("plan_task"), call("plan_task"), call("reject_task"), call("ask_clarification")]
        _, summary = measure(rows, calls, "test")
        self.assertEqual(summary["per_tool"]["plan_task"]["support"], 2)
        self.assertEqual(summary["per_tool"]["reject_task"]["support"], 1)
        self.assertEqual(summary["per_tool"]["ask_clarification"]["support"], 1)

    def test_select_rows_keeps_all_rejections_and_fills_with_normals(self):
        rows = [row(f"r{index}", "reject_task") for index in range(200)]
        rows += [row(f"n{index}", "plan_task") for index in range(800)]
        selected = select_rows(rows, limit=300)
        self.assertEqual(len(selected), 300)
        self.assertEqual(sum(1 for item in selected if item["expect_rejection"]), 200)


if __name__ == "__main__":
    unittest.main()
