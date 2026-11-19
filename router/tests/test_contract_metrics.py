"""Tests for conditional Planner and end-to-end contract metrics (no model)."""

from __future__ import annotations

import json
import unittest

from router.evaluate import _report, planner_quality, router_agent_contract, workflow_contract

PLAN = {
    "schema_version": "1.0",
    "mission_id": "mission-metric",
    "mission": {"coordinate_frame": "local_meters", "selected_drone_id": "drone-1"},
    "drone_plans": [
        {
            "drone_id": "drone-1",
            "task_type": "point_to_point",
            "waypoints": [[0.05, 0.0, 0.35]],
            "return_home": True,
            "altitude_min": 0.2,
            "altitude_max": 0.5,
        }
    ],
    "metadata": {"source": "test", "category": "point"},
}
PLAN_TEXT = json.dumps(PLAN, ensure_ascii=False)
REJECTION_TEXT = json.dumps({"status": "needs_clarification", "reason": "缺少目标位置"}, ensure_ascii=False)


def normal_row(identifier: str = "n1") -> dict:
    return {"id": identifier, "instruction": "飞到 (0.05, 0.0, 0.35)", "expect_rejection": False, "plan": PLAN}


def rejection_row(identifier: str = "r1") -> dict:
    return {"id": identifier, "instruction": "去那边", "expect_rejection": True, "action": "reject"}


class ContractMetricTests(unittest.TestCase):
    def test_planner_quality_counts_valid_and_semantic_plans(self):
        quality = planner_quality([normal_row()], ["n1"], {"n1": PLAN_TEXT})
        self.assertEqual(quality["rows"], 1)
        self.assertEqual(quality["plan_valid_rate"], 1.0)
        self.assertEqual(quality["semantic_accuracy"], 1.0)
        self.assertEqual(quality["planner_rejection_rate"], 0.0)

    def test_planner_quality_treats_rejection_as_failure(self):
        quality = planner_quality([normal_row()], ["n1"], {"n1": REJECTION_TEXT})
        self.assertEqual(quality["plan_valid_rate"], 0.0)
        self.assertEqual(quality["semantic_accuracy"], 0.0)
        self.assertEqual(quality["planner_rejection_rate"], 1.0)

    def test_router_agent_contract_accepts_correct_plan_and_rejection(self):
        rows = [normal_row("n1"), rejection_row("r1")]
        predictions = [
            {"id": "n1", "predicted_tool": "plan_task", "expected_tool": "plan_task"},
            {"id": "r1", "predicted_tool": "reject_task", "expected_tool": "reject_task"},
        ]
        summary, records = router_agent_contract(rows, predictions, {"n1": PLAN_TEXT, "r1": REJECTION_TEXT})
        self.assertEqual(summary["contract_pass_rate"], 1.0)
        self.assertTrue(all(record["contract_pass"] for record in records))

    def test_router_agent_contract_fails_unsafe_task_routed_to_plan(self):
        rows = [rejection_row("r1")]
        predictions = [{"id": "r1", "predicted_tool": "plan_task", "expected_tool": "reject_task"}]
        summary, _ = router_agent_contract(rows, predictions, {"r1": PLAN_TEXT})
        self.assertEqual(summary["contract_pass_rate"], 0.0)

    def test_workflow_contract_accepts_valid_plan_and_explicit_rejection(self):
        rows = [normal_row("n1"), rejection_row("r1")]
        summary, records = workflow_contract(rows, {"n1": PLAN_TEXT, "r1": REJECTION_TEXT})
        self.assertEqual(summary["contract_pass_rate"], 1.0)
        self.assertTrue(all(record["contract_pass"] for record in records))


class ReportTests(unittest.TestCase):
    def test_report_handles_skipped_e2e(self):
        summaries = {
            "router": {
                "config": "router",
                "rows": 1,
                "json_valid_rate": 1.0,
                "tool_accuracy": 1.0,
                "reject_recall": 1.0,
                "clarify_recall": 1.0,
                "unsafe_plan_rate": 0.0,
                "normal_plan_rate": 1.0,
                "per_tool": {"plan_task": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}},
            },
            "base": {
                "config": "base",
                "rows": 1,
                "json_valid_rate": 1.0,
                "tool_accuracy": 1.0,
                "reject_recall": 0.0,
                "clarify_recall": 0.0,
                "unsafe_plan_rate": 0.0,
                "normal_plan_rate": 1.0,
                "per_tool": {"plan_task": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}},
            },
        }
        report = _report(summaries, [], {}, [], "test note")
        self.assertIn("未运行", report)
        self.assertIn("分工具指标", report)


if __name__ == "__main__":
    unittest.main()
