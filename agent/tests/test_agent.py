"""Unit tests for the tool registry and LangGraph agent runtime (no model load)."""

from __future__ import annotations

import json
import unittest

from agent.graph import AgentRunner
from agent.tools import ToolError, ToolRegistry, detect_rejection
from drone_planner.simulation import SimulationResult

VALID_PLAN = {
    "schema_version": "1.0",
    "mission_id": "mission-test",
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
VALID_TEXT = json.dumps(VALID_PLAN, ensure_ascii=False)
REJECTION_TEXT = json.dumps({"status": "needs_clarification", "reason": "缺少具体目标位置"}, ensure_ascii=False)


class FakePlanner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str | None]] = []

    def generate(self, instruction: str, feedback: str | None = None, examples=None) -> str:
        self.calls.append({"instruction": instruction, "feedback": feedback, "examples": examples})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeSimulator:
    def __init__(self, success: bool = True) -> None:
        self.success = success
        self.calls = 0

    def run(self, plan) -> SimulationResult:
        self.calls += 1
        return SimulationResult(
            success=self.success,
            reached_waypoints=1,
            total_waypoints=1,
            max_error=0.0,
            final_error=0.0,
            terminated_early=False,
            trajectory=[[0.18, 0.0, 0.4], [0.05, 0.0, 0.35]],
            errors=[],
        )


class AgentGraphTests(unittest.TestCase):
    def test_repair_loop_accepts_a_valid_second_attempt(self):
        planner = FakePlanner(["not json", VALID_TEXT])
        simulator = FakeSimulator()
        result = AgentRunner(planner=planner, simulator=simulator, max_attempts=3).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.valid)
        self.assertEqual(simulator.calls, 1)
        self.assertEqual([call["tool"] for call in result.tool_calls], ["plan_task", "validate_plan", "plan_task", "validate_plan", "simulate_plan"])
        self.assertIsNone(planner.calls[0]["feedback"])
        self.assertIsNotNone(planner.calls[1]["feedback"])

    def test_explicit_rejection_short_circuits_without_retry(self):
        planner = FakePlanner([REJECTION_TEXT])
        simulator = FakeSimulator()
        result = AgentRunner(planner=planner, simulator=simulator).run("让它去那边")
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.rejected)
        self.assertEqual(simulator.calls, 0)
        self.assertEqual([call["tool"] for call in result.tool_calls], ["plan_task", "reject_task"])

    def test_retry_budget_is_enforced(self):
        planner = FakePlanner(["not json"])
        result = AgentRunner(planner=planner, simulator=None, max_attempts=3).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(planner.calls), 3)

    def test_valid_plan_without_simulator_is_accepted(self):
        planner = FakePlanner([VALID_TEXT])
        result = AgentRunner(planner=planner, simulator=None).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "accepted")
        self.assertIsNone(result.simulation)

    def test_seed_plan_skips_first_generation(self):
        planner = FakePlanner([VALID_TEXT])
        result = AgentRunner(planner=planner, simulator=None).run("飞到 (0.05, 0.0, 0.35)", seed_plan_text=VALID_TEXT)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(len(planner.calls), 0)

    def test_simulation_failure_is_reported(self):
        planner = FakePlanner([VALID_TEXT])
        result = AgentRunner(planner=planner, simulator=FakeSimulator(success=False)).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "execution_failed")
        self.assertFalse(result.simulation["success"])


class FakeRouter:
    def __init__(self, decision: dict) -> None:
        self.decision = decision

    def decide(self, instruction: str) -> dict:
        return dict(self.decision)


class RouterGraphTests(unittest.TestCase):
    def test_router_plan_task_routes_to_planner(self):
        planner = FakePlanner([VALID_TEXT])
        router = FakeRouter({"name": "plan_task", "arguments": {"instruction": "飞到 (0.05, 0.0, 0.35)"}, "valid": True})
        result = AgentRunner(planner=planner, simulator=None, router=router).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "accepted")
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual([call["tool"] for call in result.tool_calls][:3], ["router_decide", "plan_task", "validate_plan"])

    def test_router_reject_task_skips_planner(self):
        planner = FakePlanner([VALID_TEXT])
        router = FakeRouter({"name": "reject_task", "arguments": {"reason": "坐标越界"}, "valid": True})
        result = AgentRunner(planner=planner, simulator=None, router=router).run("飞到 (9, 9, 9)")
        self.assertEqual(result.status, "rejected")
        self.assertTrue(result.rejected)
        self.assertEqual(len(planner.calls), 0)
        self.assertEqual([call["tool"] for call in result.tool_calls], ["router_decide", "reject_task"])
        self.assertEqual(result.rejection_reason, "坐标越界")

    def test_router_clarification_status(self):
        planner = FakePlanner([VALID_TEXT])
        router = FakeRouter({"name": "ask_clarification", "arguments": {"reason": "缺少位置"}, "valid": True})
        result = AgentRunner(planner=planner, simulator=None, router=router).run("让它去那边")
        self.assertEqual(result.status, "clarification")
        self.assertEqual(len(planner.calls), 0)
        self.assertEqual(result.tool_calls[-1]["tool"], "ask_clarification")

    def test_invalid_router_output_fails_closed(self):
        planner = FakePlanner([VALID_TEXT])
        router = FakeRouter({"name": None, "arguments": {}, "valid": False})
        result = AgentRunner(planner=planner, simulator=None, router=router).run("飞到 (0.05, 0.0, 0.35)")
        self.assertEqual(result.status, "invalid")
        self.assertEqual(len(planner.calls), 0)


class ToolRegistryTests(unittest.TestCase):
    def test_specs_expose_all_tools(self):
        names = {spec["name"] for spec in ToolRegistry().specs()}
        self.assertEqual(names, {"plan_task", "validate_plan", "simulate_plan", "get_contract"})

    def test_unknown_tool_raises(self):
        with self.assertRaises(ToolError):
            ToolRegistry().invoke("missing", {})

    def test_missing_required_argument_raises(self):
        with self.assertRaises(ToolError):
            ToolRegistry().invoke("validate_plan", {})

    def test_simulate_without_simulator_returns_failed_call(self):
        call = ToolRegistry(planner=FakePlanner([VALID_TEXT])).invoke("simulate_plan", {"plan_text": VALID_TEXT})
        self.assertFalse(call.ok)
        self.assertIn("未启用", call.error or "")

    def test_contract_tool_returns_bounds(self):
        result = ToolRegistry().invoke("get_contract", {}).result
        self.assertEqual(result["max_drones"], 3)
        self.assertIn("point_to_point", result["task_types"])

    def test_detect_rejection(self):
        self.assertEqual(detect_rejection(REJECTION_TEXT), (True, "缺少具体目标位置"))
        self.assertEqual(detect_rejection(VALID_TEXT), (False, None))
        self.assertEqual(detect_rejection("not json"), (False, None))


if __name__ == "__main__":
    unittest.main()
