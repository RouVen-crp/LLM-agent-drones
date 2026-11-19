"""Service tests: FastAPI endpoints with fake planner/simulator (no model load)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from drone_planner.simulation import SimulationResult
from service.api import create_app

VALID_PLAN = {
    "schema_version": "1.0",
    "mission_id": "mission-svc",
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

    def generate(self, instruction: str, feedback: str | None = None, examples=None) -> str:
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeSimulator:
    def __init__(self, success: bool = True) -> None:
        self.success = success

    def run(self, plan) -> SimulationResult:
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


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        store_path = Path(self._temporary.name) / "runs.sqlite"
        self.app = create_app(planner=FakePlanner([VALID_TEXT]), simulator=FakeSimulator(), store_path=store_path)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.store.close()
        self._temporary.cleanup()

    def test_health_and_tools(self):
        self.assertEqual(self.client.get("/healthz").json()["status"], "ok")
        names = {spec["name"] for spec in self.client.get("/v1/tools").json()}
        self.assertIn("plan_task", names)

    def test_run_and_fetch_mission(self):
        response = self.client.post("/v1/missions", json={"instruction": "飞到 (0.05, 0.0, 0.35)"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "accepted")
        run_id = payload["thread_id"]
        fetched = self.client.get(f"/v1/missions/{run_id}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["status"], "accepted")
        self.assertEqual(len(self.client.get("/v1/missions").json()), 1)

    def test_simulate_selection(self):
        payload = self.client.post("/v1/missions", json={"instruction": "飞到 (0.05, 0.0, 0.35)", "simulate": True}).json()
        self.assertTrue(payload["simulation"]["success"])

    def test_rejection_mission(self):
        app = create_app(
            planner=FakePlanner([REJECTION_TEXT]),
            simulator=None,
            store_path=Path(self._temporary.name) / "runs_rejection.sqlite",
        )
        client = TestClient(app)
        payload = client.post("/v1/missions", json={"instruction": "让它去那边"}).json()
        self.assertEqual(payload["status"], "rejected")
        app.state.store.close()

    def test_tool_invoke_and_validation_error(self):
        response = self.client.post("/v1/tools/validate_plan", json={"plan_text": VALID_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["result"]["valid"])
        self.assertEqual(self.client.post("/v1/tools/validate_plan", json={}).status_code, 400)
        self.assertEqual(self.client.post("/v1/tools/missing", json={}).status_code, 400)

    def test_router_wiring_in_service(self):
        class FakeRouter:
            def decide(self, instruction: str) -> dict:
                return {"name": "plan_task", "arguments": {"instruction": instruction}, "valid": True}

        app = create_app(
            planner=FakePlanner([VALID_TEXT]),
            router=FakeRouter(),
            store_path=Path(self._temporary.name) / "runs_router.sqlite",
        )
        client = TestClient(app)
        payload = client.post("/v1/missions", json={"instruction": "飞到 (0.05, 0.0, 0.35)"}).json()
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["tool_calls"][0]["tool"], "router_decide")
        app.state.store.close()

    def test_stream_endpoint_emits_events_and_done(self):
        with self.client.stream("POST", "/v1/missions/stream", json={"instruction": "飞到 (0.05, 0.0, 0.35)"}) as response:
            text = "".join(response.iter_text())
        self.assertEqual(response.status_code, 200)
        self.assertIn("data:", text)
        self.assertIn("event: done", text)
        self.assertIn('"status": "accepted"', text)


if __name__ == "__main__":
    unittest.main()
