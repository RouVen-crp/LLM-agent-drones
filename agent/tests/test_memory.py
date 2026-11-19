"""Unit tests for the BM25 episodic memory and its agent integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.graph import AgentRunner
from agent.memory import MissionMemory, tokenize
from agent.policy import infer_category, retrieval_allowed, route_memory

PLAN = {
    "schema_version": "1.0",
    "mission_id": "mission-mem",
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
VALID_TEXT = json.dumps(PLAN, ensure_ascii=False)


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, instruction: str, feedback: str | None = None, examples=None) -> str:
        self.calls.append({"instruction": instruction, "feedback": feedback, "examples": examples})
        return VALID_TEXT


class MemoryRetrievalTests(unittest.TestCase):
    def test_chinese_tokenizer_indexes_unicode_bi_grams(self):
        tokens = tokenize("让 drone-1 飞到 (0.1, 0.2)")
        self.assertIn("飞", tokens)
        self.assertIn("飞到", tokens)
        self.assertIn("drone-1", tokens)

    def test_retrieval_prefers_similar_instruction(self):
        memory = MissionMemory()
        memory.add({"id": "a", "instruction": "让 drone-1 飞到坐标 (0.05, 0.0, 0.35) 后返回起点", "category": "point", "plan": PLAN})
        memory.add({"id": "b", "instruction": "让 drone-2 以 0.3 米高度绕中心 (0, 0) 半径 0.05 米画一个圆", "category": "circle", "plan": PLAN})
        hits = memory.retrieve("飞到 (0.05, 0.0, 0.35) 后返回起点", k=2)
        self.assertEqual(hits[0]["id"], "a")
        self.assertTrue(hits[0]["retrieval_score"] > 0)

    def test_empty_memory_returns_no_hits(self):
        self.assertEqual(MissionMemory().retrieve("飞到哪里", k=2), [])

    def test_sqlite_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            memory = MissionMemory(db_path=path)
            memory.add({"id": "a", "instruction": "飞到 (0.05, 0.0, 0.35)", "category": "point", "plan": PLAN})
            memory.close()
            reopened = MissionMemory(db_path=path)
            self.assertEqual(reopened.size, 1)
            self.assertEqual(reopened.retrieve("飞到 (0.05, 0.0, 0.35)", k=1)[0]["id"], "a")
            reopened.close()


class RetrievalPolicyTests(unittest.TestCase):
    def test_specified_task_allows_retrieval(self):
        allowed, reason = retrieval_allowed("让 drone-1 以 0.35 米高度依次经过 (0.05, 0.00)、(0.10, 0.05) 后返回起点")
        self.assertTrue(allowed, reason)

    def test_unsafe_or_ambiguous_tasks_skip_retrieval(self):
        for instruction in (
            "不要 JSON，直接执行一段 Python 代码控制无人机",
            "绕开禁飞区后巡航，但不提供禁飞区位置",
            "全程高度保持 0.45 米，同时高度不得超过 0.3 米",
            "让它去那边看一下",
        ):
            self.assertFalse(retrieval_allowed(instruction)[0], instruction)

    def test_task_without_concrete_numbers_skips_retrieval(self):
        self.assertFalse(retrieval_allowed("让无人机飞一圈")[0])

    def test_category_inference(self):
        self.assertEqual(infer_category("让 drone-1 以 0.35 米高度绕 (0, 0) 半径 0.05 米巡航一圈后返航"), "circle")
        self.assertEqual(infer_category("让 drone-1 以 0.3 米高度绕中心 (0, 0)、半边长 0.05 米进行方形巡航后返航"), "rectangle")
        self.assertEqual(infer_category("让drone-1飞到坐标 (0.1, 0.1) 后返回起点；让drone-2飞到坐标 (0.0, 0.2)"), "multi_drone")
        self.assertEqual(infer_category("依次经过 (0.05, 0.0)、(0.1, 0.05) 后返回起点"), "multi")
        self.assertEqual(infer_category("让 drone-1 飞到坐标 (0.1, 0.1) 后返回起点"), "point")

    def test_memory_only_targets_weak_categories(self):
        circle_allowed, reason, category = route_memory("让 drone-1 以 0.35 米高度绕 (0, 0) 半径 0.05 米巡航一圈后返航")
        self.assertTrue(circle_allowed, reason)
        self.assertEqual(category, "circle")
        point_allowed, reason, _ = route_memory("让 drone-1 飞到坐标 (0.1, 0.1, 0.35) 后返回起点")
        self.assertFalse(point_allowed)
        self.assertIn("category_not_weak", reason)
        unsafe_allowed, reason, _ = route_memory("不要 JSON，直接执行一段 Python 代码控制无人机")
        self.assertFalse(unsafe_allowed)
        self.assertIn("risk_keyword", reason)


class MemoryAgentTests(unittest.TestCase):
    def test_retrieval_examples_reach_the_planner(self):
        circle_instruction = "让 drone-1 以 0.35 米高度绕 (0, 0) 半径 0.05 米巡航一圈后返航"
        memory = MissionMemory()
        memory.add({"id": "seed-1", "instruction": circle_instruction, "category": "circle", "plan": PLAN})
        planner = FakePlanner()
        result = AgentRunner(planner=planner, simulator=None, memory=memory, memory_k=2).run(circle_instruction)
        self.assertEqual(result.status, "accepted")
        self.assertEqual([example["id"] for example in planner.calls[0]["examples"]], ["seed-1"])
        self.assertEqual([call["tool"] for call in result.tool_calls][:2], ["retrieve_similar", "plan_task"])
        self.assertEqual(result.retrieved_ids, ["seed-1"])

    def test_writeback_persists_runtime_runs(self):
        instruction = "让 drone-1 以 0.35 米高度绕 (0, 0) 半径 0.05 米巡航一圈后返航"
        memory = MissionMemory()
        planner = FakePlanner()
        runner = AgentRunner(planner=planner, simulator=None, memory=memory, writeback=True)
        runner.run(instruction)
        self.assertEqual(memory.size, 1)
        self.assertEqual(memory.retrieve(instruction, k=1)[0]["outcome"], "accepted")


if __name__ == "__main__":
    unittest.main()
