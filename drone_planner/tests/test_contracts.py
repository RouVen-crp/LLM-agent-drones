import unittest
import json
import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from drone_planner.dataset import generate_records, split_records, write_dataset
from drone_planner.evaluation import evaluate_records_batched, matches_reference_plan, reference_plan_mismatches
from drone_planner.experiments import _few_shot_examples, _summary
from drone_planner.prompts import build_messages, training_messages
from drone_planner.validation import validate_payload


class ContractTests(unittest.TestCase):
    def test_generated_records_follow_contract(self):
        for record in generate_records():
            result = validate_payload(record["plan"])
            self.assertTrue(result.valid, result.feedback())

    def test_split_has_expected_sizes(self):
        splits = split_records(generate_records())
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 400, "validation": 50, "test": 50})

    def test_out_of_bounds_plan_is_rejected(self):
        record = generate_records()[0]
        record["plan"]["drone_plans"][0]["waypoints"][0][0] = 9.0
        result = validate_payload(record["plan"])
        self.assertFalse(result.valid)
        self.assertIn("x 超出执行空间", result.feedback())

    def test_training_and_inference_share_the_same_chat_contract(self):
        record = generate_records()[0]
        messages = training_messages(record)
        self.assertEqual(messages[:-1], build_messages(record["instruction"]))
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], __import__("json").dumps(record["plan"], ensure_ascii=False, separators=(",", ":")))

    def test_dataset_null_placeholder_does_not_replace_a_normal_plan_target(self):
        record = generate_records()[0]
        record["expected_response"] = None
        self.assertNotEqual(training_messages(record)[-1]["content"], "null")
        self.assertEqual(training_messages(record)[-1]["content"], __import__("json").dumps(record["plan"], ensure_ascii=False, separators=(",", ":")))

    def test_semantic_check_rejects_a_safe_but_wrong_waypoint(self):
        record = generate_records()[0]
        result = validate_payload(record["plan"])
        self.assertTrue(matches_reference_plan(result.plan, record))
        record["plan"]["drone_plans"][0]["waypoints"][0][0] += 0.01
        self.assertFalse(matches_reference_plan(result.plan, record))

    def test_semantic_mismatches_name_the_wrong_field(self):
        record = generate_records()[0]
        result = validate_payload(record["plan"])
        record["plan"]["drone_plans"][0]["return_home"] = not record["plan"]["drone_plans"][0]["return_home"]
        self.assertIn("drone-1.return_home", reference_plan_mismatches(result.plan, record))

    def test_summary_counts_missing_semantic_match_as_a_failure(self):
        rows = [{"expected_rejection": False, "parse_valid": False, "json_object_valid": False, "constraint_valid": False, "semantic_correct": None, "safe_rejection": False, "simulation_success": None, "attempts": 1, "latency_seconds": 0.1}]
        self.assertEqual(_summary(rows)["semantic_plan_accuracy"], 0.0)

    def test_summary_reports_valid_task_false_rejections(self):
        rows = [{"expected_rejection": False, "parse_valid": False, "json_object_valid": True, "constraint_valid": False, "semantic_correct": None, "safe_rejection": False, "valid_task_false_rejection": True, "simulation_success": None, "attempts": 1, "latency_seconds": 0.1}]
        self.assertEqual(_summary(rows)["valid_task_false_rejection_rate"], 1.0)

    def test_batched_evaluation_preserves_per_record_semantics(self):
        records = generate_records()[:2]
        responses = {record["instruction"]: json.dumps(record["plan"], ensure_ascii=False) for record in records}

        class BatchPlanner:
            def generate_many(self, instructions, examples=None):
                return [responses[instruction] for instruction in instructions]

        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_records_batched(records, BatchPlanner(), "test", Path(directory) / "rows.jsonl", batch_size=2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.semantic_correct for row in rows))

    def test_v4_dataset_separates_unsafe_template_families(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset(output, include_training_rejections=True)
            splits = {name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()] for name in ("train", "validation", "test")}
        families = {name: {row["risk_family"] for row in rows if row.get("expect_rejection")} for name, rows in splits.items()}
        self.assertFalse(families["train"] & families["validation"])
        self.assertFalse(families["train"] & families["test"])
        self.assertFalse(families["validation"] & families["test"])

    def test_v4_instructions_expose_return_and_rectangle_size(self):
        records = generate_records()
        point = next(row for row in records if row["category"] == "point")
        rectangle = next(row for row in records if row["category"] == "rectangle")
        self.assertIn("返回起点", point["instruction"])
        self.assertIn("半边长", rectangle["instruction"])

    def test_few_shot_examples_are_valid_and_diverse(self):
        examples = _few_shot_examples(generate_records())
        self.assertGreaterEqual(len(examples), 4)
        self.assertFalse(any(row.get("expect_rejection") for row in examples))
        self.assertEqual(len({row["category"] for row in examples}), len(examples))

    def test_importing_sources_has_no_network_side_effect(self):
        sys.modules.pop("sources", None)
        with patch("requests.post") as post:
            importlib.import_module("sources")
        post.assert_not_called()



if __name__ == "__main__":
    unittest.main()
