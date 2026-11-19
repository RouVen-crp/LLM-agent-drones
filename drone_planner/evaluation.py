from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from .planner import Planner
from .validation import ValidationResult, parse_json_object, validate_text


def reference_plan_mismatches(plan, reference: dict, tolerance: float = 1e-3) -> list[str]:
    """Return field-level differences against a held-out structured-flight-plan label."""
    expected = reference["plan"]
    expected_plans = {item["drone_id"]: item for item in expected["drone_plans"]}
    mismatches: list[str] = []
    if plan.selected_drone_id != expected["mission"]["selected_drone_id"]:
        mismatches.append("selected_drone_id")
    actual_ids = {item.drone_id for item in plan.drone_plans}
    if actual_ids != set(expected_plans):
        mismatches.append("drone_set")
    for actual in plan.drone_plans:
        if actual.drone_id not in expected_plans:
            continue
        target = expected_plans[actual.drone_id]
        prefix = actual.drone_id
        if actual.task_type != target["task_type"]:
            mismatches.append(f"{prefix}.task_type")
        if actual.return_home != target["return_home"]:
            mismatches.append(f"{prefix}.return_home")
        if len(actual.waypoints) != len(target["waypoints"]):
            mismatches.append(f"{prefix}.waypoint_count")
        else:
            for actual_point, target_point in zip(actual.waypoints, target["waypoints"]):
                if any(abs(actual_value - target_value) > tolerance for actual_value, target_value in zip(actual_point, target_point)):
                    mismatches.append(f"{prefix}.waypoint_value")
                    break
        if abs(actual.altitude_min - target["altitude_min"]) > tolerance or abs(actual.altitude_max - target["altitude_max"]) > tolerance:
            mismatches.append(f"{prefix}.altitude_bounds")
    return mismatches


def matches_reference_plan(plan, reference: dict, tolerance: float = 1e-3) -> bool:
    """Check task semantics against the held-out label, not just safety bounds."""
    return not reference_plan_mismatches(plan, reference, tolerance)


@dataclass
class EvaluationRow:
    id: str
    category: str
    variant: str
    expected_rejection: bool
    parse_valid: bool
    json_object_valid: bool
    constraint_valid: bool
    semantic_correct: bool | None
    semantic_mismatches: list[str]
    explicit_rejection: bool
    safe_rejection: bool
    valid_task_false_rejection: bool
    simulation_success: bool | None
    attempts: int
    latency_seconds: float
    feedback: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_with_feedback(planner: Planner, instruction: str, examples: list[dict] | None = None, max_attempts: int = 3) -> tuple[ValidationResult, int, float, str]:
    feedback = None
    started = perf_counter()
    result = ValidationResult(False, ["尚未生成。"])
    raw_response = ""
    for attempt in range(1, max_attempts + 1):
        raw_response = planner.generate(instruction, feedback, examples)
        result = validate_text(raw_response)
        if result.valid:
            return result, attempt, perf_counter() - started, raw_response
        feedback = result.feedback()
    return result, max_attempts, perf_counter() - started, raw_response


def _row(record: dict, variant: str, result: ValidationResult, attempts: int, latency: float, raw_response: str, simulator=None) -> EvaluationRow:
    is_invalid = bool(record.get("expect_rejection", False))
    try:
        payload = parse_json_object(raw_response)
        json_object_valid = True
    except ValueError:
        payload = {}
        json_object_valid = False
    explicit_rejection = payload.get("status") == "needs_clarification" and isinstance(payload.get("reason"), str) and bool(payload["reason"].strip())
    mismatches = reference_plan_mismatches(result.plan, record) if result.valid and not is_invalid else []
    semantic_correct = (not mismatches) if result.valid and not is_invalid else None
    simulation_success = None
    if semantic_correct and simulator:
        selected = next(item for item in result.plan.drone_plans if item.drone_id == result.plan.selected_drone_id)
        simulation_success = simulator.run(selected).success
    return EvaluationRow(
        id=record["id"], category=record["category"], variant=variant, expected_rejection=is_invalid,
        parse_valid=result.valid, json_object_valid=json_object_valid, constraint_valid=result.valid,
        semantic_correct=semantic_correct, semantic_mismatches=mismatches,
        explicit_rejection=explicit_rejection, safe_rejection=is_invalid and explicit_rejection,
        valid_task_false_rejection=not is_invalid and explicit_rejection,
        simulation_success=simulation_success, attempts=attempts, latency_seconds=latency,
        feedback=result.feedback(), raw_response=raw_response,
    )


def evaluate_records(records: list[dict], planner: Planner, variant: str, output_path: Path, examples: list[dict] | None = None, simulator=None, resume: bool = False) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if resume and output_path.exists():
        completed = {json.loads(line)["id"] for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    with output_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for record in records:
            if record["id"] in completed:
                continue
            max_attempts = 3 if variant == "lora_feedback" else 1
            active_examples = examples if variant == "few_shot" else None
            result, attempts, latency, raw_response = generate_with_feedback(planner, record["instruction"], active_examples, max_attempts)
            row = _row(record, variant, result, attempts, latency, raw_response, simulator)
            rows.append(row)
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
    return rows


def evaluate_records_batched(records: list[dict], planner, variant: str, output_path: Path, examples: list[dict] | None = None, resume: bool = False, batch_size: int = 8) -> list[EvaluationRow]:
    """Fast path for deterministic, no-feedback, no-simulation comparisons."""
    rows: list[EvaluationRow] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if resume and output_path.exists():
        completed = {json.loads(line)["id"] for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [record for record in records if record["id"] not in completed]
    with output_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            started = perf_counter()
            raw_responses = planner.generate_many([record["instruction"] for record in batch], examples)
            latency = (perf_counter() - started) / len(batch)
            for record, raw_response in zip(batch, raw_responses):
                row = _row(record, variant, validate_text(raw_response), 1, latency, raw_response)
                rows.append(row)
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
    return rows
