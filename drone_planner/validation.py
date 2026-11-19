from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contracts import EXECUTION_BOUNDS, MAX_DRONES, MAX_WAYPOINTS, MissionPlan, SUPPORTED_TASK_TYPES


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    plan: MissionPlan | None = None

    def feedback(self) -> str:
        return "; ".join(self.errors) if self.errors else "计划已通过确定性校验。"


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object without evaluating model-produced code."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型输出中未找到有效 JSON 对象。")


def validate_payload(payload: dict[str, Any]) -> ValidationResult:
    try:
        plan = MissionPlan.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        return ValidationResult(False, [f"JSON 字段或类型错误：{error}"], None)
    return validate_plan(plan)


def validate_plan(plan: MissionPlan) -> ValidationResult:
    errors: list[str] = []
    if plan.schema_version != "1.0":
        errors.append("schema_version 必须为 1.0。")
    if plan.coordinate_frame != "local_meters":
        errors.append("coordinate_frame 必须为 local_meters。")
    if not plan.mission_id.strip():
        errors.append("mission_id 不能为空。")
    if not 1 <= len(plan.drone_plans) <= MAX_DRONES:
        errors.append(f"drone_plans 数量必须在 1 到 {MAX_DRONES} 之间。")

    drone_ids = [item.drone_id for item in plan.drone_plans]
    if len(drone_ids) != len(set(drone_ids)):
        errors.append("drone_id 必须唯一。")
    if plan.selected_drone_id not in drone_ids:
        errors.append("selected_drone_id 必须指向一个已有的 drone_id。")

    for drone in plan.drone_plans:
        prefix = f"{drone.drone_id}："
        if drone.task_type not in SUPPORTED_TASK_TYPES:
            errors.append(prefix + "task_type 不受支持。")
        if not 1 <= len(drone.waypoints) <= MAX_WAYPOINTS:
            errors.append(prefix + f"航点数量必须在 1 到 {MAX_WAYPOINTS} 之间。")
        if not EXECUTION_BOUNDS["z"][0] <= drone.altitude_min <= drone.altitude_max <= EXECUTION_BOUNDS["z"][1]:
            errors.append(prefix + "高度约束超出执行空间。")
        for waypoint_index, point in enumerate(drone.waypoints, start=1):
            if len(point) != 3:
                errors.append(prefix + f"第 {waypoint_index} 个航点必须是 [x, y, z]。")
                continue
            x, y, z = point
            if not EXECUTION_BOUNDS["x"][0] <= x <= EXECUTION_BOUNDS["x"][1]:
                errors.append(prefix + f"第 {waypoint_index} 个航点 x 超出执行空间。")
            if not EXECUTION_BOUNDS["y"][0] <= y <= EXECUTION_BOUNDS["y"][1]:
                errors.append(prefix + f"第 {waypoint_index} 个航点 y 超出执行空间。")
            if not EXECUTION_BOUNDS["z"][0] <= z <= EXECUTION_BOUNDS["z"][1]:
                errors.append(prefix + f"第 {waypoint_index} 个航点 z 超出执行空间。")
            if not drone.altitude_min <= z <= drone.altitude_max:
                errors.append(prefix + f"第 {waypoint_index} 个航点不满足高度约束。")
    return ValidationResult(not errors, errors, plan if not errors else None)


def validate_text(text: str) -> ValidationResult:
    try:
        return validate_payload(parse_json_object(text))
    except ValueError as error:
        return ValidationResult(False, [str(error)], None)
