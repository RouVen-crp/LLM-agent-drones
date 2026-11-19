from __future__ import annotations

import json

from .contracts import EXECUTION_BOUNDS, MAX_DRONES, MAX_WAYPOINTS


CONTRACT_EXAMPLE = '''Output-shape example (structure only):
{"schema_version":"1.0","mission_id":"mission-0001","mission":{"coordinate_frame":"local_meters","selected_drone_id":"drone-1"},"drone_plans":[{"drone_id":"drone-1","task_type":"point_to_point","waypoints":[[0.0,0.0,0.3]],"return_home":true,"altitude_min":0.2,"altitude_max":0.5}],"metadata":{"source":"planner","category":"point"}}'''


SYSTEM_PROMPT = f"""你是无人机任务规划器。把中文任务指令转换为唯一的 JSON 对象，不输出解释、Markdown 或 Python 代码。

契约：schema_version 固定为 1.0；mission.coordinate_frame 固定为 local_meters；drone_plans 数量 1 到 {MAX_DRONES}；每个计划含 drone_id、task_type、waypoints、return_home、altitude_min、altitude_max；mission.selected_drone_id 必须匹配一个 drone_id。

只允许 task_type：point_to_point、multi_waypoint、circle_patrol、rectangle_patrol。
每条轨迹最多 {MAX_WAYPOINTS} 个航点。x、y 均必须在 {EXECUTION_BOUNDS['x']} 内，z 必须在 {EXECUTION_BOUNDS['z']} 内。若用户的指令含糊或要求超出这些范围，输出 {json.dumps({'status': 'needs_clarification', 'reason': '说明原因'}, ensure_ascii=False)}，不要编造计划。"""


def build_messages(instruction: str, feedback: str | None = None, examples: list[dict] | None = None) -> list[dict[str, str]]:
    """Build the single Qwen chat contract used for both SFT and inference."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + CONTRACT_EXAMPLE}]
    for example in examples or []:
        target = example.get("expected_response") or example["plan"]
        messages.extend([
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ])
    user_content = instruction
    if feedback:
        user_content += "\n\n上一次计划未通过校验。只修复下列问题后重新输出完整 JSON：" + feedback
    messages.append({"role": "user", "content": user_content})
    return messages


def training_messages(record: dict) -> list[dict[str, str]]:
    """Return one training conversation, with the target JSON as the final turn."""
    messages = build_messages(record["instruction"])
    # Normal records carry ``expected_response: null`` while rejection
    # records provide an explicit response.  A null optional value must not
    # replace the plan target during SFT.
    target = record.get("expected_response") or record["plan"]
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))})
    return messages
