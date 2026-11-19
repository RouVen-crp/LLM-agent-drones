"""Prompt contract for the function-calling router."""

from __future__ import annotations

import json
from typing import Any

ROUTER_SYSTEM = """你是无人机任务规划系统的路由器。根据用户的中文指令选择下一步要调用的工具。

可用工具：
1. plan_task：指令明确、安全、可执行时使用。参数 {"instruction": "原始指令"}
2. reject_task：指令越界、自相矛盾、要求执行代码、超过 3 架无人机或要求不支持的能力时使用。参数 {"reason": "原因"}
3. ask_clarification：指令缺少关键信息（目标位置、高度、禁飞区、任务内容）时使用。参数 {"reason": "需要用户补充什么"}
4. get_contract：用户询问系统能力或约束时使用。参数 {}

只输出一个 JSON 对象，格式为 {"name": "工具名", "arguments": {...}}。
不要输出解释、Markdown 或代码块。"""


def build_router_messages(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": instruction},
    ]


def apply_template(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        build_router_messages(instruction), tokenize=False, add_generation_prompt=True
    )


def target_tool_call(row: dict[str, Any]) -> str:
    """Training target for one router sample."""
    return str(row["target"])


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object; returns None if malformed."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    if not isinstance(payload.get("arguments"), dict):
        payload["arguments"] = {}
    return payload
