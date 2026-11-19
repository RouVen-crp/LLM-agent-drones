"""Tool layer for the drone mission-planning agent.

Every capability the agent can invoke is declared here with a JSON-Schema-like
contract and executed through one registry.  The registry is deliberately
framework-agnostic: the LangGraph runtime only calls ``invoke`` and records the
returned ``ToolCall`` in the run trace.  A later FastAPI/MCP layer can expose
``specs()`` and ``invoke()`` without rewriting the tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from drone_planner.contracts import EXECUTION_BOUNDS, MAX_DRONES, MAX_WAYPOINTS, SUPPORTED_TASK_TYPES
from drone_planner.validation import parse_json_object, validate_text


class ToolError(ValueError):
    """Raised when a tool is called with an unknown name or invalid arguments."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "latency_seconds": round(self.latency_seconds, 4),
            "error": self.error,
        }


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    expected_list = [expected] if isinstance(expected, str) else expected
    for item in expected_list:
        if item == "string" and isinstance(value, str):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "null" and value is None:
            return True
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
    return False


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolError(f"{spec.name} 的参数必须是对象。")
    properties = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])
    for key in required:
        if key not in arguments:
            raise ToolError(f"{spec.name} 缺少必填参数：{key}")
    for key, value in arguments.items():
        if key not in properties:
            raise ToolError(f"{spec.name} 收到未声明参数：{key}")
        expected = properties[key].get("type")
        if expected is not None and not _type_ok(value, expected):
            raise ToolError(f"{spec.name} 参数 {key} 类型不匹配，期望 {expected}。")


class ToolRegistry:
    """Declares and executes the agent's tools.

    ``planner`` only needs a ``generate(instruction, feedback)`` method, which
    keeps the registry testable with a fake planner.  ``simulator`` only needs
    a ``run(DronePlan)`` method.
    """

    def __init__(self, planner: Any | None = None, simulator: Any | None = None) -> None:
        self._planner = planner
        self._simulator = simulator
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {}
        self._register_all()

    # ------------------------------------------------------------------ specs
    def _register_all(self) -> None:
        self._add(
            "plan_task",
            "把中文无人机指令转换为结构化飞行计划 JSON；含糊或越界任务返回 needs_clarification。",
            {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "feedback": {"type": ["string", "null"]},
                    "examples": {"type": ["array", "null"]},
                },
                "required": ["instruction"],
            },
            self._plan_task,
        )
        self._add(
            "validate_plan",
            "对飞行计划执行 JSON、字段契约与执行空间校验。",
            {"type": "object", "properties": {"plan_text": {"type": "string"}}, "required": ["plan_text"]},
            self._validate_plan,
        )
        self._add(
            "simulate_plan",
            "用单机 PID 在 PyBullet 中执行已通过校验的计划，返回是否到达全部航点。",
            {"type": "object", "properties": {"plan_text": {"type": "string"}}, "required": ["plan_text"]},
            self._simulate_plan,
        )
        self._add(
            "get_contract",
            "返回当前 Planner 的 schema、执行空间与任务类型约束。",
            {"type": "object", "properties": {}, "required": []},
            self._get_contract,
        )

    def _add(self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., dict[str, Any]]) -> None:
        self._specs[name] = ToolSpec(name=name, description=description, parameters=parameters)
        self._handlers[name] = handler

    # --------------------------------------------------------------- execution
    def specs(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
        arguments = arguments or {}
        spec = self._specs.get(name)
        if spec is None:
            raise ToolError(f"未知工具：{name}")
        _validate_arguments(spec, arguments)
        started = perf_counter()
        try:
            result = self._handlers[name](**arguments)
            return ToolCall(tool=name, arguments=arguments, ok=True, result=result, latency_seconds=perf_counter() - started)
        except Exception as error:  # tool failures are part of the trace, not crashes
            return ToolCall(
                tool=name,
                arguments=arguments,
                ok=False,
                result={"error": f"{type(error).__name__}: {error}"},
                latency_seconds=perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )

    # ----------------------------------------------------------------- handlers
    def _plan_task(self, instruction: str, feedback: str | None = None, examples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self._planner is None:
            raise ToolError("plan_task 未配置 planner。")
        return {"plan_text": self._planner.generate(instruction, feedback, examples)}

    def _validate_plan(self, plan_text: str) -> dict[str, Any]:
        result = validate_text(plan_text)
        return {
            "valid": result.valid,
            "errors": result.errors,
            "plan": result.plan.to_dict() if result.plan is not None else None,
        }

    def _simulate_plan(self, plan_text: str) -> dict[str, Any]:
        if self._simulator is None:
            raise ToolError("simulate_plan 未启用：当前运行没有配置执行器。")
        result = validate_text(plan_text)
        if not result.valid or result.plan is None:
            raise ToolError("simulate_plan 只接受已通过校验的计划。")
        selected = next(item for item in result.plan.drone_plans if item.drone_id == result.plan.selected_drone_id)
        simulation = self._simulator.run(selected)
        return {
            "success": simulation.success,
            "reached_waypoints": simulation.reached_waypoints,
            "total_waypoints": simulation.total_waypoints,
            "max_error": simulation.max_error,
            "final_error": simulation.final_error,
            "terminated_early": simulation.terminated_early,
            "errors": simulation.errors,
            "trajectory_points": len(simulation.trajectory),
        }

    def _get_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "execution_bounds": EXECUTION_BOUNDS,
            "max_drones": MAX_DRONES,
            "max_waypoints": MAX_WAYPOINTS,
            "task_types": sorted(SUPPORTED_TASK_TYPES),
        }


def detect_rejection(plan_text: str) -> tuple[bool, str | None]:
    """Return (is_explicit_rejection, reason) for a planner response."""
    try:
        payload = parse_json_object(plan_text)
    except ValueError:
        return False, None
    status = payload.get("status")
    reason = payload.get("reason")
    if status == "needs_clarification" and isinstance(reason, str) and reason.strip():
        return True, reason
    return False, None


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep tool traces readable by trimming the long plan text."""
    compact = dict(result)
    plan_text = compact.get("plan_text")
    if isinstance(plan_text, str) and len(plan_text) > 600:
        compact["plan_text"] = plan_text[:600] + "..."
    plan = compact.get("plan")
    if isinstance(plan, dict):
        compact["plan"] = {"mission_id": plan.get("mission_id"), "drone_plans": len(plan.get("drone_plans", []))}
    examples = compact.get("examples")
    if isinstance(examples, list):
        compact["examples"] = [
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "instruction": (item.get("instruction") or "")[:80],
            }
            for item in examples
            if isinstance(item, dict)
        ]
    return compact


def dumps_specs(registry: ToolRegistry) -> str:
    return json.dumps(registry.specs(), ensure_ascii=False, indent=2)
