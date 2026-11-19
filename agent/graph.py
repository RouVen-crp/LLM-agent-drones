"""LangGraph runtime for the drone mission-planning agent.

Graph shape (P1, deliberately a workflow agent rather than an autonomous
tool-chooser because the local 0.5B planner cannot reliably route tools):

    START -> plan -> (reject) -> END
                 -> validate -> (simulate) -> END
                             -> (repair -> plan, at most max_attempts)
                             -> finalize_invalid -> END

All capabilities go through ``ToolRegistry``; every step is recorded in the
run trace.  The checkpointer allows a run to be resumed by thread id.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .policy import route_memory
from .tools import ToolCall, ToolRegistry, compact_result, detect_rejection


class AgentState(TypedDict, total=False):
    instruction: str
    max_attempts: int
    use_simulator: bool
    seed_plan_text: str | None
    examples: list[dict[str, Any]] | None
    attempts: int
    feedback: str | None
    router_decision: dict[str, Any] | None
    plan_text: str
    rejected: bool
    rejection_reason: str | None
    valid: bool
    errors: list[str]
    plan: dict[str, Any] | None
    simulation: dict[str, Any] | None
    status: str
    tool_calls: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentResult:
    instruction: str
    status: str
    attempts: int
    plan_text: str
    plan: dict[str, Any] | None
    rejected: bool
    rejection_reason: str | None
    valid: bool
    errors: list[str]
    simulation: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    retrieved_ids: list[str]
    router_decision: dict[str, Any] | None
    latency_seconds: float
    thread_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "status": self.status,
            "attempts": self.attempts,
            "plan_text": self.plan_text,
            "plan": self.plan,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "valid": self.valid,
            "errors": self.errors,
            "simulation": self.simulation,
            "tool_calls": self.tool_calls,
            "retrieved_ids": self.retrieved_ids,
            "router_decision": self.router_decision,
            "latency_seconds": round(self.latency_seconds, 4),
            "thread_id": self.thread_id,
        }


@dataclass
class AgentRunner:
    planner: Any
    simulator: Any | None = None
    max_attempts: int = 3
    checkpointer: Any | None = None
    memory: Any | None = None
    memory_k: int = 2
    writeback: bool = False
    router: Any | None = None
    registry: ToolRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.registry = ToolRegistry(planner=self.planner, simulator=self.simulator)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _trace(self, node: str, call: ToolCall) -> dict[str, Any]:
        return {"node": node, **call.to_dict(), "result": compact_result(call.result)}

    def _plan_node(self, state: AgentState) -> AgentState:
        if state.get("seed_plan_text") and state.get("attempts", 0) == 0:
            plan_text = str(state["seed_plan_text"])
            call = ToolCall(
                tool="seed_plan",
                arguments={"instruction": state.get("instruction", "")},
                ok=True,
                result={"plan_text": plan_text},
                latency_seconds=0.0,
            )
            updates: AgentState = {"seed_plan_text": None}
        else:
            arguments: dict[str, Any] = {"instruction": state.get("instruction", ""), "feedback": state.get("feedback")}
            if state.get("examples"):
                arguments["examples"] = state["examples"]
            call = self.registry.invoke("plan_task", arguments)
            plan_text = call.result.get("plan_text", "") if call.ok else ""
            updates = {}
        rejected, reason = detect_rejection(plan_text)
        updates.update(
            {
                "plan_text": plan_text,
                "rejected": rejected,
                "rejection_reason": reason,
                "attempts": state.get("attempts", 0) + 1,
                "tool_calls": [self._trace("plan", call)],
            }
        )
        return updates

    def _retrieve_node(self, state: AgentState) -> AgentState:
        started = perf_counter()
        instruction = state.get("instruction", "")
        allowed, reason, category = route_memory(instruction)
        examples = self.memory.retrieve(instruction, k=self.memory_k, category=category) if (self.memory is not None and allowed) else []
        call = ToolCall(
            tool="retrieve_similar",
            arguments={"query": instruction, "k": self.memory_k, "category": category},
            ok=True,
            result={
                "ids": [example.get("id") for example in examples],
                "categories": [example.get("category") for example in examples],
                "scores": [example.get("retrieval_score") for example in examples],
                "skipped": None if allowed else reason,
                "category_filter": category,
            },
            latency_seconds=perf_counter() - started,
        )
        return {"examples": examples, "tool_calls": [self._trace("retrieve", call)]}

    def _route_node(self, state: AgentState) -> AgentState:
        started = perf_counter()
        decision = self.router.decide(state.get("instruction", ""))
        call = ToolCall(
            tool="router_decide",
            arguments={"instruction": state.get("instruction", "")},
            ok=bool(decision.get("valid")),
            result={
                "name": decision.get("name"),
                "arguments": decision.get("arguments"),
                "raw": decision.get("raw"),
            },
            latency_seconds=perf_counter() - started,
            error=None if decision.get("valid") else "router 输出不是合法工具调用。",
        )
        updates: AgentState = {"router_decision": decision, "tool_calls": [self._trace("route", call)]}
        if decision.get("name") in ("reject_task", "ask_clarification"):
            arguments = decision.get("arguments") or {}
            updates["rejection_reason"] = arguments.get("reason") or "router 判定需要拒绝或澄清。"
        return updates

    def _validate_node(self, state: AgentState) -> AgentState:
        call = self.registry.invoke("validate_plan", {"plan_text": state.get("plan_text", "")})
        if call.ok:
            result = call.result
            updates: AgentState = {
                "valid": bool(result.get("valid")),
                "errors": list(result.get("errors", [])),
                "plan": result.get("plan"),
            }
        else:
            updates = {"valid": False, "errors": [call.error or "校验工具失败。"]}
        if not updates["valid"]:
            updates["feedback"] = "; ".join(updates["errors"])
        updates["tool_calls"] = [self._trace("validate", call)]
        return updates

    def _simulate_node(self, state: AgentState) -> AgentState:
        call = self.registry.invoke("simulate_plan", {"plan_text": state.get("plan_text", "")})
        if call.ok:
            simulation = call.result
            status = "accepted" if simulation.get("success") else "execution_failed"
        else:
            simulation = {"success": False, "errors": [call.error or "仿真工具失败。"]}
            status = "execution_failed"
        return {"simulation": simulation, "status": status, "tool_calls": [self._trace("simulate", call)]}

    def _reject_node(self, state: AgentState) -> AgentState:
        trace = {
            "node": "reject",
            "tool": "reject_task",
            "arguments": {"reason": state.get("rejection_reason")},
            "ok": True,
            "result": {"status": "needs_clarification", "reason": state.get("rejection_reason")},
            "latency_seconds": 0.0,
            "error": None,
        }
        return {"status": "rejected", "rejected": True, "tool_calls": [trace]}

    def _clarify_node(self, state: AgentState) -> AgentState:
        trace = {
            "node": "clarify",
            "tool": "ask_clarification",
            "arguments": {"reason": state.get("rejection_reason")},
            "ok": True,
            "result": {"status": "needs_clarification", "reason": state.get("rejection_reason")},
            "latency_seconds": 0.0,
            "error": None,
        }
        return {"status": "clarification", "tool_calls": [trace]}

    def _finalize_valid_node(self, state: AgentState) -> AgentState:
        return {"status": "accepted"}

    def _finalize_invalid_node(self, state: AgentState) -> AgentState:
        return {"status": "invalid"}

    # --------------------------------------------------------------- routing
    def _route_start(self, state: AgentState) -> str:
        if self.router is not None:
            return "route"
        if state.get("seed_plan_text"):
            return "plan"
        return "retrieve" if self.memory is not None else "plan"

    def _route_after_router(self, state: AgentState) -> str:
        decision = state.get("router_decision") or {}
        name = decision.get("name")
        if name == "plan_task":
            return "plan"
        if name == "reject_task":
            return "reject"
        if name == "ask_clarification":
            return "clarify"
        return "invalid"

    def _route_after_plan(self, state: AgentState) -> str:
        return "reject" if state.get("rejected") else "validate"

    def _route_after_validate(self, state: AgentState) -> str:
        if state.get("valid"):
            return "simulate" if state.get("use_simulator") else "finalize_valid"
        if state.get("attempts", 0) >= state.get("max_attempts", self.max_attempts):
            return "finalize_invalid"
        return "plan"

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("simulate", self._simulate_node)
        graph.add_node("reject", self._reject_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("finalize_valid", self._finalize_valid_node)
        graph.add_node("finalize_invalid", self._finalize_invalid_node)

        graph.add_conditional_edges(START, self._route_start, {"route": "route", "retrieve": "retrieve", "plan": "plan"})
        graph.add_conditional_edges(
            "route",
            self._route_after_router,
            {"plan": "plan", "reject": "reject", "clarify": "clarify", "invalid": "finalize_invalid"},
        )
        graph.add_edge("retrieve", "plan")
        graph.add_conditional_edges("plan", self._route_after_plan, {"validate": "validate", "reject": "reject"})
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"plan": "plan", "simulate": "simulate", "finalize_valid": "finalize_valid", "finalize_invalid": "finalize_invalid"},
        )
        graph.add_edge("simulate", END)
        graph.add_edge("reject", END)
        graph.add_edge("clarify", END)
        graph.add_edge("finalize_valid", END)
        graph.add_edge("finalize_invalid", END)
        return graph.compile(checkpointer=self.checkpointer or MemorySaver())

    # ------------------------------------------------------------------ run
    def stream(self, instruction: str, seed_plan_text: str | None = None, thread_id: str | None = None):
        """Yield per-node updates for SSE / CLI progress (same graph state)."""
        thread = thread_id or f"run-{uuid.uuid4().hex[:12]}"
        initial: AgentState = {
            "instruction": instruction,
            "max_attempts": self.max_attempts,
            "use_simulator": self.simulator is not None,
            "attempts": 0,
            "tool_calls": [],
            "seed_plan_text": seed_plan_text,
            "examples": [],
            "router_decision": None,
        }
        for chunk in self._graph.stream(initial, config={"configurable": {"thread_id": thread}}, stream_mode="updates"):
            for node, update in chunk.items():
                yield {"thread_id": thread, "node": node, "update": update}

    def run(self, instruction: str, seed_plan_text: str | None = None, thread_id: str | None = None) -> AgentResult:
        thread = thread_id or f"run-{uuid.uuid4().hex[:12]}"
        initial: AgentState = {
            "instruction": instruction,
            "max_attempts": self.max_attempts,
            "use_simulator": self.simulator is not None,
            "attempts": 0,
            "tool_calls": [],
            "seed_plan_text": seed_plan_text,
            "examples": [],
            "router_decision": None,
        }
        started = perf_counter()
        final = self._graph.invoke(initial, config={"configurable": {"thread_id": thread}})
        elapsed = perf_counter() - started
        if self.writeback and self.memory is not None:
            target = final.get("plan")
            expected = None
            if final.get("rejected"):
                expected = {"status": "needs_clarification", "reason": final.get("rejection_reason") or ""}
            self.memory.add(
                {
                    "id": thread,
                    "instruction": instruction,
                    "category": "runtime",
                    "plan": target,
                    "expected_response": expected,
                },
                outcome=str(final.get("status", "unknown")),
            )
        retrieved_ids = [
            identifier
            for call in final.get("tool_calls", [])
            if call.get("tool") == "retrieve_similar"
            for identifier in (call.get("result", {}).get("ids") or [])
        ]
        return AgentResult(
            instruction=instruction,
            status=str(final.get("status", "unknown")),
            attempts=int(final.get("attempts", 0)),
            plan_text=str(final.get("plan_text", "")),
            plan=final.get("plan"),
            rejected=bool(final.get("rejected")),
            rejection_reason=final.get("rejection_reason"),
            valid=bool(final.get("valid")),
            errors=list(final.get("errors", [])),
            simulation=final.get("simulation"),
            tool_calls=list(final.get("tool_calls", [])),
            retrieved_ids=retrieved_ids,
            router_decision=final.get("router_decision"),
            latency_seconds=elapsed,
            thread_id=thread,
        )
