"""FastAPI service for the drone mission-planning agent.

Endpoints
- GET  /healthz                     liveness + tool names
- GET  /v1/tools                    tool schemas (for UI / MCP-style clients)
- POST /v1/missions                 run one mission synchronously
- POST /v1/missions/stream          run one mission and stream node updates (SSE)
- GET  /v1/missions/{run_id}        fetch a persisted run
- GET  /v1/missions                 recent runs
- POST /v1/tools/{name}             invoke one tool directly (debug / integration)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import AgentRunner
from agent.tools import ToolError

from .store import RunStore

LOGGER = logging.getLogger("agent.service")


class MissionRequest(BaseModel):
    instruction: str
    simulate: bool = False


def _merge_event(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "tool_calls" and isinstance(value, list):
            state.setdefault("tool_calls", [])
            state["tool_calls"].extend(value)
        else:
            state[key] = value


def create_app(planner: Any, simulator: Any | None = None, store_path: Path | str = Path("artifacts/agent_runs/runs.sqlite"), router: Any | None = None) -> FastAPI:
    store = RunStore(store_path)
    runner_plain = AgentRunner(planner=planner, simulator=None, router=router)
    runner_sim = AgentRunner(planner=planner, simulator=simulator, router=router) if simulator is not None else None
    app = FastAPI(title="Drone Mission Planning Agent", version="0.1.0")

    def select_runner(simulate: bool) -> AgentRunner:
        if simulate and runner_sim is None:
            raise HTTPException(status_code=400, detail="service 启动时未配置执行器，无法 simulate。")
        return runner_sim if simulate else runner_plain

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "tools": [spec["name"] for spec in runner_plain.registry.specs()]}

    @app.get("/v1/tools")
    def tools() -> list[dict[str, Any]]:
        return runner_plain.registry.specs()

    @app.post("/v1/missions")
    def run_mission(request: MissionRequest) -> dict[str, Any]:
        runner = select_runner(request.simulate)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        LOGGER.info("mission start run_id=%s simulate=%s", run_id, request.simulate)
        result = runner.run(request.instruction, thread_id=run_id)
        payload = result.to_dict()
        store.save(run_id, request.instruction, payload)
        LOGGER.info("mission done run_id=%s status=%s attempts=%s", run_id, result.status, result.attempts)
        return payload

    @app.post("/v1/missions/stream")
    def stream_mission(request: MissionRequest) -> StreamingResponse:
        runner = select_runner(request.simulate)
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        def event_stream():
            state: dict[str, Any] = {}
            for event in runner.stream(request.instruction, thread_id=run_id):
                _merge_event(state, event.get("update", {}))
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            summary = {
                "run_id": run_id,
                "status": state.get("status", "unknown"),
                "attempts": state.get("attempts", 0),
                "valid": state.get("valid", False),
                "rejected": state.get("rejected", False),
                "simulation": state.get("simulation"),
                "tool_calls": state.get("tool_calls", []),
            }
            store.save(run_id, request.instruction, summary)
            yield f"event: done\ndata: {json.dumps(summary, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/missions/{run_id}")
    def get_mission(run_id: str) -> dict[str, Any]:
        record = store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.get("/v1/missions")
    def recent_missions(limit: int = 20) -> list[dict[str, Any]]:
        return store.recent(limit=limit)

    @app.post("/v1/tools/{name}")
    def invoke_tool(name: str, arguments: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        registry = (runner_sim or runner_plain).registry
        try:
            call = registry.invoke(name, arguments)
        except ToolError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return call.to_dict()

    app.state.store = store
    return app
