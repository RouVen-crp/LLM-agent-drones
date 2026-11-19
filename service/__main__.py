"""Run the agent service: python -m service --adapter artifacts/lora_adapter_contract_v4 --simulate."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from drone_planner.planner import TransformersPlanner
from drone_planner.simulation import PIDSimulator

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动无人机任务规划 Agent 服务。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/lora_adapter_contract_v4"))
    parser.add_argument("--store", type=Path, default=Path("artifacts/agent_runs/runs.sqlite"))
    parser.add_argument("--no-simulator", action="store_true", help="不加载 PID 执行器。")
    parser.add_argument("--router", action="store_true", help="启用 P2 函数调用 router。")
    parser.add_argument("--router-adapter", type=Path, default=Path("artifacts/lora_router_contract_v1"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    planner = TransformersPlanner(model_name=args.model, adapter_path=args.adapter)
    simulator = None if args.no_simulator else PIDSimulator()
    router = None
    if args.router:
        from agent.router import ToolRouter

        router = ToolRouter(model_name=args.model, adapter_path=args.router_adapter)
    app = create_app(planner=planner, simulator=simulator, store_path=args.store, router=router)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
