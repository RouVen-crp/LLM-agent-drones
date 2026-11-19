"""Command-line entry point for one agent run.

Examples:
    python -m agent --print-tools
    python -m agent --text "让 drone-1 飞到 (0.05, 0.0, 0.35) 后返回起点" --simulate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drone_planner.planner import TransformersPlanner
from drone_planner.simulation import PIDSimulator

from .graph import AgentRunner
from .tools import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一次无人机任务规划 Agent。")
    parser.add_argument("--text", help="中文任务指令。")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/lora_adapter_contract_v4"))
    parser.add_argument("--simulate", action="store_true", help="对通过校验的计划运行 PID 单机仿真。")
    parser.add_argument("--router", action="store_true", help="启用 P2 函数调用 router 选择工具。")
    parser.add_argument("--router-adapter", type=Path, default=Path("artifacts/lora_router_contract_v1"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/agent_runs/last_run.json"))
    parser.add_argument("--print-tools", action="store_true", help="只打印工具 schema 后退出。")
    args = parser.parse_args()

    if args.print_tools:
        print(json.dumps(ToolRegistry().specs(), ensure_ascii=False, indent=2))
        return
    if not args.text:
        parser.error("--text 是必填参数（或用 --print-tools 查看工具）。")

    planner = TransformersPlanner(model_name=args.model, adapter_path=args.adapter)
    simulator = PIDSimulator() if args.simulate else None
    router = None
    if args.router:
        from .router import ToolRouter

        router = ToolRouter(model_name=args.model, adapter_path=args.router_adapter)
    runner = AgentRunner(planner=planner, simulator=simulator, max_attempts=args.max_attempts, router=router)
    result = runner.run(args.text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result.status,
                "attempts": result.attempts,
                "valid": result.valid,
                "rejected": result.rejected,
                "simulation_success": (result.simulation or {}).get("success"),
                "latency_seconds": round(result.latency_seconds, 3),
                "tool_calls": [call["tool"] for call in result.tool_calls],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
