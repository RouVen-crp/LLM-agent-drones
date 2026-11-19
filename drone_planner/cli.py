from __future__ import annotations

import argparse
import json
from pathlib import Path

from .planner import TransformersPlanner
from .simulation import PIDSimulator, TD3Simulator
from .validation import validate_text


def main() -> None:
    parser = argparse.ArgumentParser(description="自然语言无人机任务规划与单机仿真验证。")
    parser.add_argument("--text", required=True, help="中文任务指令。音频接口预留在 --audio 参数中。")
    parser.add_argument("--audio", type=Path, help="预留的音频输入接口；当前不会转写音频。")
    parser.add_argument("--adapter", type=Path, help="可选的 LoRA adapter 路径。")
    parser.add_argument("--simulate", action="store_true", help="在单无人机 PyBullet 环境中执行 selected_drone_id。")
    parser.add_argument("--simulator", choices=["pid", "td3"], default="pid", help="执行基线；默认 PID，TD3 用于对照。")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/latest.json"))
    args = parser.parse_args()
    if args.audio:
        raise SystemExit("音频接口已预留但尚未启用；请先使用 --text 运行稳定演示链路。")
    planner = TransformersPlanner(adapter_path=args.adapter)
    result = validate_text(planner.generate(args.text))
    report = {"instruction": args.text, "valid": result.valid, "feedback": result.feedback()}
    if result.valid:
        report["plan"] = result.plan.to_dict()
        if args.simulate:
            selected = next(item for item in result.plan.drone_plans if item.drone_id == result.plan.selected_drone_id)
            simulator = PIDSimulator(gui=args.gui) if args.simulator == "pid" else TD3Simulator(gui=args.gui)
            report["simulation_backend"] = args.simulator
            report["simulation"] = simulator.run(selected).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
