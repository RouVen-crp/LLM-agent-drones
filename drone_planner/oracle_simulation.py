from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .contracts import MissionPlan
from .simulation import PIDSimulator, TD3Simulator


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="用测试集标注计划校准单无人机执行器。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset_contract_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/oracle_simulation"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--simulator", choices=["pid", "td3"], default="pid")
    args = parser.parse_args()

    records = [record for record in _read_jsonl(args.dataset_dir / "test.jsonl") if not record.get("expect_rejection")]
    records = records[args.start:args.end]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "oracle.jsonl"
    completed = set()
    if args.resume and output_path.exists():
        completed = {row["id"] for row in _read_jsonl(output_path)}

    simulator = PIDSimulator() if args.simulator == "pid" else TD3Simulator()
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for record in records:
            if record["id"] in completed:
                continue
            mission = MissionPlan.from_dict(record["plan"])
            selected = next(plan for plan in mission.drone_plans if plan.drone_id == mission.selected_drone_id)
            started = perf_counter()
            result = simulator.run(selected)
            row = {
                "id": record["id"],
                "category": record["category"],
                "success": result.success,
                "reached_waypoints": result.reached_waypoints,
                "total_waypoints": result.total_waypoints,
                "max_error": result.max_error,
                "final_error": result.final_error,
                "terminated_early": result.terminated_early,
                "errors": result.errors,
                "latency_seconds": perf_counter() - started,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    rows = _read_jsonl(output_path)
    summary = {
        "simulator": args.simulator,
        "rows": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "average_reached_waypoint_rate": sum(row["reached_waypoints"] / row["total_waypoints"] for row in rows) / len(rows),
        "terminated_early_rate": sum(row["terminated_early"] for row in rows) / len(rows),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
