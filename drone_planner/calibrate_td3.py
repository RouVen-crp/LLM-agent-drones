from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import MissionPlan
from .simulation import TD3Simulator


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out reference plans directly through TD3 to establish its tracking ceiling.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = [row for row in _read(args.dataset_dir / "test.jsonl") if not row.get("expect_rejection", False)]
    done: set[str] = set()
    if args.resume and args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    simulator = TD3Simulator()
    rows: list[dict] = []
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for record in records:
            if record["id"] in done:
                continue
            plan = MissionPlan.from_dict(record["plan"])
            result = simulator.run(next(item for item in plan.drone_plans if item.drone_id == plan.selected_drone_id))
            row = {"id": record["id"], "reference_tracking_success": result.success, **result.to_dict()}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    all_rows = _read(args.output)
    summary = {
        "records": len(all_rows),
        "reference_tracking_success_rate": sum(row["reference_tracking_success"] for row in all_rows) / len(all_rows) if all_rows else None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
