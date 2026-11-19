"""Tool-call data for the P2 function-calling router.

The v5 dataset already separates normal tasks from rejection tasks and marks
each rejection with ``action`` = reject | clarify.  That is enough to supervise
a router whose only job is to choose the next tool:

- normal task -> ``plan_task(instruction)``
- unsafe / contradictory / unsupported -> ``reject_task(reason)``
- missing information -> ``ask_clarification(reason)``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .prompts import build_router_messages, target_tool_call

REJECT_REASON = "任务含糊、矛盾或超出单无人机执行空间"
CLARIFY_REASON = "缺少必要信息，请补充目标位置、高度或任务约束"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_to_call(row: dict[str, Any]) -> dict[str, Any]:
    """Map one v5 record to the expected router tool call."""
    if row.get("expect_rejection"):
        if row.get("action") == "clarify":
            return {"name": "ask_clarification", "arguments": {"reason": CLARIFY_REASON}}
        return {"name": "reject_task", "arguments": {"reason": REJECT_REASON}}
    return {"name": "plan_task", "arguments": {"instruction": row["instruction"]}}


def row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    call = record_to_call(row)
    return {
        "instruction": row["instruction"],
        "expected_action": row.get("action") or "plan",
        "tool": call["name"],
        "arguments": call["arguments"],
        "target": json.dumps(call, ensure_ascii=False, separators=(",", ":")),
    }


def write_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_dir / f"{split}.jsonl")
    return [row_to_sample(row) for row in rows]


def build_balanced_train(samples: list[dict[str, Any]], seed: int = 42, normal_limit: int = 2500, reject_multiplier: int = 2, clarify_multiplier: int = 6) -> list[dict[str, Any]]:
    """Oversample reject/clarify because the raw train split is 95% normal."""
    rng = random.Random(seed)
    plan = [row for row in samples if row["tool"] == "plan_task"]
    reject = [row for row in samples if row["tool"] == "reject_task"]
    clarify = [row for row in samples if row["tool"] == "ask_clarification"]
    rng.shuffle(plan)
    plan = plan[:normal_limit]
    balanced = plan + reject * reject_multiplier + clarify * clarify_multiplier
    rng.shuffle(balanced)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 router function-calling 数据。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset_contract_v5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/router_data_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = write_split(args.dataset_dir, "train")
    validation = write_split(args.dataset_dir, "validation")
    test = write_split(args.dataset_dir, "test")
    balanced = build_balanced_train(train, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", balanced), ("validation", validation), ("test", test)):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            result[row["tool"]] = result.get(row["tool"], 0) + 1
        return result

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "splits": {"train": len(balanced), "validation": len(validation), "test": len(test)},
                "train_tools": counts(balanced),
                "validation_tools": counts(validation),
                "test_tools": counts(test),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
