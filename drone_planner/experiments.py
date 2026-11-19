from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .evaluation import evaluate_records, evaluate_records_batched
from .planner import TransformersPlanner
from .simulation import PIDSimulator, TD3Simulator


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(rows: list[dict]) -> dict:
    valid_cases = [row for row in rows if not row["expected_rejection"]]
    rejection_cases = [row for row in rows if row["expected_rejection"]]
    semantically_correct = [row for row in valid_cases if row.get("semantic_correct")]
    simulated = [row for row in semantically_correct if row["simulation_success"] is not None]
    return {
        "rows": len(rows),
        "json_object_rate": sum(row.get("json_object_valid", row["parse_valid"]) for row in rows) / len(rows),
        "plan_constraint_valid_rate": sum(row["constraint_valid"] for row in valid_cases) / len(valid_cases),
        "semantic_plan_accuracy": sum(bool(row.get("semantic_correct", False)) for row in valid_cases) / len(valid_cases),
        "safe_rejection_correct_rate": (sum(row["safe_rejection"] for row in rejection_cases) / len(rejection_cases)) if rejection_cases else None,
        "valid_task_false_rejection_rate": (sum(row.get("valid_task_false_rejection", False) for row in valid_cases) / len(valid_cases)) if valid_cases else None,
        "end_to_end_contract_pass_rate": (sum(row["constraint_valid"] for row in valid_cases) + sum(row["safe_rejection"] for row in rejection_cases)) / len(rows),
        "average_attempts": sum(row["attempts"] for row in rows) / len(rows),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "simulation_success_rate": (sum(row["simulation_success"] for row in simulated) / len(simulated)) if simulated else None,
    }


def _few_shot_examples(rows: list[dict]) -> list[dict]:
    """Use valid, correctly paired demonstrations from distinct task categories."""
    examples, seen = [], set()
    for row in rows:
        if row.get("expect_rejection") or row["category"] in seen:
            continue
        examples.append(row)
        seen.add(row["category"])
        if len(examples) == 5:
            break
    if len(examples) < 4:
        raise ValueError("Few-shot 演示不足：训练集至少需要 4 类有效任务。")
    return examples


def _write_breakdowns(rows: list[dict], output_dir: Path, variant: str) -> None:
    task_counts: dict[str, dict[str, int]] = {}
    failure_counts: dict[str, int] = {}
    for row in rows:
        if row["expected_rejection"]:
            continue
        bucket = task_counts.setdefault(row["category"], {"rows": 0, "semantic_correct": 0, "false_rejections": 0})
        bucket["rows"] += 1
        bucket["semantic_correct"] += int(bool(row.get("semantic_correct")))
        bucket["false_rejections"] += int(bool(row.get("valid_task_false_rejection")))
        if not row.get("semantic_correct"):
            labels = row.get("semantic_mismatches") or (["explicit_rejection"] if row.get("valid_task_false_rejection") else ["invalid_or_constraint_error"])
            for label in labels:
                failure_counts[label] = failure_counts.get(label, 0) + 1
    task_rows = [{"variant": variant, "category": category, **counts, "semantic_accuracy": counts["semantic_correct"] / counts["rows"]} for category, counts in sorted(task_counts.items())]
    failure_rows = [{"variant": variant, "failure_type": label, "count": count} for label, count in sorted(failure_counts.items())]
    for name, data in (("task_wise", task_rows), ("failure_breakdown", failure_rows)):
        (output_dir / f"{variant}_{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / f"{variant}_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]) if data else (["variant", "category", "rows", "semantic_correct", "false_rejections", "semantic_accuracy"] if name == "task_wise" else ["variant", "failure_type", "count"]))
            writer.writeheader()
            writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行四组 Planner 对照实验并写入可复现结果。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--simulator", choices=["td3", "pid"], default="pid", help="启用 --simulate 时使用的单机执行器。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="测试集起始下标（含）。")
    parser.add_argument("--end", type=int, help="测试集结束下标（不含）。")
    parser.add_argument("--variants", nargs="+", choices=["zero_shot", "few_shot", "lora_sft", "lora_feedback"], default=["zero_shot", "few_shot", "lora_sft", "lora_feedback"])
    parser.add_argument("--batch-size", type=int, default=1, help="无反馈、无仿真评测的批量推理大小。")
    args = parser.parse_args()
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    test_rows, train_rows = _read(args.dataset_dir / "test.jsonl"), _read(args.dataset_dir / "train.jsonl")
    test_rows = test_rows[args.start:args.end]
    examples = _few_shot_examples(train_rows)
    selected = set(args.variants)
    # Load only the model(s) needed for this invocation.  This matters when a
    # long-running evaluation is resumed in small, crash-safe chunks.
    base = TransformersPlanner() if selected & {"zero_shot", "few_shot"} else None
    lora = TransformersPlanner(adapter_path=args.adapter) if selected & {"lora_sft", "lora_feedback"} else None
    simulator = (PIDSimulator() if args.simulator == "pid" else TD3Simulator()) if args.simulate else None
    variants = [("zero_shot", base), ("few_shot", base), ("lora_sft", lora), ("lora_feedback", lora)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_config.json").write_text(json.dumps({
        "dataset_dir": str(args.dataset_dir), "adapter": str(args.adapter), "seed": args.seed,
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "max_new_tokens": 512,
        "do_sample": False, "variants": args.variants, "simulate": args.simulate,
        "simulator": args.simulator if args.simulate else None, "batch_size": args.batch_size,
        "start": args.start, "end": args.end, "few_shot_example_ids": [row["id"] for row in examples],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for name, planner in variants:
        if name not in args.variants:
            continue
        output_path = args.output_dir / f"{name}.jsonl"
        if args.batch_size > 1 and simulator is None and name != "lora_feedback":
            active_examples = examples if name == "few_shot" else None
            evaluate_records_batched(test_rows, planner, name, output_path, active_examples, args.resume, args.batch_size)
        else:
            evaluate_records(test_rows, planner, name, output_path, examples, simulator, args.resume)
        rows = _read(output_path)
        summary = {"variant": name, **_summary(rows)}
        summaries.append(summary)
        _write_breakdowns(rows, args.output_dir, name)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
