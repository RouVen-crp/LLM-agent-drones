"""Agent-level evaluation harness (P1 direct/agent + P3 memory/RAG).

Configurations are evaluated on the same fixed v4 test subset:

- ``direct``: one planner call -> validate -> semantic check
- ``agent``: the first plan seeded into the LangGraph runner (repair + PID)
- ``agent+memory``: LangGraph runner with BM25 episodic memory seeded from the
  v4 **train** split; retrieval examples are injected into the planner prompt

The first-attempt plans for ``direct`` and ``agent`` are generated in one batch
and reused, so the comparison isolates the agent loop rather than stochasticity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from drone_planner.evaluation import matches_reference_plan
from drone_planner.planner import TransformersPlanner
from drone_planner.simulation import PIDSimulator
from drone_planner.validation import parse_json_object, validate_text

from .graph import AgentRunner
from .memory import MissionMemory
from .tools import detect_rejection


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_object_valid(text: str) -> bool:
    try:
        parse_json_object(text)
        return True
    except ValueError:
        return False


def _semantic_correct(record: dict[str, Any], plan_text: str) -> bool:
    result = validate_text(plan_text)
    if not result.valid or result.plan is None:
        return False
    return matches_reference_plan(result.plan, record)


def _retrieval_info(result: Any, record: dict[str, Any]) -> dict[str, Any]:
    categories: list[str] = []
    for call in result.tool_calls:
        if call.get("tool") == "retrieve_similar":
            categories = list(call.get("result", {}).get("categories") or [])
    retrieved_ids = list(result.retrieved_ids)
    hit = (record["category"] in categories) if retrieved_ids else None
    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_categories": categories,
        "retrieval_hit": hit,
    }


def direct_row(record: dict[str, Any], plan_text: str, latency: float) -> dict[str, Any]:
    expected_rejection = bool(record.get("expect_rejection", False))
    result = validate_text(plan_text)
    rejected, _reason = detect_rejection(plan_text)
    semantic = None if expected_rejection or not result.valid else _semantic_correct(record, plan_text)
    contract_pass = bool(rejected) if expected_rejection else bool(result.valid)
    return {
        "id": record["id"],
        "category": record["category"],
        "expected_rejection": expected_rejection,
        "config": "direct",
        "status": "rejected" if rejected else ("valid" if result.valid else "invalid"),
        "json_object_valid": _json_object_valid(plan_text),
        "plan_valid": bool(result.valid),
        "semantic_correct": semantic,
        "explicit_rejection": bool(rejected),
        "valid_task_false_rejection": (not expected_rejection) and bool(rejected),
        "contract_pass": contract_pass,
        "simulation_success": None,
        "attempts": 1,
        "tool_calls_count": 0,
        "latency_seconds": round(latency, 4),
        "errors": list(result.errors),
        "plan_text": plan_text,
        "retrieved_ids": [],
        "retrieved_categories": [],
        "retrieval_hit": None,
        "trace": [],
    }


def agent_row(record: dict[str, Any], result: Any, generation_latency: float = 0.0) -> dict[str, Any]:
    expected_rejection = bool(record.get("expect_rejection", False))
    semantic = None
    if not expected_rejection and result.valid:
        semantic = _semantic_correct(record, result.plan_text)
    simulation = result.simulation or {}
    contract_pass = bool(result.rejected) if expected_rejection else bool(result.valid)
    retrieval = _retrieval_info(result, record)
    return {
        "id": record["id"],
        "category": record["category"],
        "expected_rejection": expected_rejection,
        "config": "agent",
        "status": result.status,
        "json_object_valid": _json_object_valid(result.plan_text),
        "plan_valid": bool(result.valid),
        "semantic_correct": semantic,
        "explicit_rejection": bool(result.rejected),
        "valid_task_false_rejection": (not expected_rejection) and bool(result.rejected),
        "contract_pass": contract_pass,
        "simulation_success": simulation.get("success") if result.simulation else None,
        "attempts": result.attempts,
        "tool_calls_count": len(result.tool_calls),
        "latency_seconds": round(generation_latency + result.latency_seconds, 4),
        "graph_latency_seconds": round(result.latency_seconds, 4),
        "errors": list(result.errors),
        "plan_text": result.plan_text,
        **retrieval,
        "trace": result.tool_calls,
    }


def summarize(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    normals = [row for row in rows if not row["expected_rejection"]]
    rejections = [row for row in rows if row["expected_rejection"]]
    simulated = [row for row in normals if row["simulation_success"] is not None]
    with_retrieval = [row for row in rows if row.get("retrieved_ids")]
    return {
        "config": config,
        "rows": len(rows),
        "normal_rows": len(normals),
        "rejection_rows": len(rejections),
        "json_object_rate": sum(row["json_object_valid"] for row in rows) / len(rows),
        "plan_valid_rate": sum(row["plan_valid"] for row in normals) / len(normals) if normals else 0.0,
        "semantic_accuracy": sum(bool(row["semantic_correct"]) for row in normals) / len(normals) if normals else 0.0,
        "rejection_correct_rate": sum(row["explicit_rejection"] for row in rejections) / len(rejections) if rejections else 0.0,
        "valid_task_false_rejection_rate": sum(row["valid_task_false_rejection"] for row in normals) / len(normals) if normals else 0.0,
        "contract_pass_rate": sum(row["contract_pass"] for row in rows) / len(rows),
        "simulation_success_rate": (sum(bool(row["simulation_success"]) for row in simulated) / len(simulated)) if simulated else None,
        "simulated_rows": len(simulated),
        "repaired_rows": sum(row["attempts"] > 1 for row in rows),
        "execution_failed_rows": sum(row["status"] == "execution_failed" for row in rows),
        "rejected_rows": sum(row["status"] == "rejected" for row in rows),
        "rows_with_retrieval": len(with_retrieval),
        "retrieval_hit_rate": (sum(bool(row["retrieval_hit"]) for row in with_retrieval) / len(with_retrieval)) if with_retrieval else None,
        "average_retrieved": sum(len(row.get("retrieved_ids") or []) for row in rows) / len(rows),
        "average_attempts": sum(row["attempts"] for row in rows) / len(rows),
        "average_tool_calls": sum(row["tool_calls_count"] for row in rows) / len(rows),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
    }


def _format_markdown(summaries: list[dict[str, Any]], agent_rows: list[dict[str, Any]], limit_note: str, memory_note: str) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    configs = [summary["config"] for summary in summaries]
    lines = [
        "# Agent 评测报告（P1 + P3）",
        "",
        f"数据集：`artifacts/dataset_contract_v4`，{limit_note}",
        "",
        f"记忆：{memory_note}",
        "",
        "| 指标 | " + " | ".join(configs) + " |",
        "| --- | " + " | ".join("---:" for _ in configs) + " |",
    ]
    metric_rows = [
        ("任务数", "rows"),
        ("正常任务", "normal_rows"),
        ("应拒绝任务", "rejection_rows"),
        ("JSON 对象率", "json_object_rate"),
        ("正常任务计划合法率", "plan_valid_rate"),
        ("正常任务语义准确率", "semantic_accuracy"),
        ("应拒绝任务正确拒绝率", "rejection_correct_rate"),
        ("正常任务误拒率", "valid_task_false_rejection_rate"),
        ("契约通过率", "contract_pass_rate"),
        ("正常任务仿真成功率（合法计划）", "simulation_success_rate"),
        ("参与仿真的正常任务数", "simulated_rows"),
        ("发生修复的任务数", "repaired_rows"),
        ("执行失败任务数", "execution_failed_rows"),
        ("有检索的任务数", "rows_with_retrieval"),
        ("检索类别命中率", "retrieval_hit_rate"),
        ("平均检索条数", "average_retrieved"),
        ("平均尝试次数", "average_attempts"),
        ("平均工具调用数", "average_tool_calls"),
        ("平均时延（秒）", "average_latency_seconds"),
    ]
    for label, key in metric_rows:
        lines.append(f"| {label} | " + " | ".join(fmt(summary[key]) for summary in summaries) + " |")
    lines.extend(
        [
            "",
            "> 说明：agent 的修复只处理 JSON、字段契约与执行空间边界失败；语义错误不会被反馈给 Planner。",
            "> direct 的时延只包含首次批量规划，agent 时延还包含校验、修复与 PID 仿真；agent+memory 的时延包含检索与单条生成。",
            "",
            "## 典型 agent 轨迹",
            "",
        ]
    )
    interesting: list[dict[str, Any]] = []
    seen: set[str] = set()
    for predicate in (
        lambda item: item["attempts"] > 1,
        lambda item: bool(item.get("retrieved_ids")),
        lambda item: item["status"] == "execution_failed",
        lambda item: item["status"] == "rejected",
    ):
        for item in agent_rows:
            if item["id"] not in seen and predicate(item):
                interesting.append(item)
                seen.add(item["id"])
            if len(interesting) >= 3:
                break
    if not interesting:
        interesting = agent_rows[:3]
    for row in interesting:
        lines.append(f"- `{row['id']}` status=`{row['status']}` attempts={row['attempts']} category={row['category']}")
        if row.get("retrieved_ids"):
            lines.append(f"  - retrieved: {row['retrieved_ids']} (hit={row.get('retrieval_hit')})")
        for call in row["trace"]:
            args = call.get("arguments") or {}
            compact_args = {key: (value[:80] + "..." if isinstance(value, str) and len(value) > 80 else value) for key, value in args.items()}
            lines.append(f"  - `{call['tool']}` ok={call['ok']} {json.dumps(compact_args, ensure_ascii=False)[:160]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 direct、LangGraph agent 与 agent+memory 配置。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset_contract_v4"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/lora_adapter_contract_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/agent_eval_v1"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=30, help="0 表示使用完整测试集。")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-sim", action="store_true", help="不运行 PID 仿真（只测规划与校验）。")
    parser.add_argument("--memory", action="store_true", help="增加 agent+memory 配置（BM25 检索训练集）。")
    parser.add_argument("--memory-k", type=int, default=2, help="每次检索注入的示例数。")
    parser.add_argument("--memory-db", type=Path, default=None, help="可选：把记忆持久化到 SQLite。")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset_dir / "test.jsonl")
    if args.limit:
        rejections = [row for row in rows if row.get("expect_rejection")]
        normals = [row for row in rows if not row.get("expect_rejection")]
        n_rej = min(len(rejections), max(1, args.limit // 3), args.limit)
        n_normal = max(0, args.limit - n_rej)
        rows = rejections[:n_rej] + normals[:n_normal]
    limit_note = f"共 {len(rows)} 条（{sum(1 for r in rows if r.get('expect_rejection'))} 条应拒绝）"

    planner = TransformersPlanner(model_name=args.model, adapter_path=args.adapter)
    instructions = [row["instruction"] for row in rows]
    started = perf_counter()
    first_plans = planner.generate_many(instructions, None)
    batch_seconds = perf_counter() - started
    generation_latency = batch_seconds / len(rows)

    simulator = None if args.no_sim else PIDSimulator()
    runner = AgentRunner(planner=planner, simulator=simulator, max_attempts=args.max_attempts)

    memory = None
    memory_runner = None
    memory_note = "未启用"
    if args.memory:
        memory = MissionMemory(db_path=args.memory_db)
        seeded = memory.seed_from_jsonl(args.dataset_dir / "train.jsonl", outcome="seed")
        memory_runner = AgentRunner(
            planner=planner,
            simulator=simulator,
            max_attempts=args.max_attempts,
            memory=memory,
            memory_k=args.memory_k,
        )
        memory_note = f"BM25Plus（字符 uni/bi-gram）+ 风险门控/弱类别定向，从 train 集注入 {seeded} 条，k={args.memory_k}"

    direct_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for record, plan_text in zip(rows, first_plans):
        direct_rows.append(direct_row(record, plan_text, generation_latency))
        agent_rows.append(agent_row(record, runner.run(record["instruction"], seed_plan_text=plan_text), generation_latency))
        if memory_runner is not None:
            memory_rows.append(agent_row(record, memory_runner.run(record["instruction"]), 0.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [("direct", direct_rows), ("agent", agent_rows)]
    if memory_rows:
        outputs.append(("agent_memory", memory_rows))
    for name, data in outputs:
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = [summarize(rows, name) for name, rows in outputs if name != "agent_memory"]
    if memory_rows:
        summaries.append(summarize(memory_rows, "agent+memory"))
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(
        _format_markdown(summaries, memory_rows or agent_rows, limit_note, memory_note), encoding="utf-8"
    )
    if memory is not None:
        memory.close()
    print(
        json.dumps(
            {
                "summaries": summaries,
                "first_plan_seconds_per_row": round(generation_latency, 3),
                "memory": memory_note,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
