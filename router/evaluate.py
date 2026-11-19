"""Evaluate the P2 function-calling router on the v5 test split.

Comparisons
- ``router``: Qwen2.5-0.5B + LoRA trained to emit tool calls
- ``base``: the same model without adapter, zero-shot with the same prompt
- ``workflow``: the v4 planner used directly (plan or reject), i.e. no router
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.tools import detect_rejection
from drone_planner.evaluation import matches_reference_plan
from drone_planner.planner import TransformersPlanner
from drone_planner.validation import parse_json_object, validate_text

from .prompts import apply_template, parse_tool_call

TOOLS = ("plan_task", "reject_task", "ask_clarification")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(rows: list[dict[str, Any]], limit: int, seed: int = 42) -> list[dict[str, Any]]:
    rejections = [row for row in rows if row.get("expect_rejection")]
    normals = [row for row in rows if not row.get("expect_rejection")]
    random.Random(seed).shuffle(normals)
    if not limit:
        return rejections + normals
    n_normal = max(0, limit - len(rejections))
    return rejections + normals[:n_normal]


def expected_tool(row: dict[str, Any]) -> str:
    if not row.get("expect_rejection"):
        return "plan_task"
    return "ask_clarification" if row.get("action") == "clarify" else "reject_task"


def load_model(model_name: str, adapter: Path | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def generate_calls(tokenizer, model, instructions: list[str], batch_size: int = 8, max_new_tokens: int = 96) -> list[str]:
    outputs: list[str] = []
    device = next(model.parameters()).device
    for offset in range(0, len(instructions), batch_size):
        batch = instructions[offset:offset + batch_size]
        prompts = [apply_template(tokenizer, instruction) for instruction in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.extend(tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))
    return outputs


def measure(rows: list[dict[str, Any]], calls: list[str], name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    for row, text in zip(rows, calls):
        parsed = parse_tool_call(text)
        predicted = (parsed or {}).get("name")
        predictions.append(
            {
                "id": row["id"],
                "instruction": row["instruction"],
                "expected_tool": expected_tool(row),
                "predicted_tool": predicted,
                "valid_json": parsed is not None,
                "arguments": (parsed or {}).get("arguments"),
                "raw": text[:300],
            }
        )
    predicted_counts = {tool: 0 for tool in TOOLS}
    for item in predictions:
        if item["predicted_tool"] in predicted_counts:
            predicted_counts[item["predicted_tool"]] += 1

    def rate(rows: list[dict[str, Any]], predicate) -> float:
        return sum(1 for item in rows if predicate(item)) / len(rows) if rows else 0.0

    plan_rows = [item for item in predictions if item["expected_tool"] == "plan_task"]
    reject_rows = [item for item in predictions if item["expected_tool"] == "reject_task"]
    clarify_rows = [item for item in predictions if item["expected_tool"] == "ask_clarification"]
    non_plan_rows = reject_rows + clarify_rows

    per_tool = {}
    for tool in TOOLS:
        tp = sum(1 for item in predictions if item["expected_tool"] == tool and item["predicted_tool"] == tool)
        fp = sum(1 for item in predictions if item["expected_tool"] != tool and item["predicted_tool"] == tool)
        fn = sum(1 for item in predictions if item["expected_tool"] == tool and item["predicted_tool"] != tool)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_tool[tool] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    summary = {
        "config": name,
        "rows": len(predictions),
        "json_valid_rate": rate(predictions, lambda item: item["valid_json"]),
        "tool_accuracy": rate(predictions, lambda item: item["predicted_tool"] == item["expected_tool"]),
        "reject_recall": rate(reject_rows, lambda item: item["predicted_tool"] == "reject_task"),
        "clarify_recall": rate(clarify_rows, lambda item: item["predicted_tool"] == "ask_clarification"),
        "unsafe_plan_rate": rate(non_plan_rows, lambda item: item["predicted_tool"] == "plan_task"),
        "normal_plan_rate": rate(plan_rows, lambda item: item["predicted_tool"] == "plan_task"),
        "predicted_counts": predicted_counts,
        "per_tool": per_tool,
    }
    return predictions, summary


def generate_plan_texts(rows: list[dict[str, Any]], selected_ids: list[str], planner: TransformersPlanner) -> dict[str, str]:
    """Generate plans only for the selected rows (by id), preserving labels."""
    ids = set(selected_ids)
    subset = [row for row in rows if row["id"] in ids]
    texts = planner.generate_many([row["instruction"] for row in subset], None) if subset else []
    return {row["id"]: text for row, text in zip(subset, texts)}


def planner_quality(rows: list[dict[str, Any]], selected_ids: list[str], plan_texts: dict[str, str]) -> dict[str, Any]:
    """Conditional plan quality on normal tasks that reached the Planner."""
    ids = set(selected_ids)
    normal = [row for row in rows if row["id"] in ids and not row.get("expect_rejection")]
    valid = semantic = rejected = 0
    for row in normal:
        text = plan_texts.get(row["id"], "")
        result = validate_text(text)
        if result.valid:
            valid += 1
            semantic += int(matches_reference_plan(result.plan, row))
        if detect_rejection(text)[0]:
            rejected += 1
    return {
        "rows": len(normal),
        "plan_valid_rate": valid / len(normal) if normal else None,
        "semantic_accuracy": semantic / len(normal) if normal else None,
        "planner_rejection_rate": rejected / len(normal) if normal else None,
    }


def router_agent_contract(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], plan_texts: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for row, prediction in zip(rows, predictions):
        tool = prediction["predicted_tool"]
        if row.get("expect_rejection"):
            contract = tool == prediction["expected_tool"]
        elif tool == "plan_task":
            contract = bool(validate_text(plan_texts.get(row["id"], "")).valid)
        else:
            contract = False
        records.append(
            {
                "id": row["id"],
                "expected_tool": prediction["expected_tool"],
                "predicted_tool": tool,
                "contract_pass": contract,
            }
        )
    return {
        "config": "router_agent",
        "rows": len(records),
        "contract_pass_rate": sum(item["contract_pass"] for item in records) / len(records),
    }, records


def workflow_contract(rows: list[dict[str, Any]], plan_texts: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for row in rows:
        text = plan_texts.get(row["id"], "")
        if row.get("expect_rejection"):
            contract = detect_rejection(text)[0]
        else:
            contract = bool(validate_text(text).valid)
        records.append({"id": row["id"], "contract_pass": contract})
    return {
        "config": "workflow_planner",
        "rows": len(records),
        "contract_pass_rate": sum(item["contract_pass"] for item in records) / len(records),
    }, records


def _report(summaries: dict[str, dict[str, Any]], e2e: list[dict[str, Any]], conditional: dict[str, Any], predictions: list[dict[str, Any]], data_note: str) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def table(rows: list[tuple[str, str]]) -> list[str]:
        lines = ["| 指标 | " + " | ".join(summary["config"] for _, summary in [(name, summaries[name]) for name in ("router", "base")]) + " |", "| --- | " + " | ".join("---:" for _ in range(2)) + " |"]
        for label, key in rows:
            lines.append(f"| {label} | " + " | ".join(fmt(summaries[name][key]) for name in ("router", "base")) + " |")
        return lines

    lines = [
        "# P2 函数调用 Router 评测",
        "",
        data_note,
        "",
        "## 工具选择（router vs base zero-shot）",
        "",
        *table(
            [
                ("任务数", "rows"),
                ("JSON 合法率", "json_valid_rate"),
                ("工具选择准确率", "tool_accuracy"),
                ("reject_task 召回", "reject_recall"),
                ("ask_clarification 召回", "clarify_recall"),
                ("危险任务被误判为可规划", "unsafe_plan_rate"),
                ("正常任务被正确规划", "normal_plan_rate"),
            ]
        ),
        "",
        "## 端到端契约通过率",
        "",
    ]
    if e2e:
        lines.extend(
            [
                "| 配置 | 契约通过率 |",
                "| --- | ---: |",
                "| router agent（router 选择工具后执行） | " + fmt(e2e[0]["contract_pass_rate"]) + " |",
                "| workflow planner（同一 Planner 直接规划/拒绝） | " + fmt(e2e[1]["contract_pass_rate"]) + " |",
                "",
            ]
        )
    else:
        lines.extend(["未运行（--skip-e2e）", ""])
    lines.extend(["## 条件 Planner 指标（Planner 实际处理的任务）", ""])
    if conditional:
        lines.extend(
            [
                "| 指标 | router 路由到 plan 的正常任务 | workflow 全部正常任务 |",
                "| --- | ---: | ---: |",
                "| 任务数 | " + fmt(conditional["router"]["rows"]) + " | " + fmt(conditional["workflow"]["rows"]) + " |",
                "| 计划合法率 | " + fmt(conditional["router"]["plan_valid_rate"]) + " | " + fmt(conditional["workflow"]["plan_valid_rate"]) + " |",
                "| 结构化计划语义准确率 | " + fmt(conditional["router"]["semantic_accuracy"]) + " | " + fmt(conditional["workflow"]["semantic_accuracy"]) + " |",
                "| Planner 自身拒绝率 | " + fmt(conditional["router"]["planner_rejection_rate"]) + " | " + fmt(conditional["workflow"]["planner_rejection_rate"]) + " |",
                "",
            ]
        )
    else:
        lines.extend(["未运行（--skip-e2e）", ""])
    lines.extend(
        [
            "## 分工具指标（router）",
            "",
            "| 工具 | support | precision | recall | f1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for tool, metrics in summaries["router"]["per_tool"].items():
        lines.append(f"| {tool} | {metrics['support']} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    lines.extend(["", "## 错误样例", ""])
    errors = [item for item in predictions if item["predicted_tool"] != item["expected_tool"]][:5]
    for item in errors:
        lines.append(f"- `{item['id']}` expected=`{item['expected_tool']}` predicted=`{item['predicted_tool']}`：{item['instruction'][:60]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 P2 router 的工具选择与端到端契约。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset_contract_v5"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/lora_router_contract_v1"))
    parser.add_argument("--planner-adapter", type=Path, default=Path("artifacts/lora_adapter_contract_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/router_eval_v1"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=300, help="0 表示完整 test。")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    test_rows = select_rows(read_jsonl(args.dataset_dir / "test.jsonl"), args.limit)
    instructions = [row["instruction"] for row in test_rows]
    data_note = f"v5 test 子集：{len(test_rows)} 条（{sum(1 for r in test_rows if r.get('expect_rejection'))} 条应拒绝）"

    tokenizer, router_model = load_model(args.model, args.adapter)
    started = perf_counter()
    router_calls = generate_calls(tokenizer, router_model, instructions, args.batch_size)
    router_seconds = perf_counter() - started
    router_predictions, router_summary = measure(test_rows, router_calls, "router")
    router_summary["seconds_per_row"] = round(router_seconds / len(test_rows), 3)
    print(f"[router] done rows={len(test_rows)} seconds={router_seconds:.1f}", file=sys.stderr, flush=True)
    del router_model
    torch.cuda.empty_cache()

    base_tokenizer, base_model = load_model(args.model, None)
    base_calls = generate_calls(base_tokenizer, base_model, instructions, args.batch_size)
    base_predictions, base_summary = measure(test_rows, base_calls, "base")
    print(f"[base] done rows={len(test_rows)}", file=sys.stderr, flush=True)
    del base_model
    torch.cuda.empty_cache()

    summaries = {"router": router_summary, "base": base_summary}
    e2e: list[dict[str, Any]] = []
    conditional: dict[str, Any] = {}
    if not args.skip_e2e:
        planner = TransformersPlanner(adapter_path=args.planner_adapter)
        router_plan_ids = [row["id"] for row, prediction in zip(test_rows, router_predictions) if prediction["predicted_tool"] == "plan_task"]
        workflow_ids = [row["id"] for row in test_rows]
        router_plan_texts = generate_plan_texts(test_rows, router_plan_ids, planner)
        workflow_plan_texts = generate_plan_texts(test_rows, workflow_ids, planner)
        router_e2e, _ = router_agent_contract(test_rows, router_predictions, router_plan_texts)
        workflow_e2e, _ = workflow_contract(test_rows, workflow_plan_texts)
        e2e = [router_e2e, workflow_e2e]
        conditional = {
            "router": planner_quality(test_rows, router_plan_ids, router_plan_texts),
            "workflow": planner_quality(test_rows, workflow_ids, workflow_plan_texts),
        }
        del planner
        torch.cuda.empty_cache()
        print(f"[e2e] router_plan_rows={len(router_plan_ids)} workflow_rows={len(workflow_ids)}", file=sys.stderr, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "router_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in router_predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "base_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in base_predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries, "end_to_end": e2e, "conditional_planner": conditional}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_report(summaries, e2e, conditional, router_predictions, data_note), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "end_to_end": e2e, "conditional_planner": conditional, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
