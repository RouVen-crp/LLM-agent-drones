from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def paraphrase(instruction: str, plan: dict, model: str) -> str:
    """Use an optional API only to diversify wording; the JSON label remains untouched."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("受控改写需要 openai 依赖。") from error
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL") or None)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "将中文无人机指令改写成自然、简洁的同义表达。绝不改变坐标、数量、高度、返航或约束；只输出改写后的单句指令。"},
            {"role": "user", "content": f"原指令：{instruction}\n对应计划：{json.dumps(plan, ensure_ascii=False)}"},
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="对合成任务进行可审计的受控语言改写。")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--model", default=os.environ.get("PARAPHRASE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=0, help="0 表示改写全部记录。")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("请设置 OPENAI_API_KEY 后再运行受控改写。")
    remaining = args.limit or float("inf")
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = args.dataset_dir / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row.get("expect_rejection") or remaining <= 0:
                continue
            row["template_instruction"] = row["instruction"]
            row["instruction"] = paraphrase(row["instruction"], row["plan"], args.model)
            row["paraphrase_model"] = args.model
            remaining -= 1
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
