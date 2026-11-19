"""Fine-tune a Qwen2.5-0.5B LoRA router for P2 function calling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .prompts import build_router_messages


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 function-calling router 的 LoRA adapter。")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/router_data_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lora_router_contract_v1"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-limit", type=int, default=200)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except ImportError as error:
        raise SystemExit("缺少训练依赖。") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict) -> dict:
        messages = build_router_messages(row["instruction"]) + [{"role": "assistant", "content": row["target"]}]
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(full, truncation=True, max_length=768, add_special_tokens=False)
        prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        encoded["labels"] = [-100] * min(prefix_length, len(encoded["input_ids"])) + encoded["input_ids"][prefix_length:]
        return encoded

    def load(path: Path, limit: int = 0) -> Dataset:
        rows = read_jsonl(path)
        if limit and len(rows) > limit:
            random.Random(args.seed).shuffle(rows)
            rows = rows[:limit]
        dataset = Dataset.from_list(rows)
        return dataset.map(lambda row: tokenize(row), remove_columns=dataset.column_names)

    train_ds = load(args.data_dir / "train.jsonl")
    val_ds = load(args.data_dir / "validation.jsonl", limit=args.val_limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    run_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=run_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    (args.output_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
