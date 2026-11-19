from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .prompts import build_messages


class Planner(Protocol):
    def generate(self, instruction: str, feedback: str | None = None, examples: list[dict] | None = None) -> str: ...


@dataclass
class TransformersPlanner:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Path | None = None
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("请先安装 transformers、torch 和 peft 依赖。") from error
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        self._device = torch.device("cuda:0" if use_cuda else "cpu")
        # ``device_map=auto`` may keep this small model on CPU on Windows.
        # Pin it explicitly so evaluation uses the available RTX GPU.
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(self.adapter_path))
        self._model.to(self._device)
        self._model.eval()

    def generate(self, instruction: str, feedback: str | None = None, examples: list[dict] | None = None) -> str:
        messages = build_messages(instruction, feedback, examples)
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([rendered], return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
        generated = output[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def generate_many(self, instructions: list[str], examples: list[dict] | None = None) -> list[str]:
        """Greedy batched generation with the same prompt and decoding contract."""
        previous_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        try:
            rendered = [
                self._tokenizer.apply_chat_template(build_messages(instruction, examples=examples), tokenize=False, add_generation_prompt=True)
                for instruction in instructions
            ]
            inputs = self._tokenizer(rendered, return_tensors="pt", padding=True).to(self._device)
            with self._torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
            generated = output[:, inputs.input_ids.shape[1]:]
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        finally:
            self._tokenizer.padding_side = previous_padding_side
