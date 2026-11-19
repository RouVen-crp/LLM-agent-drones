"""Tool-calling router used by the agent when function calling is enabled."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from router.prompts import apply_template, parse_tool_call


@dataclass
class ToolRouter:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Path | None = Path("artifacts/lora_router_contract_v1")
    max_new_tokens: int = 96
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("router 依赖未安装（transformers/peft/torch）。") from error
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        if self.adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._model.eval()

    def decide(self, instruction: str) -> dict:
        if self._model is None:
            raise RuntimeError("router 未加载。")
        rendered = apply_template(self._tokenizer, instruction)
        inputs = self._tokenizer([rendered], return_tensors="pt").to(self._device)
        started = perf_counter()
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_tool_call(text)
        if parsed is None:
            return {"name": None, "arguments": {}, "valid": False, "raw": text[:300], "latency_seconds": perf_counter() - started}
        return {
            "name": parsed["name"],
            "arguments": parsed.get("arguments", {}),
            "valid": True,
            "raw": text[:300],
            "latency_seconds": perf_counter() - started,
        }
