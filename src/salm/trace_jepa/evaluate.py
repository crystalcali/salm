"""Held-out generation check on explicitly known answers.

This is intentionally conservative. Exact-match on numerical/simple text
outputs is not a substitute for MATH-500/math_verify or code execution.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")


def extract_answer(text: str) -> str:
    """Take the LAST unambiguous structured final answer, not an arbitrary digit."""
    box = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if box:
        return box[-1].strip()
    labels = re.findall(r"(?:final answer|answer)\s*:\s*([^\n]+)",
                        text, flags=re.IGNORECASE)
    return labels[-1].strip().rstrip(".") if labels else ""


def normalized_equal(predicted: str, expected: str) -> bool:
    def clean(s):
        s = s.strip().replace("$", "")
        if s.startswith(r"\boxed{") and s.endswith("}"):
            s = s[7:-1]
        return re.sub(r"\s+", "", s).strip().rstrip(".")
    a, b = clean(predicted), clean(expected)
    if not a or not b:
        return False
    if NUMERIC.fullmatch(a) and NUMERIC.fullmatch(b):
        try:
            return Decimal(a) == Decimal(b)
        except InvalidOperation:
            return False
    return a.casefold() == b.casefold()


def evaluate_model(run: str, data_file: str, *, limit: int = 50,
                   max_new_tokens: int = 256, out: str | None = None) -> dict:
    import torch
    from peft import PeftModel
    from .data import read_traces
    from .engine import make_model

    run_dir = Path(run)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model, tokenizer = make_model(
        cfg["model"], cfg.get("revision"), use_lora=False,
        lora_rank=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
        gradient_checkpointing=False,
    )
    adapter_dir = run_dir / "adapter_or_model"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError("Evaluation currently requires a PEFT LoRA adapter")
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    rows = read_traces(data_file)
    results = []
    for row in rows[:limit]:
        expected = str(row.get("final_answer") or row.get("answer") or "").strip()
        if not expected:
            continue
        prompt = "Task:\n" + row["prompt"].strip() + "\nSolution:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=cfg.get("max_length", 384))
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        predicted = extract_answer(response)
        results.append({
            "task_id": row["task_id"], "expected": expected, "predicted": predicted,
            "correct": normalized_equal(predicted, expected),
            "missing_structured_answer": not bool(predicted),
        })
    report = {
        "experiment": "trace_generation_eval",
        "run": str(run_dir), "data": data_file,
        "scored_tasks": len(results),
        "exact_match": sum(r["correct"] for r in results) / max(len(results),1),
        "missing_answer_rate": sum(r["missing_structured_answer"] for r in results) / max(len(results),1),
        "notes": "Conservative formatted-answer exact match. NOT an official reasoning benchmark.",
        "results": results,
    }
    outfile = Path(out) if out else run_dir / "generation_eval.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
