"""Small *verifiable* synthetic smoke corpus and optional real OpenR1 ingestion."""
from __future__ import annotations

import hashlib
import random

from .data import split_tasks, write_jsonl


def prepare_arithmetic(out_train: str, out_val: str, tasks: int = 128, seed: int = 42):
    """Procedural arithmetic traces; smoke ONLY, not evidence of reasoning improvement."""
    if tasks < 8:
        raise ValueError("Use at least 8 tasks")
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < tasks:
        a, b = rng.randint(2, 500), rng.randint(2, 500)
        op = ["add", "subtract", "multiply"][len(rows) % 3]
        key = (op, a, b)
        if key in seen:
            continue
        seen.add(key)
        if op == "add":
            v, expr = a+b, f"{a} + {b}"
            steps = [f"Compute the sum of {a} and {b}.", f"Adding gives {v}."]
        elif op == "subtract":
            v, expr = a-b, f"{a} - {b}"
            steps = [f"Subtract {b} from {a}.", f"The difference is {v}."]
        else:
            v, expr = a*b, f"{a} * {b}"
            steps = [f"Multiply {a} by {b}.", f"The product is {v}."]
        rows.append({
            "task_id": f"synthetic-{op}-{a}-{b}",
            "source": "procedural_arithmetic_smoke",
            "prompt": f"Calculate {expr}. Show steps.",
            "steps": steps, "final_answer": str(v),
            "verification": {"arithmetic": True},
        })
    train, val = split_tasks(rows, seed=seed)
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    return {"source": "synthetic_smoke_only", "train": len(train), "val": len(val)}


def prepare_openr1(out_train: str, out_val: str, *,
                   max_train: int = 1000, max_val: int = 100,
                   seed: int = 42):
    """Streaming from publicly documented fields, math_verify verified only.

    NOTE: checks verify the FINAL answer, not correctness of every reasoning
    step. Split is task-level by stable UUID hash before generation selection.
    Dataset is Apache-2.0; no large archive is downloaded as a single file.
    """
    from datasets import load_dataset
    import re
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=1024)
    train, val = [], []
    seen = set()
    for obj in ds:
        prompt = obj.get("problem")
        uid = str(obj.get("uuid") or "")
        generations = obj.get("generations") or []
        flags = obj.get("correctness_math_verify") or []
        if not isinstance(prompt, str) or not uid or uid in seen:
            continue
        seen.add(uid)
        chosen = next((text for text, flag in zip(generations, flags)
                       if flag is True and isinstance(text, str) and len(text) > 80), None)
        if not chosen:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", chosen)
                      if part.strip()]
        if len(paragraphs) < 2:
            paragraphs = [part.strip() for part in chosen.splitlines() if part.strip()]
        if len(paragraphs) < 2:
            continue
        # Do not silently mark each step verified.
        row = {
            "task_id": "openr1-" + uid, "prompt": prompt,
            "steps": paragraphs,
            "source": "open-r1/OpenR1-Math-220k/default",
            "verification": {"final_answer_math_verify": True,
                             "steps_verified": False},
            "license": "apache-2.0",
        }
        digest = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
        is_val = digest % 10 == 0
        if is_val and len(val) < max_val:
            val.append(row)
        elif not is_val and len(train) < max_train:
            train.append(row)
        if len(train) >= max_train and len(val) >= max_val:
            break
    if not train or not val:
        raise RuntimeError("Insufficient verified traces in streamed selection")
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    return {"source": "OpenR1 default math_verify", "train": len(train), "val": len(val)}
