"""Trace JSONL ingestion, reproducible task-level splits and causal views."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_traces(path: str) -> list[dict]:
    """Input format: task_id, prompt, steps: list[str], optional final_answer.

    The required steps must be separate sequential chunks; a final answer may
    become the last chunk. Generic completion text is split by nonempty lines
    only if no 'steps' field is supplied. Unverified traces are allowed for
    research but their verification status must be explicitly recorded.
    """
    traces = []
    ids: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        task_id = str(obj.get("task_id") or "").strip()
        prompt = obj.get("prompt") or obj.get("question")
        if not task_id or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{number}: task_id and prompt required")
        if task_id in ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        steps = obj.get("steps")
        if steps is None:
            steps = (obj.get("completion") or "").splitlines()
        if not isinstance(steps, list) or any(not isinstance(s, str) for s in steps):
            raise ValueError(f"{path}:{number}: steps must be a list of strings")
        steps = [s.strip() for s in steps if s.strip()]
        answer = obj.get("final_answer")
        if isinstance(answer, str) and answer.strip():
            steps.append("Final answer: " + answer.strip())
        if not steps:
            raise ValueError(f"{path}:{number}: no steps or completion")
        obj["task_id"], obj["prompt"], obj["steps"] = task_id, prompt, steps
        traces.append(obj)
        ids.add(task_id)
    if not traces:
        raise ValueError("Empty trace dataset: " + path)
    return traces


def verify_disjoint(train: list[dict], validation: list[dict]):
    # It is essential to split by ORIGINAL problem, never by segment.
    overlap = {x["task_id"] for x in train} & {x["task_id"] for x in validation}
    if overlap:
        raise ValueError(f"Train/validation task_id leakage: {sorted(overlap)[:5]}")
    fingerprints_train = {
        hashlib.sha256(x["prompt"].strip().encode()).hexdigest() for x in train
    }
    fingerprints_val = {
        hashlib.sha256(x["prompt"].strip().encode()).hexdigest() for x in validation
    }
    if fingerprints_train & fingerprints_val:
        raise ValueError("Train/validation duplicate prompts")


def split_tasks(rows: list[dict], validation_fraction: float = 0.2, seed: int = 42):
    if len(rows) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need >=2 tasks and 0 < validation_fraction < 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    nval = max(1, int(len(shuffled) * validation_fraction))
    train, validation = shuffled[nval:], shuffled[:nval]
    verify_disjoint(train, validation)
    return train, validation


def render_trace(row: dict, cut: int) -> tuple[str, str, str]:
    """Return (task prompt, visible prefix, target suffix) without target leakage."""
    steps = row["steps"]
    if not 0 <= cut < len(steps):
        raise ValueError("cut must leave at least one hidden segment")
    prefix = "Task:\n" + row["prompt"].strip() + "\nSolution:\n"
    seen = "".join(f"Step {i+1}: {s}\n" for i, s in enumerate(steps[:cut]))
    future = "".join(f"Step {i+1}: {s}\n" for i, s in enumerate(steps[cut:], cut))
    return prefix, prefix + seen, future


def build_pair(row: dict, tokenizer, rng: random.Random, max_length: int = 512) -> dict:
    if max_length < 32:
        raise ValueError("max_length must be >=32")
    cut = rng.randrange(len(row["steps"]))  # random, reproducible per seed
    task, visible, target = render_trace(row, cut)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    instruction = encode(task)
    visible_ids = encode(visible)
    target_ids = encode(target)
    if not target_ids:
        raise ValueError("Empty target after tokenization")
    # Separate tokenization makes the two views exact concatenable segments.
    # Only the prefix is accessed from the causal full-trace pass.
    full = visible_ids + target_ids
    if len(instruction) >= max_length - 2:
        raise ValueError(f"Task prompt too long: {row['task_id']}")
    if len(visible_ids) >= max_length - 2:
        # Keep enough of the visible prefix for the suffix JEPA target.
        visible_ids = visible_ids[:max_length - 2]
    remaining = max_length - len(visible_ids)
    target_ids = target_ids[:remaining]
    if not target_ids or len(visible_ids) < len(instruction):
        raise ValueError("Truncation removed the target or the instruction")
    full = visible_ids + target_ids
    labels = ([-100] * len(instruction)) + full[len(instruction):]
    return {
        "task_id": row["task_id"],
        "full_ids": full,
        "target_ids": target_ids,
        "labels": labels,
        "cut_pos": len(visible_ids) - 1,
        "cut": cut,
    }


def collate_pairs(pairs: list[dict], pad_id: int) -> dict:
    import torch
    if not pairs:
        raise ValueError("Empty batch")
    full_len = max(len(x["full_ids"]) for x in pairs)
    target_len = max(len(x["target_ids"]) for x in pairs)
    batch = {
        "full_ids": [], "full_mask": [], "labels": [],
        "target_ids": [], "target_mask": [], "cut_pos": [],
    }
    for x in pairs:
        f, t = x["full_ids"], x["target_ids"]
        batch["full_ids"].append(f + [pad_id] * (full_len - len(f)))
        batch["full_mask"].append([1] * len(f) + [0] * (full_len - len(f)))
        batch["labels"].append(x["labels"] + [-100] * (full_len - len(f)))
        batch["target_ids"].append(t + [pad_id] * (target_len - len(t)))
        batch["target_mask"].append([1] * len(t) + [0] * (target_len - len(t)))
        batch["cut_pos"].append(x["cut_pos"])
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def write_jsonl(path: str, rows: list[dict]):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
