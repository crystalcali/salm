"""Causal training examples from math or chronological tool-agent trajectories.

The target is always the NEXT assistant-produced step/message. Tool/user
observations are never synthesized or included from the future.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class Example:
    task_id: str
    past_ids: tuple[int, ...]  # old context -> memory encoder only
    recent_ids: tuple[int, ...]  # directly visible to language model
    target_ids: tuple[int, ...]  # only next assistant step
    target_kind: str
    boundary: int


def load_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    with Path(path).open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{lineno}: malformed JSON') from exc
            tid = str(row.get('task_id', '')).strip()
            if not tid or tid in seen:
                raise ValueError(f'{path}:{lineno}: missing/duplicate task_id {tid!r}')
            if not isinstance(row.get('prompt') or row.get('question'), str):
                raise ValueError(f'{path}:{lineno}: missing prompt/question')
            if 'messages' not in row and not isinstance(row.get('steps'), list):
                raise ValueError(f'{path}:{lineno}: steps or messages required')
            seen.add(tid)
            rows.append(row)
    if not rows:
        raise ValueError(f'Empty JSONL file: {path}')
    return rows


def require_disjoint(train: list[dict], val: list[dict]) -> None:
    left = {r['task_id'] for r in train}
    right = {r['task_id'] for r in val}
    if left & right:
        raise ValueError(f'task_id leak between train and val: {list(left & right)[:3]}')
    def fingerprints(rs):
        return {hashlib.sha256(str(r.get('prompt') or r.get('question')).strip().encode()).hexdigest()
                for r in rs}
    if fingerprints(train) & fingerprints(val):
        raise ValueError('Repeated prompt/question between train and validation')


def _content(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _serialize_message(m: dict) -> str:
    role = str(m.get('role', '')).lower()
    if role not in {'system', 'user', 'assistant', 'tool'}:
        raise ValueError(f'Unsupported role: {role!r}')
    ident = m.get('name') or m.get('tool_call_id') or ''
    payload = _content(m.get('content'))
    calls = m.get('tool_calls')
    if calls:
        payload += '\nTool calls: ' + _content(calls)
    return f'<{role}{" " + str(ident) if ident else ""}>\n{payload}\n</{role}>\n'


def _events(row: dict) -> tuple[list[str], list[bool], list[str]]:
    """Chronological events + masks for assistant-authored targets.

    Source order is trusted as chronology but never inferred from string content.
    A target event must be an assistant message, never tool/user observation.
    """
    prompt = str(row.get('prompt') or row.get('question')).strip()
    first = f'Task:\n{prompt}\n'
    if 'messages' in row:
        messages = row['messages']
        if not isinstance(messages, list):
            raise ValueError('messages must be a list')
        segments = [first]
        flags = [False]
        kinds = ['prompt']
        for m in messages:
            if not isinstance(m, dict):
                raise ValueError('message must be dict')
            role = str(m.get('role', '')).lower()
            s = _serialize_message(m)
            segments.append(s)
            flags.append(role == 'assistant' and (bool(_content(m.get('content')).strip()) or bool(m.get('tool_calls'))))
            kinds.append('tool_call' if role == 'assistant' and m.get('tool_calls') else role)
        return segments, flags, kinds
    steps = row.get('steps') or []
    if not all(isinstance(s, str) for s in steps):
        raise ValueError('steps must be strings')
    segments = [first] + [f'Step {i+1}: {s.strip()}\n' for i, s in enumerate(steps) if s.strip()]
    answer = row.get('final_answer')
    if isinstance(answer, str) and answer.strip():
        segments.append('Final answer: ' + answer.strip() + '\n')
    flags = [False] + [True] * (len(segments) - 1)
    kinds = (['prompt'] + ['reasoning'] * (len(segments) - 2) +
             ["answer" if isinstance(answer, str) and answer.strip() else "reasoning"]) if len(segments) > 1 else ['prompt']
    return segments, flags, kinds


class CausalTraceCorpus:
    def __init__(self, rows: list[dict], tokenizer, *, memory_tokens: int = 2048,
                 recent_tokens: int = 384, target_tokens: int = 192,
                 min_memory_tokens: int = 64):
        if min(memory_tokens, recent_tokens, target_tokens) < 2:
            raise ValueError('Memory/recent/target token budgets must be at least 2')
        self.tokenizer = tokenizer
        self.memory_tokens = memory_tokens
        self.recent_tokens = recent_tokens
        self.target_tokens = target_tokens
        self.min_memory_tokens = min_memory_tokens
        self.candidates: list[tuple[str, tuple[int, ...], tuple[int, ...], str, int]] = []
        self._candidate_targets: list[tuple[int, ...]] = []
        self.by_task: dict[str, list[int]] = {}
        self.row_count = len(rows)
        self.tool_targets = 0
        self.reasoning_targets = 0
        for row in rows:
            segments, flags, kinds = _events(row)
            ids = [tokenizer.encode(s, add_special_tokens=False) for s in segments]
            prefix: list[int] = []
            for i, tokens in enumerate(ids):
                if flags[i] and tokens:
                    if len(prefix) >= recent_tokens + min_memory_tokens:
                        old = tuple(prefix[:-recent_tokens][-memory_tokens:])
                        recent = tuple(prefix[-recent_tokens:])
                        next_tokens = tuple(tokens[:target_tokens])
                        if len(next_tokens) >= 2 and len(old) >= min_memory_tokens:
                            idx = len(self.candidates)
                            self.candidates.append((row['task_id'], old, recent, kinds[i], i))
                            self._candidate_targets.append(next_tokens)
                            self.by_task.setdefault(row['task_id'], []).append(idx)
                            if kinds[i] == 'tool_call':
                                self.tool_targets += 1
                            else:
                                self.reasoning_targets += 1
                prefix.extend(tokens)
        if not self.candidates:
            raise ValueError('No eligible transitions: reduce recent_tokens/min_memory_tokens, or use longer traces')
        self.task_ids = sorted(self.by_task)

    def sample(self, rng: random.Random, tasks: int) -> list[Example]:
        if tasks > len(self.task_ids):
            raise ValueError(f'Need {tasks} independent tasks; only {len(self.task_ids)} eligible')
        result = []
        for task in rng.sample(self.task_ids, tasks):
            j = rng.choice(self.by_task[task])
            tid, old, recent, kind, boundary = self.candidates[j]
            # target is next assistant-authored segment ONLY; prior tool outputs
            # may appear in old/recent, future tool/user events never can.
            original_index = self.candidates[j][4]
            result.append(Example(tid, old, recent, self._get_target_tokens(j), kind, original_index))
        return result

    def _get_target_tokens(self, j: int) -> tuple[int, ...]:
        # Target tokens stored in a separate array to avoid exporting source text
        return self._targets[j]

    @property
    def _targets(self) -> list[tuple[int, ...]]:
        # Populated by the class factory below; always aligned with candidates.
        return self._candidate_targets


def collate(examples: list[Example], pad_id: int):
    import torch
    if not examples:
        raise ValueError('Empty examples')
    max_old = max(len(x.past_ids) for x in examples)
    max_cur = max(len(x.recent_ids) + len(x.target_ids) for x in examples)
    mem = []
    mem_mask = []
    current = []
    cur_mask = []
    labels = []
    for e in examples:
        old = list(e.past_ids)
        vis = list(e.recent_ids)
        target = list(e.target_ids)
        if not old or not vis or not target:
            raise ValueError('A view was empty')
        joined = vis + target
        mem.append(old + [pad_id] * (max_old-len(old)))
        mem_mask.append([1]*len(old) + [0]*(max_old-len(old)))
        current.append(joined + [pad_id]*(max_cur-len(joined)))
        cur_mask.append([1]*len(joined) + [0]*(max_cur-len(joined)))
        labels.append([-100]*len(vis)+target+[-100]*(max_cur-len(joined)))
    return {
        'memory_ids': torch.tensor(mem, dtype=torch.long),
        'memory_mask': torch.tensor(mem_mask, dtype=torch.long),
        'current_ids': torch.tensor(current, dtype=torch.long),
        'current_mask': torch.tensor(cur_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }
