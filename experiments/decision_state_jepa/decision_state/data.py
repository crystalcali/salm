"""Causally valid event transitions, stable task-level split, and streaming HF ingestion.

Important: a target is ALWAYS one assistant-authored event. Tool output, user
messages, and fresh external observations become context only AFTER arrival.
Do not use metadata.reference_patch or resolved flags as model input.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterator


def stable_split(key: str, val_percent: int = 10) -> str:
    """Task-disjoint, independent of stream order and model/harness."""
    if not 1 <= val_percent < 50:
        raise ValueError('val_percent must be in [1,49]')
    bucket = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:12], 16) % 100
    return 'val' if bucket < val_percent else 'train'


def content_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(content_text(x) for x in value if x is not None)
    if isinstance(value, dict):
        if value.get('type') == 'text' and 'text' in value:
            return str(value['text'])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_messages(messages: Any) -> list[dict]:
    """Keep the actual order and provenance; never promote observations to labels."""
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError('messages must be a list')
    result = []
    for raw in messages:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError('message is not an object')
        role = str(raw.get('role') or '').lower()
        if role == 'ai':
            role = 'assistant'
        if role in {'function', 'observation'}:
            role = 'tool'
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise ValueError('unsupported role: ' + role)
        text = content_text(raw.get('content') if 'content' in raw else raw.get('text'))
        tool_calls = raw.get('tool_calls') or raw.get('function_call')
        # The original tool-call arguments remain assistant-authored content.
        if tool_calls:
            text += '\n<tool_call>\n' + content_text(tool_calls) + '\n</tool_call>'
        if not text.strip():
            continue
        category = ('tool_call' if role == 'assistant' and tool_calls
                    else 'answer' if role == 'assistant' and raw.get('finish_reason') == 'stop'
                    else 'reasoning' if role == 'assistant' else role)
        result.append({'role': role, 'kind': category, 'text': text})
    return result


def convert_nvidia(row: dict) -> dict | None:
    """Accept only resolved SWE traces, exclude Terminal-Bench evaluation content."""
    key = str(row.get('instance_id') or '')
    # Never train on Terminal-Bench 2.1 tasks, their mirrors or canaries.
    provenance = ' '.join((key, str(row.get('repo','')),str(row.get('hf_dataset_name','')))).lower()
    if any(x in provenance for x in ('terminal-bench', 'terminal_bench', 'tbench.ai',
                                     'harborframework/terminal', 'harbor-canary')):
        return None
    if row.get('resolved') != 1:
        return None
    key = str(row.get('instance_id') or '').strip()
    if not key:
        return None
    messages = normalize_messages(row.get('messages'))
    if sum(m['role'] == 'assistant' for m in messages) < 2:
        return None
    if not any(m['role'] == 'tool' for m in messages):
        return None
    if not any(m['role'] == 'user' for m in messages):
        return None
    # Scan task text and tool observations too; accidental benchmark contamination
    # must not enter training even if the source metadata is mislabeled.
    if any(bad in m['text'].lower() for m in messages for bad in
           ('harbor-canary', 'terminal-bench-2.1', 'terminal-bench 2.1')):
        return None
    return {
        'task_id': 'openswe:' + key,
        'messages': messages,
        'source': 'nvidia/Open-SWE-Traces:v2.0',
        'trajectory_id': str(row.get('trajectory_id') or ''),
        'verification': {'task_resolved': True, 'steps_individually_verified': False},
        'license': str(row.get('license') or 'see source dataset'),
    }


def prepare_nvidia(out_dir: str, *, train_limit: int = 12000, val_limit: int = 1000,
                   split: str = 'claudecode', val_percent: int = 10,
                   scan_limit: int = 200000) -> dict:
    """Stream from HF on DGX; store raw event order in local, ignored JSONL files.

    Fails loudly when the requested number of accepted tasks is not found.
    Exact same instance_id across multiple harnesses is never split across sets.
    The source dataset is large: do NOT vendor its records into GitHub.
    """
    if min(train_limit, val_limit, scan_limit) < 1:
        raise ValueError('limits must be positive')
    from datasets import load_dataset
    ds = load_dataset('nvidia/Open-SWE-Traces', 'v2.0', split=split, streaming=True)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'train': directory/'train.jsonl', 'val': directory/'val.jsonl'}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Output exists; pick an empty directory (never overwrite datasets)')
    temporary = {k: p.with_suffix('.jsonl.partial') for k, p in paths.items()}
    counts = {'train': 0, 'val': 0}
    seen: set[str] = set()
    scanned = 0
    try:
        with temporary['train'].open('w', encoding='utf-8') as tf, temporary['val'].open('w', encoding='utf-8') as vf:
            files = {'train': tf, 'val': vf}
            for row in ds:
                scanned += 1
                if scanned > scan_limit:
                    break
                try:
                    item = convert_nvidia(row)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if item is None or item['task_id'] in seen:
                    continue
                seen.add(item['task_id'])
                part = stable_split(item['task_id'], val_percent)
                if counts[part] >= (val_limit if part == 'val' else train_limit):
                    continue
                files[part].write(json.dumps(item, ensure_ascii=False) + '\n')
                counts[part] += 1
                if counts['train'] >= train_limit and counts['val'] >= val_limit:
                    break
        if not all(counts[k] > 0 for k in counts):
            raise RuntimeError(f'No usable train or val trajectories in {scanned} scanned rows: {counts}')
        if counts['train'] < train_limit or counts['val'] < val_limit:
            raise RuntimeError(f'Dataset under target size (scanned {scanned}): {counts}; increase --scan-limit or reduce limits')
        for k in paths:
            temporary[k].replace(paths[k])
    except BaseException:
        for p in temporary.values():
            p.unlink(missing_ok=True)
        raise
    manifest = {'dataset': 'nvidia/Open-SWE-Traces', 'config': 'v2.0', 'split': split,
                'success_filter': 'resolved == 1', 'train': counts['train'], 'val': counts['val'],
                'scanned': scanned, 'train_file': str(paths['train']), 'val_file': str(paths['val']),
                'notes': 'Task-level success is source-reported; reasoning steps are not individually verified. '
                         'No gold patches, success flags or future observations are in model input.'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def serialize_event(message: dict) -> str:
    return f"<{message['role']} kind={message.get('kind','other')}>\n{message['text']}\n</{message['role']}>\n"


def targets_for_row(row: dict) -> list[int]:
    return [i for i, m in enumerate(row['messages']) if m['role'] == 'assistant' and i > 0]


def decision_importance(row: dict, index: int) -> float:
    """Heuristic event-level importance, not verified causal impact.

    Both SFT and JEPA use identical sampling. The test set is NOT reweighted.
    Provenance role, not a future tool result, defines the sampling priority.
    """
    events = row['messages']
    if not 0 < index < len(events) or events[index]['role'] != 'assistant':
        raise ValueError('Importance is defined only for assistant events')
    previous = events[index - 1]['role']
    if events[index].get('kind') == 'tool_call':
        return 3.0
    if previous == 'tool':
        return 2.5
    if previous == 'user':
        return 2.0
    return 1.0


@dataclass
class Example:
    task_id: str
    input_ids: list[int]
    labels: list[int]
    prefix_length: int
    kind: str


class DiskCorpus:
    """JSONL byte-offset index: avoids loading 12k long traces into RAM."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.index: list[tuple[int, str, list[int]]] = []
        self.task_ids: set[str] = set()
        with self.path.open('rb') as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                r = json.loads(line)
                tid = str(r.get('task_id') or '')
                if not tid or tid in self.task_ids:
                    raise ValueError('Missing or duplicate task_id: '+tid)
                if not isinstance(r.get('messages'), list):
                    raise ValueError('Expected normalized messages list')
                indices = targets_for_row(r)
                if indices:
                    self.index.append((off, tid, indices))
                    self.task_ids.add(tid)
        if not self.index:
            raise ValueError('No assistant targets in '+str(path))

    def _read(self, offset: int) -> dict:
        with self.path.open('rb') as f:
            f.seek(offset)
            return json.loads(f.readline())

    def sample(self, tokenizer, rng: random.Random, *, max_length: int,
               target_tokens: int, max_attempts: int = 64,
               min_context_tokens: int = 64) -> Example:
        for _ in range(max_attempts):
            offset, tid, indices = rng.choice(self.index)
            row = self._read(offset)
            choice = rng.choices(indices, weights=[decision_importance(row,i) for i in indices], k=1)[0]
            try:
                return make_example(row, choice, tokenizer, max_length=max_length,
                                    target_tokens=target_tokens, min_context_tokens=min_context_tokens)
            except ValueError:
                continue
        raise RuntimeError('Too few valid training transitions; check dataset and context configuration')

    def fixed_examples(self, tokenizer, *, count: int, max_length: int, target_tokens: int,
                       seed: int = 1910) -> list[Example]:
        rng = random.Random(seed)
        choices = rng.sample(self.index, min(count, len(self.index)))
        samples = []
        for offset, _, indices in choices:
            row = self._read(offset)
            for _ in range(min(3, len(indices))):
                choice = rng.choice(indices)
                try:
                    samples.append(make_example(row, choice, tokenizer, max_length=max_length,
                                                target_tokens=target_tokens))
                    break
                except ValueError:
                    continue
        return samples


def make_example(row: dict, target_index: int, tokenizer, *, max_length: int,
                 target_tokens: int, min_context_tokens: int = 64) -> Example:
    if not 2 <= target_tokens < max_length:
        raise ValueError('Invalid length limits')
    events = row['messages']
    if target_index <= 0 or target_index >= len(events) or events[target_index]['role'] != 'assistant':
        raise ValueError('Targets must be assistant-authored and have earlier context')
    # Fully independent of messages after target_index. No future observation.
    target = tokenizer.encode(serialize_event(events[target_index]), add_special_tokens=False)[:target_tokens]
    if len(target) < 2:
        raise ValueError('Target too short')
    context_budget = max_length - len(target)
    # Encode events backwards until context budget; never include later events.
    pieces = []
    total = 0
    for event in reversed(events[:target_index]):
        ids = tokenizer.encode(serialize_event(event), add_special_tokens=False)
        if total + len(ids) <= context_budget:
            pieces.append(ids)
            total += len(ids)
        else:
            remaining = context_budget - total
            if remaining > 0:
                pieces.append(ids[-remaining:])
                total += remaining
            break
    prefix = [t for piece in reversed(pieces) for t in piece]
    if len(prefix) < min_context_tokens:
        raise ValueError('Too little prefix for decision state')
    full = prefix + target
    return Example(str(row['task_id']), full, [-100] * len(prefix) + target,
                   len(prefix), events[target_index].get('kind', 'reasoning'))


def verify_disjoint(a: DiskCorpus, b: DiskCorpus):
    both = a.task_ids & b.task_ids
    if both:
        raise ValueError(f'Task leakage between splits: {sorted(both)[:3]}')


def collate(examples: list[Example], pad_id: int) -> dict:
    import torch
    max_len = max(len(x.input_ids) for x in examples)
    return {
        'input_ids': torch.tensor([x.input_ids + [pad_id] * (max_len-len(x.input_ids)) for x in examples], dtype=torch.long),
        'attention_mask': torch.tensor([[1]*len(x.input_ids)+[0]*(max_len-len(x.input_ids)) for x in examples], dtype=torch.long),
        'labels': torch.tensor([x.labels + [-100]*(max_len-len(x.labels)) for x in examples],dtype=torch.long),
        'prefix_length': torch.tensor([x.prefix_length for x in examples],dtype=torch.long),
    }
