"""Training and held-out evaluation: direct token CE -> LoRA + latent memory.

This module intentionally does not modify SALM source. CLI is invoked using
PYTHONPATH=/path/to/memory_lora_lab python -m memory_lora.cli.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import warnings

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .data import CausalTraceCorpus, collate, load_rows, require_disjoint
from .model import LatentMemoryAdapter, MemoryConditioner
from .sigreg import SIGReg


def default_config() -> dict:
    return {
        'model': 'LiquidAI/LFM3-1.7B-Base', 'revision': None,
        'mode': 'memory', 'out': 'runs/memory_lora/pilot',
        'train': 'runs/trace_data/train_4k.jsonl',
        'val': 'runs/trace_data/val_4k.jsonl',
        'seed': 42, 'max_steps': 10,
        'batch_size': 2, 'grad_accum': 4,
        'memory_tokens': 2048, 'recent_tokens': 384, 'target_tokens': 192,
        'min_memory_tokens': 64, 'slots': 16, 'latent_size': 256,
        'heads': 4, 'source_layer': 7, 'inject_layer': 16,
        'encoder_mode': 'backbone', 'gate_init': 0.05,
        'sigreg_weight': 0.0, 'lr': 1e-4, 'weight_decay': 0.01,
        'lora_rank': 16, 'lora_alpha': 32,
        'eval_every': 5, 'eval_tasks': 24,
        'checkpoint_every': 5, 'max_grad_norm': 1.0,
        'minimum_free_gib': 80, 'resume': None,
    }


def validate_config(config: dict) -> dict:
    cfg = {**default_config(), **config}
    if cfg['mode'] not in {'memory', 'memory_sigreg', 'recent_only'}:
        raise ValueError('mode = memory | memory_sigreg | recent_only')
    if cfg['mode'] == 'memory_sigreg' and cfg['sigreg_weight'] <= 0:
        raise ValueError('memory_sigreg needs sigreg_weight > 0')
    if cfg['mode'] != 'memory_sigreg' and cfg['sigreg_weight']:
        raise ValueError('sigreg_weight must be 0 unless mode = memory_sigreg')
    for key in ('batch_size', 'grad_accum', 'max_steps', 'memory_tokens',
                'recent_tokens', 'target_tokens', 'slots', 'eval_every',
                'eval_tasks', 'checkpoint_every'):
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['mode'] == 'memory_sigreg' and cfg['batch_size'] < 2:
        raise ValueError('SIGReg needs >=2 independent tasks per batch')
    if cfg['mode'] == 'memory_sigreg' and cfg['batch_size'] < 8:
        warnings.warn('SIGReg on <8 independent tasks is very noisy and experimental', stacklevel=2)
    if cfg['source_layer'] >= cfg['inject_layer']:
        raise ValueError('source_layer must be before inject_layer')
    return cfg


def load_lfm3_with_lora(cfg: dict, *, with_adapter_weights: str | None = None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for LFM3 runs. CPU toy tests are separate.')
    free, total = torch.cuda.mem_get_info()
    if free < float(cfg['minimum_free_gib'])*2**30:
        raise RuntimeError(f'Only {free/2**30:.1f} GiB CUDA memory free. '
                           'Do not start alongside large VLLM; check nvidia-smi.')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg.get('revision'),
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg['model'], revision=cfg.get('revision'), trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to('cuda')
    base.config.use_cache = False
    if with_adapter_weights:
        model = PeftModel.from_pretrained(base, with_adapter_weights, is_trainable=False)
    else:
        wanted = ['w1','w2','w3','q_proj','k_proj','v_proj','o_proj','in_proj','out_proj']
        names = {name.rsplit('.', 1)[-1] for name, m in base.named_modules()
                 if isinstance(m, nn.Linear)}
        targets = [x for x in wanted if x in names]
        if not {'w1','w2','w3'}.issubset(targets):
            raise RuntimeError(f'Cannot resolve LFM3 FFN names. Available: {sorted(names)}')
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg['lora_rank']), lora_alpha=int(cfg['lora_alpha']),
            lora_dropout=0.0, bias='none', target_modules=targets,
        ))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[Memory-LoRA] FFN + attention + conv LoRA targets: {targets}; '
              f'trainable={trainable:,}', flush=True)
    return model, tokenizer


def build_conditioner(model, cfg):
    hidden = int(model.config.hidden_size)
    adapter = LatentMemoryAdapter(hidden, latent_size=cfg['latent_size'],
                                  slots=cfg['slots'], heads=cfg['heads'],
                                  gate_init=cfg['gate_init']).to('cuda')
    cond = MemoryConditioner(model, adapter, cfg['source_layer'], cfg['inject_layer'],
                             cfg['encoder_mode'])
    return adapter, cond


def token_loss_batch(model, cond, adapter, batch: dict, cfg: dict,
                     *, memory_mode: str = 'real', sigreg=None):
    device = next(model.parameters()).device
    tensors = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    use_memory = cfg['mode'] != 'recent_only' or memory_mode != 'real'
    # In 'recent_only' evaluation memory_mode='zero' bypasses encoding.
    use_memory = use_memory and memory_mode != 'zero'
    memory = cond.encode(tensors['memory_ids'], tensors['memory_mask']) if use_memory else None
    with cond.inject(memory, mode=memory_mode if use_memory else 'zero'):
        output = model(input_ids=tensors['current_ids'],
                       attention_mask=tensors['current_mask'],
                       labels=tensors['labels'], use_cache=False)
    ce = output.loss.float()
    reg = ce.new_zeros(())
    if sigreg is not None and memory is not None:
        # One per TASK (slots averaged) avoids treating slots from one example
        # as statistically independent samples.
        reg = sigreg(memory.mean(dim=1))
    loss = ce + float(cfg['sigreg_weight']) * reg
    if not torch.isfinite(loss):
        raise FloatingPointError(f'NaN/Inf token or SIGReg objective: ce={ce}, sigreg={reg}')
    with torch.no_grad():
        gate = float(adapter.gate) if memory is not None else 0.0
        tokens = int((tensors['labels'][:, 1:] != -100).sum().item())
    return loss, {'token_ce': float(ce.detach()), 'sigreg': float(reg.detach()),
                  'gate': gate, 'tokens': tokens}


@torch.no_grad()
def evaluate_nll(model, adapter, cond, corpus: CausalTraceCorpus,
                 cfg: dict, *, memory_mode: str = 'real', tasks: int | None = None):
    model.eval()
    adapter.eval()
    rng = random.Random(187331)
    selected = corpus.sample(rng, min(tasks or cfg['eval_tasks'], len(corpus.task_ids)))
    losses = 0.0
    tokens = 0
    for e in selected:
        # One example per forward to avoid eval memory spikes.
        # For shuffled memory we need >=2 independent tasks, handled separately.
        if memory_mode == 'shuffle':
            raise ValueError('Use evaluate_shuffled for the shuffled-memory control')
        batch = collate([e], cfg['_pad_id'])
        _, log = token_loss_batch(model, cond, adapter, batch,
                                  {**cfg, 'sigreg_weight': 0.0},
                                  memory_mode=memory_mode)
        losses += log['token_ce'] * log['tokens']
        tokens += log['tokens']
    model.train()
    adapter.train()
    return {'nll': losses / max(tokens, 1), 'tokens': tokens, 'tasks': len(selected)}


@torch.no_grad()
def evaluate_shuffled(model, adapter, cond, corpus, cfg, *, tasks: int | None = None):
    model.eval()
    adapter.eval()
    rng = random.Random(187331)
    selected = corpus.sample(rng, min(tasks or cfg['eval_tasks'], len(corpus.task_ids)))
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(selected), 2):
        examples = selected[i:i+2]
        if len(examples) < 2:
            continue
        batch = collate(examples, cfg['_pad_id'])
        _, log = token_loss_batch(model, cond, adapter, batch,
                                  {**cfg, 'sigreg_weight': 0.0},
                                  memory_mode='shuffle')
        total_loss += log['token_ce'] * log['tokens']
        total_tokens += log['tokens']
    model.train()
    adapter.train()
    return {'nll': total_loss/max(total_tokens,1), 'tokens': total_tokens,
            'tasks': 2*(len(selected)//2)}


def _save_checkpoint(run_dir: Path, step: int, model, adapter, optimizer, cfg):
    dest = run_dir / f'checkpoint_{step:06d}'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {dest}')
    dest.mkdir(parents=True)
    model.save_pretrained(dest / 'lora', safe_serialization=True)
    torch.save(adapter.state_dict(), dest / 'memory.pt')
    torch.save({'step': step, 'optimizer': optimizer.state_dict()},
               dest / 'trainer_state.pt')
    (dest / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f'[Memory-LoRA] checkpoint: {dest}', flush=True)


def _restore_checkpoint(checkpoint: str, model, adapter, optimizer):
    # Restores weights through PEFT PeftModel (LoRA weights only).
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    src = Path(checkpoint)
    lora = load_file(str(src / 'lora' / 'adapter_model.safetensors'))
    set_peft_model_state_dict(model, lora)
    adapter.load_state_dict(torch.load(src/'memory.pt', map_location='cpu', weights_only=True))
    state = torch.load(src/'trainer_state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    return int(state['step'])


def _verify_run_dir(out: Path, resume: str | None):
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out/'history.jsonl').exists():
        raise FileExistsError(f'Existing experiment history in {out}. '
                              'Choose another output directory or resume explicitly.')


def train(user_config: dict):
    cfg = validate_config(user_config)
    out = Path(cfg['out'])
    _verify_run_dir(out, cfg.get('resume'))
    train_rows = load_rows(cfg['train'])
    val_rows = load_rows(cfg['val'])
    require_disjoint(train_rows, val_rows)
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    model, tokenizer = load_lfm3_with_lora(cfg)
    cfg['_pad_id'] = tokenizer.pad_token_id
    cfg['resolved_revision'] = getattr(model.config, '_commit_hash', None)
    train_corpus = CausalTraceCorpus(train_rows, tokenizer,
                                    memory_tokens=cfg['memory_tokens'],
                                    recent_tokens=cfg['recent_tokens'],
                                    target_tokens=cfg['target_tokens'],
                                    min_memory_tokens=cfg['min_memory_tokens'])
    val_corpus = CausalTraceCorpus(val_rows, tokenizer,
                                  memory_tokens=cfg['memory_tokens'],
                                  recent_tokens=cfg['recent_tokens'],
                                  target_tokens=cfg['target_tokens'],
                                  min_memory_tokens=cfg['min_memory_tokens'])
    if len(train_corpus.task_ids) < cfg['batch_size']:
        raise ValueError('Not enough independent eligible tasks for training batch')
    print(json.dumps({'eligible_train_tasks':len(train_corpus.task_ids),
                      'eligible_val_tasks':len(val_corpus.task_ids),
                      'train_transitions':len(train_corpus.candidates),
                      'tool_targets':train_corpus.tool_targets,
                      'reasoning_targets':train_corpus.reasoning_targets}), flush=True)
    adapter, conditioner = build_conditioner(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg['mode'] != 'recent_only':
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sigreg = SIGReg() if cfg['mode'] == 'memory_sigreg' else None
    if sigreg is not None:
        sigreg.to('cuda')
    start = _restore_checkpoint(cfg['resume'], model, adapter, optimizer) if cfg.get('resume') else 0
    (out/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    history = out/'history.jsonl'
    model.train(); adapter.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    for step in range(start, cfg['max_steps']):
        totals: list[dict] = []
        for micro in range(cfg['grad_accum']):
            rng = random.Random(cfg['seed'] + step*10009 + micro)
            examples = train_corpus.sample(rng, cfg['batch_size'])
            batch = collate(examples, tokenizer.pad_token_id)
            loss, log = token_loss_batch(model, conditioner, adapter, batch, cfg,
                                         memory_mode='zero' if cfg['mode']=='recent_only' else 'real',
                                         sigreg=sigreg)
            (loss / cfg['grad_accum']).backward()
            totals.append(log)
        norm = clip_grad_norm_(params, float(cfg['max_grad_norm']))
        if not math.isfinite(float(norm)):
            raise FloatingPointError('Non-finite gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        done = step+1
        entry = {'step':done, 'mode':cfg['mode'],
                 'ce':sum(r['token_ce'] for r in totals)/len(totals),
                 'sigreg':sum(r['sigreg'] for r in totals)/len(totals),
                 'gate':float(adapter.gate.detach()),
                 'grad_norm':float(norm),
                 'train_tokens':sum(r['tokens'] for r in totals),
                 'seconds':time.monotonic()-start_time,
                 'cuda_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        if done % cfg['eval_every']==0 or done == cfg['max_steps']:
            metrics = evaluate_nll(model, adapter, conditioner, val_corpus, cfg,
                                   memory_mode='real' if cfg['mode']!='recent_only' else 'zero')
            entry.update({'val_nll':metrics['nll'], 'val_tokens':metrics['tokens']})
            if cfg['mode']!='recent_only':
                # Same model, SAME tasks: memory on versus removed.
                zero = evaluate_nll(model, adapter, conditioner, val_corpus, cfg, memory_mode='zero')
                entry['val_zero_memory_nll'] = zero['nll']
                entry['memory_gain_nll'] = zero['nll'] - metrics['nll']
        with history.open('a',encoding='utf-8') as f:
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if done % cfg['checkpoint_every']==0 or done == cfg['max_steps']:
            _save_checkpoint(out, done, model, adapter, optimizer, cfg)
    return {'out':str(out), 'steps':cfg['max_steps'], 'mode':cfg['mode'],
            'train_tasks':len(train_corpus.task_ids), 'val_tasks':len(val_corpus.task_ids)}


def evaluate_checkpoint(checkpoint: str, *, limit: int = 50):
    ckpt = Path(checkpoint)
    cfg = json.loads((ckpt/'config.json').read_text())
    model, tokenizer = load_lfm3_with_lora(cfg, with_adapter_weights=str(ckpt/'lora'))
    cfg['_pad_id'] = tokenizer.pad_token_id
    rows = load_rows(cfg['val'])
    corpus = CausalTraceCorpus(rows, tokenizer,
                               memory_tokens=cfg['memory_tokens'],
                               recent_tokens=cfg['recent_tokens'],
                               target_tokens=cfg['target_tokens'],
                               min_memory_tokens=cfg['min_memory_tokens'])
    adapter, cond = build_conditioner(model, cfg)
    adapter.load_state_dict(torch.load(ckpt/'memory.pt',map_location='cpu',weights_only=True))
    model.eval(); adapter.eval()
    real = evaluate_nll(model, adapter, cond, corpus, cfg, memory_mode='real' if cfg['mode']!='recent_only' else 'zero', tasks=limit)
    zero = evaluate_nll(model, adapter, cond, corpus, cfg, memory_mode='zero', tasks=limit)
    shuffle = evaluate_shuffled(model, adapter, cond, corpus, cfg, tasks=limit) if cfg['mode']!='recent_only' else None
    result = {'checkpoint':str(ckpt), 'memory':real, 'zero_memory':zero,
              'shuffled_memory':shuffle, 'memory_gain_nll':zero['nll']-real['nll']}
    output = ckpt/'evaluation.json'
    output.write_text(json.dumps(result,indent=2))
    return result


@torch.no_grad()
def generate_sample(checkpoint: str, *, index: int = 0, max_new_tokens: int = 120):
    """Qualitative test only; no external tool execution or benchmark scoring."""
    ckpt = Path(checkpoint)
    cfg = json.loads((ckpt/'config.json').read_text())
    model, tokenizer = load_lfm3_with_lora(cfg,with_adapter_weights=str(ckpt/'lora'))
    rows = load_rows(cfg['val'])
    corpus = CausalTraceCorpus(rows, tokenizer,
                               memory_tokens=cfg['memory_tokens'],recent_tokens=cfg['recent_tokens'],
                               target_tokens=cfg['target_tokens'],min_memory_tokens=cfg['min_memory_tokens'])
    adapter,cond = build_conditioner(model,cfg)
    adapter.load_state_dict(torch.load(ckpt/'memory.pt', map_location='cpu', weights_only=True))
    model.eval(); adapter.eval()
    tid, old, recent,kind,boundary = corpus.candidates[index%len(corpus.candidates)]
    old_t = torch.tensor([old],device='cuda')
    mask = torch.ones_like(old_t)
    recent_t = torch.tensor([recent],device='cuda')
    memory = cond.encode(old_t,mask)
    result = {'task_id':tid, 'kind':kind,
              'recent_tail':tokenizer.decode(recent[-120:]),
              'reference_next':tokenizer.decode(corpus._targets[index%len(corpus.candidates)])}
    for mode in ['real','zero']:
        with cond.inject(memory if mode=='real' else None,mode=mode):
            gen=model.generate(input_ids=recent_t,do_sample=False,max_new_tokens=max_new_tokens,
                               use_cache=True,pad_token_id=tokenizer.pad_token_id)
        result[f'generated_{mode}']=tokenizer.decode(gen[0,len(recent):],skip_special_tokens=True)
    return result
