"""Run as python -m memory_lora.cli ... from the SALM working directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import CausalTraceCorpus, load_rows, require_disjoint
from .runner import default_config, train, evaluate_checkpoint, generate_sample, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Memory-LoRA laboratory for SALM (standalone)')
    cmd = parser.add_subparsers(dest='command', required=True)
    p = cmd.add_parser('inspect-data', help='Check causal target coverage without loading LFM3 weights')
    p.add_argument('--config', required=True)
    p = cmd.add_parser('train', help='Train memory LoRA or recent-only control')
    p.add_argument('--config', required=True)
    p = cmd.add_parser('evaluate', help='Compare memory/zero/shuffled on same held-out tasks')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--limit', type=int, default=24)
    p = cmd.add_parser('sample', help='Qualitative generation with real vs no memory')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=120)
    p = cmd.add_parser('print-config', help='Print valid default config')
    args = parser.parse_args(argv)
    if args.command=='print-config':
        print(json.dumps(default_config(), indent=2, ensure_ascii=False))
        return
    if args.command in {'train','inspect-data'}:
        cfg=validate_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
        if args.command=='train':
            print(json.dumps(train(cfg), ensure_ascii=False, indent=2))
            return
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg.get('revision'),trust_remote_code=True)
        tr=load_rows(cfg['train']); va=load_rows(cfg['val'])
        require_disjoint(tr,va)
        kw=dict(memory_tokens=cfg['memory_tokens'], recent_tokens=cfg['recent_tokens'],
                target_tokens=cfg['target_tokens'],min_memory_tokens=cfg['min_memory_tokens'])
        a=CausalTraceCorpus(tr,tok,**kw)
        b=CausalTraceCorpus(va,tok,**kw)
        print(json.dumps({'train_tasks_total':len(tr),'train_eligible_tasks':len(a.task_ids),
                          'train_transitions':len(a.candidates),'train_tool_targets':a.tool_targets,
                          'train_other_targets':a.reasoning_targets,
                          'val_tasks_total':len(va),'val_eligible_tasks':len(b.task_ids),
                          'val_transitions':len(b.candidates)},ensure_ascii=False,indent=2))
        return
    if args.command=='evaluate':
        print(json.dumps(evaluate_checkpoint(args.checkpoint, limit=args.limit),ensure_ascii=False,indent=2))
    elif args.command=='sample':
        print(json.dumps(generate_sample(args.checkpoint,index=args.index,max_new_tokens=args.max_new_tokens),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
