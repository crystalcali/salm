"""CLI: python -m decision_state.cli prepare | inspect | train | evaluate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='SALM Decision-State JEPA vs matched LoRA SFT')
    cmds=p.add_subparsers(dest='command',required=True)
    d=cmds.add_parser('prepare',help='Download successful NVIDIA agent trajectories to local JSONL')
    d.add_argument('--out-dir',default='runs/decision_state_data')
    d.add_argument('--split',default='claudecode')
    d.add_argument('--train-limit',type=int,default=12000)
    d.add_argument('--val-limit',type=int,default=1000)
    d.add_argument('--scan-limit',type=int,default=200000)
    d.add_argument('--val-percent',type=int,default=10)
    v=cmds.add_parser('inspect',help='Show local counts and check task-level leakage')
    v.add_argument('--train',default='runs/decision_state_data/train.jsonl')
    v.add_argument('--val',default='runs/decision_state_data/val.jsonl')
    t=cmds.add_parser('train',help='Train one of exactly two variants')
    t.add_argument('--config',required=True)
    t.add_argument('--out')
    t.add_argument('--resume')
    e=cmds.add_parser('evaluate',help='Evaluate held-out teacher-forced token NLL')
    e.add_argument('--checkpoint',required=True)
    e.add_argument('--limit',type=int,default=100)
    e.add_argument('--out')
    b=cmds.add_parser('benchmark',help='Official Harbor Terminal-Bench 2.1 verifier')
    b.add_argument('--checkpoint',required=True)
    b.add_argument('--limit',type=int,default=5,help='0 means all 89 tasks')
    b.add_argument('--trials',type=int,default=1)
    b.add_argument('--hf-repo',action='store_true',help='Use the user-specified HF mirror')
    args=p.parse_args()
    if args.command=='prepare':
        from .data import prepare_nvidia
        result=prepare_nvidia(args.out_dir,train_limit=args.train_limit,
                val_limit=args.val_limit,split=args.split,scan_limit=args.scan_limit,
                val_percent=args.val_percent)
    elif args.command=='inspect':
        from .data import DiskCorpus,verify_disjoint
        a=DiskCorpus(args.train);b=DiskCorpus(args.val);verify_disjoint(a,b)
        result={'train_tasks':len(a.index),'val_tasks':len(b.index),'overlap':0,
                'train':args.train,'val':args.val}
    elif args.command=='train':
        from .train import train
        cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
        if args.out:cfg['out']=args.out
        if args.resume:cfg['resume']=args.resume
        result=train(cfg)
    elif args.command=='benchmark':
        from .benchmark import run_benchmark
        result=run_benchmark(args.checkpoint,limit=None if args.limit==0 else args.limit,
                             trials=args.trials,use_hf_repo=args.hf_repo)
    else:
        from .train import evaluate_checkpoint
        result=evaluate_checkpoint(args.checkpoint,count=args.limit)
        if args.out:
            out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
