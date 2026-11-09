"""Repeatable Terminal-Bench 2.1 evaluation for the *two* SALM variants.

The 89 benchmark problems are NEVER converted to train examples. Runs invoke
Harbor's official sandbox and task verifier, not a proxy NLL metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HARBOR_DATASET = 'terminal-bench/terminal-bench-2-1'
HUGGINGFACE_REPO = 'https://huggingface.co/datasets/harborframework/terminal-bench-2.1'


def manifest(checkpoint: Path) -> dict:
    cfgfile=checkpoint/'config.json'
    if not cfgfile.is_file():
        raise FileNotFoundError(f'Missing checkpoint config: {cfgfile}')
    cfg=json.loads(cfgfile.read_text())
    if cfg['mode'] not in ('sft','decision_jepa'):
        raise ValueError('Unsupported experiment mode')
    files=[checkpoint/'lora'/'adapter_model.safetensors']
    if cfg['mode']=='decision_jepa':files.append(checkpoint/'decision_head.safetensors')
    hashes={}
    for p in files:
        if not p.is_file():raise FileNotFoundError(f'Missing checkpoint weights: {p}')
        digest=hashlib.sha256()
        with p.open('rb') as fp:
            for block in iter(lambda:fp.read(1<<20),b''):
                digest.update(block)
        hashes[str(p.relative_to(checkpoint))]=digest.hexdigest()
    return {'checkpoint':str(checkpoint.resolve()),'mode':cfg['mode'],
            'source_data':cfg['train'],'training_context':cfg['max_seq_length'],
            'weights_sha256':hashes}


def command_for_benchmark(*, limit: int | None = 5, trials: int = 1,
                          use_hf_repo: bool = False) -> list[str]:
    cmd=['harbor','run']
    if use_hf_repo:
        cmd.extend(['--repo',HUGGINGFACE_REPO])
    else:
        cmd.extend(['-d',HARBOR_DATASET])
    cmd.extend(['--agent-import-path','decision_state.harbor_agent:SALMTerminalAgent',
                '-k',str(trials),'--n-concurrent','1'])
    if limit is not None:
        cmd.extend(['-l',str(limit)])
    return cmd


def run_benchmark(checkpoint: str, *, limit: int | None = 5,
                  trials: int = 1, use_hf_repo: bool = False):
    if trials<1 or (limit is not None and limit<1):
        raise ValueError('trials and limit must be positive')
    ckpt=Path(checkpoint).expanduser().resolve()
    record=manifest(ckpt)
    if shutil.which('harbor') is None:
        raise RuntimeError('Harbor CLI not installed. Install with: uv tool install harbor')
    jobs=Path.home()/'.harbor'/'jobs'
    before={str(p.resolve()) for p in jobs.iterdir() if p.is_dir()} if jobs.exists() else set()
    cmd=command_for_benchmark(limit=limit,trials=trials,use_hf_repo=use_hf_repo)
    record.update({'benchmark':HARBOR_DATASET,'hf_mirror':HUGGINGFACE_REPO,
                   'task_limit':limit,'trials_per_task':trials,
                   'command':cmd,'started_unix':time.time()})
    outdir=Path('runs/decision_state/benchmark')
    outdir.mkdir(parents=True,exist_ok=True)
    name='tb21_'+record['mode']+'_'+str(int(record['started_unix']))
    record_file=outdir/(name+'.json')
    record_file.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Official Harbor benchmark; task solutions are excluded from training.',flush=True)
    print('Running: '+' '.join(cmd),flush=True)
    proc=subprocess.run(cmd,env={**os.environ,'SALM_DECISION_CHECKPOINT':str(ckpt)})
    after={str(p.resolve()) for p in jobs.iterdir() if p.is_dir()} if jobs.exists() else set()
    record.update({'ended_unix':time.time(),'return_code':proc.returncode,
                   'new_job_directories':sorted(after-before)})
    record_file.write_text(json.dumps(record,indent=2),encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Harbor failed with exit code {proc.returncode}. See job logs and {record_file}')
    return record


def main():
    parser=argparse.ArgumentParser(description='Harbor Terminal-Bench 2.1 official evaluation')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--limit',type=int,default=5,
                        help='Number of benchmark tasks; 0 runs all 89 tasks')
    parser.add_argument('--trials',type=int,default=1,help='Trials per task; 3–5 for robust results')
    parser.add_argument('--hf-repo',action='store_true',
                        help='Use the user-provided Hugging Face dataset URL instead of public Harbor registry')
    args=parser.parse_args()
    outcome=run_benchmark(args.checkpoint,limit=None if args.limit==0 else args.limit,
                          trials=args.trials,use_hf_repo=args.hf_repo)
    print(json.dumps(outcome,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
