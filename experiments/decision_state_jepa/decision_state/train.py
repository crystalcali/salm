"""Two strictly defined experiment arms: standard LoRA SFT and Decision-State JEPA.

Both use the same task-level splits, event targets, LoRA rank, seed and token
budgets. The JEPA arm additionally injects a prefix-predicted hidden state into
future token generation; JEPA alignment is auxiliary to the token CE.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .data import DiskCorpus,verify_disjoint,collate
from .model import DecisionStateHead,DecisionStateConditioner


def defaults():
    return {
        'model':'LiquidAI/LFM3-1.7B-Base','revision':None,
        'mode':'sft','seed':42,'train':'runs/decision_state_data/train.jsonl',
        'val':'runs/decision_state_data/val.jsonl','out':'runs/decision_state/sft_8k',
        'max_seq_length':8192,'target_tokens':768,'batch_size':1,'grad_accum':8,
        'max_steps':2000,'lr':0.0001,'weight_decay':0.01,
        'lora_rank':16,'lora_alpha':32,
        'source_layer':12,'inject_layer':18,'latent_size':256,
        'predictor_width':512,'gate_init':0.05,'jepa_weight':0.2,
        'eval_every':200,'eval_tasks':100,'save_every':200,'grad_clip':1.0,
        'min_free_gib':70.0,'resume':None,
    }


def check_cfg(cfg):
    c={**defaults(),**cfg}
    if c['mode'] not in {'sft','decision_jepa'}:
        raise ValueError('Only sft and decision_jepa are supported')
    for key in ('max_seq_length','target_tokens','batch_size','grad_accum','max_steps',
                'eval_every','eval_tasks','save_every'):
        if c[key]<1:
            raise ValueError(f'{key} must be positive')
    if c['target_tokens']>=c['max_seq_length']-64:
        raise ValueError('No room for prefix')
    if c['source_layer']>=c['inject_layer']:
        raise ValueError('source layer must precede injection layer')
    if c['max_seq_length']>16384:
        raise ValueError('>16k context is unvalidated on GB10; adjust and test deliberately')
    return c


def build_model(cfg, *, lora_dir=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,TaskType,get_peft_model,PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for LFM3 GPU training')
    free,total=torch.cuda.mem_get_info()
    if free/2**30 < cfg['min_free_gib']:
        raise RuntimeError(f'CUDA free {free/2**30:.1f} GiB < requested minimum {cfg["min_free_gib"]}')
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg.get('revision'),trust_remote_code=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError('Tokenizer missing pad/eos')
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg.get('revision'),
        trust_remote_code=True,dtype=torch.bfloat16).to('cuda')
    base.config.use_cache=False
    if lora_dir:
        model=PeftModel.from_pretrained(base,str(lora_dir),is_trainable=False)
    else:
        wanted={'w1','w2','w3','q_proj','k_proj','v_proj','o_proj','in_proj','out_proj'}
        available={name.rsplit('.',1)[-1] for name,m in base.named_modules() if isinstance(m,nn.Linear)}
        targets=sorted(wanted & available)
        if not {'w1','w2','w3'}<=set(targets):
            raise RuntimeError('Expected LFM3 FFN projections not found')
        model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg['lora_rank'],
                     lora_alpha=cfg['lora_alpha'],lora_dropout=0.0,bias='none',target_modules=targets))
        print('[Decision-State] LoRA targets:',targets,'trainable',
              sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    return model,tok


def build_head(model,cfg):
    if cfg['mode']=='sft':
        return None,None
    head=DecisionStateHead(model.config.hidden_size,cfg['latent_size'],
              cfg['predictor_width'],cfg['gate_init']).to('cuda')
    return head, DecisionStateConditioner(model,head,cfg['source_layer'],cfg['inject_layer'])


def forward_batch(model,head,cond,batch,cfg,*,inject=True):
    d=next(model.parameters()).device
    batch={k:v.to(d) for k,v in batch.items()}
    if cond is None:
        out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                  labels=batch['labels'],use_cache=False)
        ce=out.loss.float()
        jepa=ce.new_zeros(())
    else:
        cond.set_end_lengths(batch['attention_mask'].sum(-1))
        with cond.active(batch['prefix_length'],enabled=inject):
            out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                      labels=batch['labels'],use_cache=False)
            ce=out.loss.float()
            jepa=cond.jepa_loss()
    if not torch.isfinite(ce) or not torch.isfinite(jepa):
        raise FloatingPointError('Nonfinite CE or JEPA loss')
    total=ce+(cfg['jepa_weight']*jepa if cond is not None and inject else 0.)
    num=int((batch['labels'][:,1:]!=-100).sum().item())
    return total,{'ce':float(ce.detach()),'jepa':float(jepa.detach()),'tokens':num,
                  'gate':float(head.gate.detach()) if head is not None else 0.0}


@torch.no_grad()
def evaluate(model,head,cond,corpus,tok,cfg,*,count=None,inject=True):
    model.eval()
    if head: head.eval()
    examples=corpus.fixed_examples(tok,count=count or cfg['eval_tasks'],
        max_length=cfg['max_seq_length'],target_tokens=cfg['target_tokens'])
    loss_total=0.; token_total=0
    for ex in examples:
        batch=collate([ex],tok.pad_token_id)
        _, metrics=forward_batch(model,head,cond,batch,cfg,inject=inject)
        loss_total+=metrics['ce']*metrics['tokens'];token_total+=metrics['tokens']
    model.train()
    if head:head.train()
    return {'nll':loss_total/max(token_total,1),'tokens':token_total,'tasks':len(examples)}


def save(out,step,model,head,optim,cfg):
    folder=out/f'checkpoint_{step:06d}'
    if folder.exists():
        raise FileExistsError('Checkpoint exists: '+str(folder))
    folder.mkdir(parents=True)
    model.save_pretrained(folder/'lora',safe_serialization=True)
    if head is not None:
        from safetensors.torch import save_file
        save_file({n:t.detach().cpu().contiguous() for n,t in head.state_dict().items()},str(folder/'decision_head.safetensors'))
    torch.save({'optimizer':optim.state_dict(),'step':step},folder/'optimizer.pt')
    (folder/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[Decision-State] checkpoint',folder,flush=True)


def restore(path,model,head,opt):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    p=Path(path)
    set_peft_model_state_dict(model,load_file(str(p/'lora'/'adapter_model.safetensors')))
    if head is not None:
        head.load_state_dict(load_file(str(p/'decision_head.safetensors')))
    state=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=True)
    opt.load_state_dict(state['optimizer'])
    return int(state['step'])


def train(user_cfg):
    cfg=check_cfg(user_cfg)
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and not cfg.get('resume'):
        raise FileExistsError('History exists. Change out or specify resume.')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    training=DiskCorpus(cfg['train']);validation=DiskCorpus(cfg['val']);verify_disjoint(training,validation)
    model,tok=build_model(cfg)
    head,cond=build_head(model,cfg)
    cfg['model_revision_resolved']=getattr(model.config,'_commit_hash',None)
    params=[p for p in model.parameters() if p.requires_grad]
    if head:params+=list(head.parameters())
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    start=restore(cfg['resume'],model,head,opt) if cfg.get('resume') else 0
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'mode':cfg['mode'],'train_tasks':len(training.index),'val_tasks':len(validation.index),
                      'max_seq_length':cfg['max_seq_length'],'batch_size':cfg['batch_size']},ensure_ascii=False),flush=True)
    t0=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
    if head:head.train()
    history=out/'history.jsonl'
    for step in range(start,cfg['max_steps']):
        totals=[]
        for micro in range(cfg['grad_accum']):
            rng=random.Random(cfg['seed']+step*10009+micro)
            examples=[training.sample(tok,rng,max_length=cfg['max_seq_length'],
                         target_tokens=cfg['target_tokens']) for _ in range(cfg['batch_size'])]
            batch=collate(examples,tok.pad_token_id)
            loss,info=forward_batch(model,head,cond,batch,cfg)
            (loss/cfg['grad_accum']).backward()
            totals.append(info)
        norm=float(clip_grad_norm_(params,cfg['grad_clip']))
        if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        opt.step();opt.zero_grad(set_to_none=True)
        done=step+1
        entry={'step':done,'mode':cfg['mode'],
               'ce':sum(r['ce']*r['tokens'] for r in totals)/max(sum(r['tokens'] for r in totals),1),
               'jepa':sum(r['jepa'] for r in totals)/len(totals),
               'gate':float(head.gate.detach()) if head else 0.0,
               'grad_norm':norm,'train_tokens':sum(r['tokens'] for r in totals),
               'seconds':time.monotonic()-t0,'cuda_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        if done%cfg['eval_every']==0 or done==cfg['max_steps']:
            val=evaluate(model,head,cond,validation,tok,cfg)
            entry.update({'val_nll':val['nll'],'val_tokens':val['tokens'],'val_tasks':val['tasks']})
            if cond:
                no=evaluate(model,head,cond,validation,tok,cfg,inject=False)
                entry['val_no_inject_nll']=no['nll']
                entry['injection_gain_nll']=no['nll']-val['nll']
        with history.open('a',encoding='utf-8') as f:
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        print(json.dumps(entry,ensure_ascii=False),flush=True)
        if done%cfg['save_every']==0 or done==cfg['max_steps']:
            save(out,done,model,head,opt,cfg)
    return {'mode':cfg['mode'],'steps':cfg['max_steps'],'out':str(out)}


def evaluate_checkpoint(path,count=100):
    p=Path(path)
    cfg=check_cfg(json.loads((p/'config.json').read_text(encoding='utf-8')))
    model,tok=build_model(cfg,lora_dir=p/'lora')
    head,cond=build_head(model,cfg)
    if head:
        from safetensors.torch import load_file
        head.load_state_dict(load_file(str(p/'decision_head.safetensors')))
    val=DiskCorpus(cfg['val'])
    on=evaluate(model,head,cond,val,tok,cfg,count=count)
    result={'checkpoint':str(p),'mode':cfg['mode'],'evaluation':on}
    if cond:
        off=evaluate(model,head,cond,val,tok,cfg,count=count,inject=False)
        result.update({'injection_off':off,'gain_nll':off['nll']-on['nll']})
    return result
