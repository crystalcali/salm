import json
from pathlib import Path
from types import SimpleNamespace
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from decision_state.data import (stable_split,normalize_messages,convert_nvidia,
    make_example,DiskCorpus,verify_disjoint,collate)
from decision_state.model import DecisionStateHead,DecisionStateConditioner
from decision_state.train import forward_batch,check_cfg


class ToyTok:
    pad_token_id = 0
    def encode(self,s,add_special_tokens=False):
        return [1+(ord(x)%128) for x in s]


class ToyBlock(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.ff=nn.Linear(n,n)
    def forward(self,x):
        # causal: all positions depend only on themselves and preceding tokens
        return x+0.01*self.ff(torch.cumsum(x,dim=1)/(1+torch.arange(x.shape[1],device=x.device)[None,:,None]))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(160,16)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([ToyBlock(16) for _ in range(5)])
        self.lm_head=nn.Linear(16,160)
        self.config=SimpleNamespace(hidden_size=16)
    def forward(self,input_ids,attention_mask,labels,use_cache=False):
        h=self.emb(input_ids)
        for layer in self.model.layers:
            h=layer(h)
        logits=self.lm_head(h)
        loss=F.cross_entropy(logits[:,:-1].contiguous().view(-1,160).float(),
                             labels[:,1:].contiguous().view(-1),ignore_index=-100)
        return SimpleNamespace(loss=loss,logits=logits)


def row():
    return {'task_id':'one','messages':[
        {'role':'user','kind':'user','text':'compute the new value using a tool now'},
        {'role':'assistant','kind':'tool_call','text':'call python run(2+2)'},
        {'role':'tool','kind':'tool','text':'4'},
        {'role':'user','kind':'user','text':'please explain the result'},
        {'role':'assistant','kind':'reasoning','text':'the output equals four'},
        {'role':'tool','kind':'tool','text':'FUTURE_SECRET'},
    ]}


def test_split_deterministic_and_grouped():
    assert stable_split('same')==stable_split('same')
    assert stable_split('same') in {'train','val'}


def test_convert_nvidia_keeps_only_success_and_no_gold():
    raw={'instance_id':'repo/issue','resolved':1,'trajectory_id':'abc',
        'metadata':{'reference_patch':{'patch':'LEAK_REFERENCE_PATCH'}},
        'messages':[
            {'role':'user','content':'fix crash'},
            {'role':'assistant','content':'I will test','tool_calls':[{'name':'bash','args':{'command':'pytest'}}]},
            {'role':'tool','content':'fail'},
            {'role':'assistant','content':'here is patch'},
        ]}
    out=convert_nvidia(raw)
    assert out['verification']['task_resolved'] is True
    assert 'LEAK_REFERENCE_PATCH' not in json.dumps(out)
    assert any(m['kind']=='tool_call' for m in out['messages'])
    raw['resolved']=0
    assert convert_nvidia(raw) is None


def test_future_tool_and_user_never_target_or_prefix():
    r=row(); tok=ToyTok()
    ex=make_example(r,4,tok,max_length=8192,target_tokens=120)
    text=''.join(map(chr, []))
    assert 'FUTURE_SECRET' not in json.dumps(ex.input_ids)
    assert ex.prefix_length>10
    assert ex.labels[:ex.prefix_length]==[-100]*ex.prefix_length
    assert ex.labels[ex.prefix_length:]==ex.input_ids[ex.prefix_length:]
    r2=row(); r2['messages'][5]['text']='DIFFERENT FUTURE OBSERVATION'
    e2=make_example(r2,4,tok,max_length=8192,target_tokens=120)
    assert e2.input_ids==ex.input_ids
    with pytest.raises(ValueError):
        make_example(r,5,tok,max_length=8192,target_tokens=120)


def test_long_context_keeps_more_than_4096_tokens():
    tok=ToyTok(); r=row()
    r['messages'][0]['text']='a'*9000
    e=make_example(r,4,tok,max_length=8192,target_tokens=128)
    assert 4096<len(e.input_ids)<=8192
    assert len(e.input_ids)-e.prefix_length<=128


def test_disk_corpus_duplicate_and_leakage(tmp_path):
    r=row()
    train=tmp_path/'train.jsonl';val=tmp_path/'val.jsonl'
    train.write_text(json.dumps(r)+'\n')
    val.write_text(json.dumps(r)+'\n')
    a=DiskCorpus(str(train));b=DiskCorpus(str(val))
    with pytest.raises(ValueError):verify_disjoint(a,b)
    r['task_id']='two';val.write_text(json.dumps(r)+'\n')
    b=DiskCorpus(str(val));verify_disjoint(a,b)
    ex=a.sample(ToyTok(),random.Random(0),max_length=256,target_tokens=48,min_context_tokens=1)
    assert ex.task_id=='one'


def test_architecture_ce_updates_predictor_without_jepa_loss():
    torch.manual_seed(1)
    model=ToyLM();head=DecisionStateHead(16,latent_size=8,width=16,gate_init=.05)
    cond=DecisionStateConditioner(model,head,source_layer=1,inject_layer=3)
    r=row();ex=make_example(r,4,ToyTok(),max_length=256,target_tokens=50)
    batch=collate([ex],ToyTok.pad_token_id)
    loss,metrics=forward_batch(model,head,cond,batch,{'jepa_weight':0.0})
    loss.backward()
    assert head.predictor[-1].weight.grad is not None
    assert head.predictor[-1].weight.grad.abs().sum()>0
    assert head.gate.grad is not None and head.gate.grad.abs().sum()>0
    assert model.emb.weight.grad is not None and model.emb.weight.grad.abs().sum()>0
    assert metrics['tokens']>0


def test_predicted_state_is_future_independent():
    torch.manual_seed(2)
    model=ToyLM();head=DecisionStateHead(16,latent_size=8,width=16)
    cond=DecisionStateConditioner(model,head,source_layer=1,inject_layer=3)
    e=make_example(row(),4,ToyTok(),max_length=256,target_tokens=50)
    a=collate([e],0)
    with cond.active(a['prefix_length']) as _:
        cond.set_end_lengths(a['attention_mask'].sum(-1))
        model(input_ids=a['input_ids'],attention_mask=a['attention_mask'],labels=a['labels'])
        z1=cond.predicted.detach().clone()
    b={k:v.clone() for k,v in a.items()}
    b['input_ids'][:,e.prefix_length:]=2
    with cond.active(b['prefix_length']) as _:
        cond.set_end_lengths(b['attention_mask'].sum(-1))
        model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],labels=b['labels'])
        z2=cond.predicted.detach().clone()
    assert torch.allclose(z1,z2,atol=1e-6)


def test_baseline_and_jepa_config():
    assert check_cfg({'mode':'sft','max_seq_length':8192})['mode']=='sft'
    assert check_cfg({'mode':'decision_jepa','max_seq_length':12288})['mode']=='decision_jepa'
    with pytest.raises(ValueError):check_cfg({'mode':'third'})


def test_source_content_serialization_and_tool_target():
    m=normalize_messages([{'role':'ai','content':'run','tool_calls':[{'name':'python'}]},
                          {'role':'tool','content':{'result':'3'}}])
    assert m[0]['kind']=='tool_call' and m[1]['role']=='tool'


def test_higher_sampling_weight_on_tool_decisions():
    from decision_state.data import decision_importance
    events = {'messages':[
        {'role':'user','kind':'user','text':'task'},
        {'role':'assistant','kind':'reasoning','text':'thinking'},
        {'role':'assistant','kind':'tool_call','text':'bash'},
        {'role':'tool','kind':'tool','text':'out'},
        {'role':'assistant','kind':'reasoning','text':'inspect output'},
    ]}
    assert decision_importance(events,2) > decision_importance(events,1)
    assert decision_importance(events,4) > decision_importance(events,1)


def test_terminal_bench_training_exclusion():
    raw={'instance_id':'terminal-bench/task_001','repo':'other/repo',
         'resolved':1,'messages':[
             {'role':'user','content':'fix bug'},
             {'role':'assistant','content':'act','tool_calls':[{'name':'Bash','args':{'command':'echo 1'}}]},
             {'role':'tool','content':'1'},
             {'role':'assistant','content':'done'},
         ]}
    assert convert_nvidia(raw) is None
    raw['instance_id']='repo/issue'
    raw['messages'][0]['content']='harbor-canary GUID X'
    assert convert_nvidia(raw) is None


def test_cached_generation_state_uses_prefix_once():
    from decision_state.model import DecisionGenerationConditioner
    torch.manual_seed(5)
    model=ToyLM();head=DecisionStateHead(16,latent_size=8,width=16)
    cond=DecisionStateConditioner(model,head,source_layer=1,inject_layer=3)
    prompt=torch.tensor([[2,3,4,5]])
    # Simulated cached decoding: a long prefill then a single new token.
    with DecisionGenerationConditioner(cond) as generation:
        model(prompt, torch.ones_like(prompt), torch.ones_like(prompt))
        prefix_state=generation.state.detach().clone()
        model(torch.tensor([[99]]),torch.ones(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.long))
        assert torch.allclose(generation.state,prefix_state)
    assert generation.state is None
    assert len(model.model.layers[1]._forward_hooks)==0
    assert len(model.model.layers[3]._forward_hooks)==0


def test_harbor_tool_parser_and_command_mapping():
    from decision_state.terminal_protocol import parse_tool_call,command_for_tool,forbidden_command
    text='<assistant kind=tool_call>\n<tool_call>{"name":"Bash","args":{"command":"pwd && ls"}}</tool_call>\n</assistant>'
    call=parse_tool_call(text)
    assert call=={'name':'Bash','args':{'command':'pwd && ls'}}
    assert command_for_tool(call)=='pwd && ls'
    claude='Tool calls: [{"type":"function","function":{"name":"Bash","arguments":"{\\"command\\":\\"pytest -q\\"}"}}]'
    assert parse_tool_call(claude)['args']['command']=='pytest -q'
    assert forbidden_command('cat /logs/verifier/reward.txt')
    assert not forbidden_command('python -m pytest tests/test_something.py')


def test_harbor_reproducible_pair_and_no_benchmark_train(tmp_path):
    from decision_state.benchmark import command_for_benchmark,manifest
    c=command_for_benchmark(limit=5,trials=1)
    assert '--agent-import-path' in c and '-l' in c
    assert 'terminal-bench/terminal-bench-2-1' in c
    assert '--n-concurrent' in c
    assert '--repo' in command_for_benchmark(limit=None,trials=3,use_hf_repo=True)
    ck=tmp_path/'checkpoint';(ck/'lora').mkdir(parents=True)
    (ck/'config.json').write_text(json.dumps({'mode':'sft','max_seq_length':8192,'train':'NVIDIA'}))
    (ck/'lora'/'adapter_model.safetensors').write_bytes(b'weight_bytes_for_test')
    m=manifest(ck)
    assert m['mode']=='sft' and m['training_context']==8192 and len(m['weights_sha256'])==1
