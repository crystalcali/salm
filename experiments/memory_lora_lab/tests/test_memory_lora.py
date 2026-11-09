import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from memory_lora.data import CausalTraceCorpus, load_rows, require_disjoint, collate
from memory_lora.model import LatentMemoryAdapter, MemoryConditioner, chronological_slots
from memory_lora.runner import token_loss_batch, validate_config
from memory_lora.sigreg import SIGReg


class TinyTokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 99 + 1 for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr((i-1) % 99 + 32) for i in ids)


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)
    def forward(self, hidden):
        return hidden + torch.tanh(self.proj(hidden))


class TinyLM(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.embed = nn.Embedding(101, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(width) for _ in range(3)])
        self.lm_head = nn.Linear(width, 101)
        self.config = SimpleNamespace(hidden_size=width, use_cache=False)
    def get_input_embeddings(self):
        return self.embed
    def forward(self, input_ids, attention_mask=None, labels=None,
                use_cache=False, logits_to_keep=0):
        hidden = self.embed(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        loss = (F.cross_entropy(logits[:, :-1].reshape(-1,101),labels[:, 1:].reshape(-1),ignore_index=-100)
                if labels is not None else None)
        return SimpleNamespace(loss=loss, logits=logits)


def test_chronological_pooling_ignores_padding():
    h = torch.tensor([[[1.], [3.], [5.], [100.]]])
    mask = torch.tensor([[1, 1, 1, 0]])
    slots = chronological_slots(h, mask, slots=2)
    assert slots.shape == (1,2,1)
    assert torch.allclose(slots.flatten(),torch.tensor([1.,4.]))


def test_math_next_step_only_no_future_answer():
    t = TinyTokenizer()
    row = {'task_id':'math1','prompt':'Q'*160,
           'steps':['reason one', 'reason two', 'reason three'],
           'final_answer':'99'}
    corpus=CausalTraceCorpus([row],t, memory_tokens=80,recent_tokens=70,
                             target_tokens=100,min_memory_tokens=8)
    assert len(corpus.candidates)>=1
    examples=corpus.sample(random.Random(4),1)
    ex=examples[0]
    assert ex.past_ids and ex.recent_ids and ex.target_ids
    batch=collate(examples,0)
    assert batch['labels'].shape == batch['current_ids'].shape
    assert (batch['labels'][0,:len(ex.recent_ids)]==-100).all()
    assert torch.equal(batch['labels'][0,len(ex.recent_ids):len(ex.recent_ids)+len(ex.target_ids)],
                       torch.tensor(ex.target_ids))


def test_agent_tool_observation_in_past_not_future():
    t=TinyTokenizer()
    row={'task_id':'agent1','prompt':'Fix a bug in project '+('x'*120),
         'messages':[
            {'role':'assistant','content':'Need to run tests'},
            {'role':'assistant','content':'','tool_calls':[{'name':'pytest','arguments':'{}'}]},
            {'role':'tool','content':'SPECIAL_TOOL_RESULT: failing test parser'},
            {'role':'assistant','content':'Inspect parser now'},
            {'role':'tool','content':'FUTURE_SECRET_RESULT'},
            {'role':'assistant','content':'Prepare final answer'},
         ]}
    corpus=CausalTraceCorpus([row],t,memory_tokens=1000,recent_tokens=90,
                             target_tokens=100,min_memory_tokens=4)
    assistant_after_tool = [i for i,c in enumerate(corpus.candidates) if c[4]==4]
    assert assistant_after_tool
    j=assistant_after_tool[0]
    tid, old, recent, kind, boundary=corpus.candidates[j]
    visible=list(old)+list(recent)
    past_text=''.join(chr((i-1)%99+32) for i in visible)
    # Validate causality at the source segment level too: prior tool result
    # must be in history, future result must not be.
    from memory_lora.data import _events
    segments, flags, kinds=_events(row)
    assert 'SPECIAL_TOOL_RESULT' in ''.join(segments[:boundary])
    assert 'FUTURE_SECRET_RESULT' not in ''.join(segments[:boundary])
    assert 'Inspect parser now' in segments[boundary]
    assert kinds[boundary]=='assistant'


def test_gradient_from_token_ce_reaches_memory():
    torch.manual_seed(2)
    model=TinyLM(width=16)
    adapter=LatentMemoryAdapter(16,latent_size=8,slots=3,heads=2,gate_init=0.2)
    cond=MemoryConditioner(model,adapter,source_layer=0,inject_layer=1)
    batch={
        'memory_ids':torch.tensor([[1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1]]),
        'memory_mask':torch.ones(2,8,dtype=torch.long),
        'current_ids':torch.tensor([[10,11,12,13,14,15,16],[20,21,22,23,24,25,26]]),
        'current_mask':torch.ones(2,7,dtype=torch.long),
        'labels':torch.tensor([[-100,-100,-100,13,14,15,16],[-100,-100,-100,23,24,25,26]]),
    }
    cfg={'mode':'memory','sigreg_weight':0.0}
    loss, report=token_loss_batch(model,cond,adapter,batch,cfg)
    loss.backward()
    assert adapter.gate.grad is not None and adapter.gate.grad.abs().item()>0
    assert adapter.mem_proj.weight.grad is not None
    assert adapter.mem_proj.weight.grad.abs().sum().item()>0
    assert adapter.query_proj.weight.grad.abs().sum().item()>0
    assert model.model.layers[2].proj.weight.grad.abs().sum().item()>0
    assert not model.model.layers[1]._forward_hooks
    assert not model.model.layers[0]._forward_hooks
    assert report['tokens']==8


def test_real_vs_zero_and_shuffled_controls():
    torch.manual_seed(4)
    m=TinyLM(width=16)
    a=LatentMemoryAdapter(16,latent_size=8,slots=3,heads=2,gate_init=1.0)
    c=MemoryConditioner(m,a,0,1)
    mem=c.encode(torch.tensor([[1,2,3],[4,5,6]]),torch.ones(2,3,dtype=torch.long))
    h=torch.randn(2,5,16)
    assert torch.allclose(a(h,mem,memory_mode='zero'),h)
    assert not torch.allclose(a(h,mem,memory_mode='real'),a(h,mem,memory_mode='shuffle'))


def test_sigreg_is_not_token_loss():
    torch.manual_seed(1)
    reg=SIGReg(projections=12,knots=7)
    x=torch.randn(10,8,requires_grad=True)
    y=reg(x)
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.warns(UserWarning):
        validate_config({'mode':'memory_sigreg','batch_size':2,'sigreg_weight':0.01})


def test_split_detects_prompt_duplicates():
    a={'task_id':'x','prompt':'hello','steps':['a']}
    b={'task_id':'y','prompt':'hello','steps':['b']}
    with pytest.raises(ValueError,match='Repeated prompt'):
        require_disjoint([a],[b])
