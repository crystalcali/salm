"""Causal single-forward Decision-State JEPA injected into LFM3 token generation.

Source state is ALWAYS extracted at the last *prefix* token at an earlier
causal decoder block. Target state for the auxiliary loss is extracted at the
end of the supervised assistant event; it is detached and NEVER used to
compute logits. The predictor and residual injection are trained by token CE.
No EMA or independent teacher forward is required.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


def decoder_layers(model):
    core = model.get_base_model() if hasattr(model,'get_base_model') else model
    if hasattr(core,'model') and hasattr(core.model,'layers'):
        return core.model.layers
    if hasattr(core,'layers'):
        return core.layers
    raise RuntimeError('LFM3 decoder layers not found')


def unpack_hidden(output):
    if isinstance(output,(tuple,list)):
        return output[0]
    return output


def replace_hidden(output, h):
    if isinstance(output,tuple):
        return (h,)+output[1:]
    if isinstance(output,list):
        return [h]+output[1:]
    return h


class DecisionStateHead(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int = 256, width: int = 512,
                 gate_init: float = 0.05):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, latent_size))
        self.predictor = nn.Sequential(nn.LayerNorm(latent_size), nn.Linear(latent_size,width),
                                       nn.GELU(),nn.Linear(width,latent_size))
        self.query = nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,latent_size))
        self.key = nn.Linear(latent_size,latent_size,bias=False)
        self.value = nn.Linear(latent_size,hidden_size,bias=False)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(gate_init,dtype=torch.float32))

    def predict(self, prefix: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        z = self.project(prefix.float())
        predicted = z + self.predictor(z)
        return z, predicted

    def apply(self, hidden: torch.Tensor, state: torch.Tensor, prefix_len: torch.Tensor,
              *, enabled: bool) -> torch.Tensor:
        if not enabled:
            return hidden
        # Single predicted state conditions the entire answer, including the
        # logit at the last prefix position that predicts the FIRST target token.
        b,t,d = hidden.shape
        q = self.query(hidden.float())
        k = self.key(state.float()).unsqueeze(1)
        relevance = (q*k).sum(-1,keepdim=True)/max(q.shape[-1]**0.5,1.0)
        delta = self.value(state.float()).unsqueeze(1) * torch.sigmoid(relevance)
        mask = torch.arange(t,device=hidden.device).unsqueeze(0) >= (prefix_len-1).unsqueeze(1)
        delta = self.out_norm(delta).to(hidden.dtype) * mask.unsqueeze(-1)
        return hidden + (self.gate.to(hidden.dtype)*delta)


class DecisionStateConditioner:
    """Per-forward hooks with strict source-before-injection and no future leakage."""
    def __init__(self, model, head: DecisionStateHead, source_layer: int = 12,
                 inject_layer: int = 18):
        self.model = model
        self.head = head
        self.layers = decoder_layers(model)
        if not 0 <= source_layer < inject_layer < len(self.layers):
            raise ValueError('Require 0 <= source_layer < inject_layer < decoder_layers')
        self.source_layer=source_layer
        self.inject_layer=inject_layer
        self.predicted=None
        self.target=None
        self.prefix=None

    @contextmanager
    def active(self, prefix_length: torch.Tensor, *, enabled: bool = True):
        self.predicted=self.target=self.prefix=None
        def at_source(module,args,output):
            hidden=unpack_hidden(output)
            B,T,H=hidden.shape
            if prefix_length.shape!=(B,):
                raise ValueError('prefix_length shape mismatch')
            pos=prefix_length.to(hidden.device)-1
            if torch.any(pos<0) or torch.any(pos>=T):
                raise ValueError('Invalid prefix bounds')
            ids=torch.arange(B,device=hidden.device)
            prefix_hidden=hidden[ids,pos]
            self.prefix,self.predicted=self.head.predict(prefix_hidden)
            # True target is computed from teacher-forced tokens but detached.
            # It is only for latent supervision, never fed into the generator.
            # True end index comes from final valid attention positions. Since
            # batches can be padded, the caller supplies end lengths below.
            if self._end_lengths is not None:
                ends=self._end_lengths.to(hidden.device)-1
                future_hidden=hidden[ids,ends]
                with torch.no_grad():
                    self.target=self.head.project(future_hidden.float()).detach()
        def at_inject(module,args,output):
            if self.predicted is None:
                raise RuntimeError('Source hook did not run')
            h=unpack_hidden(output)
            return replace_hidden(output,self.head.apply(h,self.predicted,
                                 prefix_length.to(h.device),enabled=enabled))
        h1=self.layers[self.source_layer].register_forward_hook(at_source)
        h2=self.layers[self.inject_layer].register_forward_hook(at_inject)
        try:
            yield self
        finally:
            h2.remove(); h1.remove()
            self._end_lengths=None

    _end_lengths=None

    def set_end_lengths(self, lengths: torch.Tensor):
        self._end_lengths=lengths

    def jepa_loss(self):
        if self.target is None or self.predicted is None:
            raise RuntimeError('No target state captured')
        # Normalized alignment avoids latent scale-only minimization.
        z=F.layer_norm(self.predicted.float(),(self.predicted.shape[-1],))
        y=F.layer_norm(self.target.float(),(self.target.shape[-1],))
        return F.smooth_l1_loss(z,y)


class DecisionGenerationConditioner:
    """Persistent causal decision state across autoregressive decoding steps.

    The source state is computed on the FIRST full-prefix forward only. It is
    reused on cached next-token forwards, never re-estimated from generated
    future tokens. This matches a single prediction per decision boundary.
    """
    def __init__(self, conditioner: DecisionStateConditioner):
        self.cond = conditioner
        self.state = None
        self.first_forward = True
        self._handles = []

    def __enter__(self):
        source = self.cond.layers[self.cond.source_layer]
        inject = self.cond.layers[self.cond.inject_layer]

        def on_source(_module, _args, output):
            if self.state is None:
                hidden = unpack_hidden(output)
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError('Generation requires batch=1')
                _, self.state = self.cond.head.predict(hidden[:, -1, :])

        def on_inject(_module, _args, output):
            if self.state is None:
                raise RuntimeError('No decision state before injection')
            h = unpack_hidden(output)
            if self.first_forward:
                mask_from = h.shape[1] - 1
                self.first_forward = False
            else:
                mask_from = 0
            # The causal prefix state conditions only the last prefill token
            # and every subsequently decoded token.
            b, t, _ = h.shape
            q = self.cond.head.query(h.float())
            k = self.cond.head.key(self.state.float()).unsqueeze(1)
            relevance = (q*k).sum(-1,keepdim=True) / max(q.shape[-1]**0.5, 1.0)
            delta = self.cond.head.value(self.state.float()).unsqueeze(1) * torch.sigmoid(relevance)
            mask = (torch.arange(t, device=h.device) >= mask_from).view(1,t,1)
            delta = self.cond.head.out_norm(delta).to(h.dtype) * mask
            adjusted = h + self.cond.head.gate.to(h.dtype) * delta
            return replace_hidden(output, adjusted)

        self._handles = [source.register_forward_hook(on_source),
                         inject.register_forward_hook(on_inject)]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for h in reversed(self._handles):
            h.remove()
        self._handles.clear()
        self.state = None
        return False
