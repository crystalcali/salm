"""Active latent memory conditioning for a causal language model.

No JEPA-prediction auxiliary is needed for the primary experiment. The memory
reader and compressor receive gradients from the SAME token CE as LoRA.
Past contextual states are computed without gradient through the (frozen)
backbone to minimize DGX Spark unified-memory usage.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Callable

import torch
from torch import nn


def decoder_layers(model) -> nn.ModuleList:
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if hasattr(base, 'model') and hasattr(base.model, 'layers'):
        return base.model.layers
    if hasattr(base, 'layers'):
        return base.layers
    raise RuntimeError('Could not identify LFM3 decoder .model.layers')


def _hidden(output: torch.Tensor | tuple) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    if isinstance(output, list):
        return [new_hidden] + output[1:]
    if isinstance(output, torch.Tensor):
        return new_hidden
    raise TypeError(f'Unsupported LFM3 layer output type: {type(output).__name__}')


def chronological_slots(hidden: torch.Tensor, valid_mask: torch.Tensor, slots: int):
    """Temporal chunk mean of contextualized history, no future information.

    Each task is treated independently: no mixing across batch elements.
    Oldest history -> first slot; newest history -> last slot.
    """
    if hidden.ndim != 3 or valid_mask.shape != hidden.shape[:2]:
        raise ValueError('Expected hidden [B,T,H] and attention mask [B,T]')
    rows = []
    for b in range(hidden.shape[0]):
        n = int(valid_mask[b].sum().item())
        if n < 1:
            raise ValueError('Empty history in memory encoder')
        # Left/right padding independent; select by mask to avoid reading padding.
        good = hidden[b][valid_mask[b].bool()]
        chunks = []
        for i in range(slots):
            a = i * n // slots
            e = (i + 1) * n // slots
            if e > a:
                chunks.append(good[a:e].float().mean(dim=0))
            else:
                chunks.append(torch.zeros_like(good[0], dtype=torch.float32))
        rows.append(torch.stack(chunks, dim=0))
    return torch.stack(rows, dim=0)


class LatentMemoryAdapter(nn.Module):
    """Cross-attend to memory slots and inject into the LFM3 residual stream."""
    def __init__(self, hidden_size: int, latent_size: int = 256, slots: int = 16,
                 heads: int = 4, gate_init: float = 0.05):
        super().__init__()
        if latent_size % heads:
            raise ValueError('latent_size must divide heads')
        if slots < 1:
            raise ValueError('slots >= 1 required')
        self.slots = slots
        self.mem_norm = nn.LayerNorm(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, latent_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, latent_size)
        self.attn = nn.MultiheadAttention(latent_size, heads, batch_first=True)
        self.out_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def compress(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = chronological_slots(hidden, attention_mask, self.slots)
        return self.mem_proj(self.mem_norm(pooled.float()))

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor | None,
                *, memory_mode: str = 'real') -> torch.Tensor:
        if memory is None or memory_mode == 'zero':
            return hidden
        if memory_mode == 'shuffle':
            if memory.shape[0] < 2:
                raise ValueError('Shuffled-memory control requires >=2 tasks')
            memory = torch.roll(memory, shifts=1, dims=0)
        elif memory_mode != 'real':
            raise ValueError(f'Unknown memory mode: {memory_mode}')
        queries = self.query_proj(self.query_norm(hidden.float()))
        projected, _ = self.attn(queries, memory.float(), memory.float(), need_weights=False)
        delta = self.out_proj(projected)
        return hidden + (self.gate * delta).to(hidden.dtype)


class MemoryConditioner:
    """Captures fixed past representations and activates residual injection.

    The forward hook is registered only for the duration of current-context
    execution, and NEVER for the past encoder pass. Both views are chronological.
    """
    def __init__(self, model: nn.Module, adapter: LatentMemoryAdapter,
                 source_layer: int = 7, inject_layer: int = 16,
                 encoder_mode: str = 'backbone'):
        layers = decoder_layers(model)
        if not (0 <= source_layer < len(layers) and 0 <= inject_layer < len(layers)):
            raise ValueError('Invalid source/inject layer index')
        if source_layer >= inject_layer:
            raise ValueError('Require source_layer < inject_layer')
        if encoder_mode not in {'backbone', 'embeddings'}:
            raise ValueError('encoder_mode must be backbone or embeddings')
        self.model = model
        self.adapter = adapter
        self.source_layer = source_layer
        self.inject_layer = inject_layer
        self.encoder_mode = encoder_mode

    def encode(self, mem_ids: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        # The past encoder is a no-grad reading of LFM3 hidden states.
        # Projection/compression is trainable and is applied AFTER no_grad.
        if self.encoder_mode == 'embeddings':
            with torch.no_grad():
                features = self.model.get_input_embeddings()(mem_ids).detach()
            return self.adapter.compress(features, mem_mask)

        capture = []
        layer = decoder_layers(self.model)[self.source_layer]
        def capture_hidden(module, args, output):
            capture.append(_hidden(output).detach())
        handle = layer.register_forward_hook(capture_hidden)
        was_training = self.model.training
        self.model.eval()
        # Prefer frozen backbone representations rather than a changing LoRA
        # target encoder. disable_adapter is a context manager on PeftModel.
        disable = self.model.disable_adapter() if hasattr(self.model, 'disable_adapter') else nullcontext()
        try:
            with torch.no_grad(), disable:
                self.model(input_ids=mem_ids, attention_mask=mem_mask,
                           use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
            self.model.train(was_training)
        if len(capture) != 1:
            raise RuntimeError(f'Expected 1 LFM3 source hook call; got {len(capture)}')
        return self.adapter.compress(capture[0], mem_mask)

    @contextmanager
    def inject(self, memory: torch.Tensor | None, mode: str = 'real'):
        if memory is None or mode == 'zero':
            yield
            return
        layer = decoder_layers(self.model)[self.inject_layer]
        def inject_hook(module, args, output):
            h = _hidden(output)
            return _replace_hidden(output, self.adapter(h, memory, memory_mode=mode))
        handle = layer.register_forward_hook(inject_hook)
        try:
            yield
        finally:
            handle.remove()
