"""Optional sliced SIGReg loss. Not a replacement for token CE."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class SIGReg(nn.Module):
    def __init__(self, projections: int = 64, knots: int = 13, tmax: float = 3.0):
        super().__init__()
        self.projections = projections
        t = torch.linspace(0, tmax, knots)
        dt = tmax / (knots - 1)
        weight = 2 * dt * torch.ones_like(t)
        weight[[0, -1]] = dt
        phi = torch.exp(-t.square()/2)
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', weight*phi)

    def forward(self, z: torch.Tensor):
        # One independent representation per task. Never use S memory slots
        # from the same task as if they were independent samples.
        if z.ndim != 2 or z.shape[0] < 2:
            raise ValueError('SIGReg needs 2D embeddings with >=2 distinct tasks')
        a = F.normalize(torch.randn(z.shape[-1], self.projections,
                                    device=z.device, dtype=torch.float32), dim=0)
        x = z.float() @ a
        angles = x.unsqueeze(-1) * self.t
        real, imag = angles.cos().mean(0), angles.sin().mean(0)
        err = (real-self.phi).square()+imag.square()
        return ((err@self.weights)*len(z)).mean()
