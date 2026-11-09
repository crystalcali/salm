"""Sketched Isotropic Gaussian Regularization (SIGReg).

Sliced Epps-Pulley statistic adapted from the published LeJEPA minimal
implementation: compare projected empirical characteristic functions against
those of N(0,1). Unlike moment-only regularization, this tests distribution
shape on random 1D projections. All math executes in float32.
Reference: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    def __init__(self, projections: int = 128, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        if projections < 1 or knots < 2 or t_max <= 0:
            raise ValueError("Invalid SIGReg parameters")
        self.projections = projections
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        # The reference code integrates the positive half-line and doubles it.
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("SIGReg expects [independent examples, latent dimensions]")
        if z.shape[0] < 2:
            raise ValueError("SIGReg needs at least 2 independent samples")
        a = F.normalize(
            torch.randn(z.shape[-1], self.projections, device=z.device, dtype=torch.float32),
            dim=0,
        )
        u = z.float() @ a  # [N,P]
        args = u.unsqueeze(-1) * self.t  # [N,P,K]
        # Epps-Pulley statistic: compare real and imaginary parts of E exp(i t X).
        real = args.cos().mean(dim=0)
        imag = args.sin().mean(dim=0)
        discrepancy = (real - self.phi).square() + imag.square()
        return ((discrepancy @ self.weights) * z.shape[0]).mean()
