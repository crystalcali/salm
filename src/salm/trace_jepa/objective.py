"""Shared-weight predictive latent objective, SIGReg, no EMA, no stopgrad."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .sigreg import SIGReg


@dataclass
class ObjectiveOutput:
    total: torch.Tensor
    prediction: torch.Tensor
    sigreg_online: torch.Tensor
    sigreg_target: torch.Tensor
    online_variance: torch.Tensor
    target_variance: torch.Tensor


class TraceJEPAObjective(nn.Module):
    """Predict suffix representation from causally visible prefix representation.

    Both views use the SAME projector, and BOTH receive gradients. No EMA
    teacher, no stop-gradient. The target hidden state must come from a
    separate forward on suffix-only token IDs, never from the full-trace
    forward (that would encode the answer from visible context).
    """
    def __init__(self, hidden_size: int, latent_size: int = 256,
                 predictor_width: int = 512, projections: int = 128,
                 knots: int = 17, sigreg_weight: float = 0.02):
        super().__init__()
        if latent_size < 2 or predictor_width < 2 or sigreg_weight <= 0:
            raise ValueError("Dimensions must be >=2 and SIGReg weight positive")
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_size),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, predictor_width),
            nn.GELU(),
            nn.Linear(predictor_width, latent_size),
        )
        self.regularizer = SIGReg(projections, knots)
        self.sigreg_weight = sigreg_weight

    def forward(self, online_h: torch.Tensor, target_h: torch.Tensor) -> ObjectiveOutput:
        if online_h.shape != target_h.shape or online_h.ndim != 2:
            raise ValueError("Both hidden states must be [batch, hidden] with equal shape")
        z_online = self.projector(online_h.float())
        z_target = self.projector(target_h.float())  # not detached
        prediction = self.predictor(z_online)
        prediction_loss = F.mse_loss(prediction, z_target)
        sigreg_online = self.regularizer(z_online)
        sigreg_target = self.regularizer(z_target)
        regularizer_loss = 0.5 * (sigreg_online + sigreg_target)
        total = prediction_loss + self.sigreg_weight * regularizer_loss
        # Variance along the batch dimension. For very small batches this is
        # only a coarse collapse warning, not a proof of representation quality.
        return ObjectiveOutput(
            total=total,
            prediction=prediction_loss,
            sigreg_online=sigreg_online,
            sigreg_target=sigreg_target,
            online_variance=z_online.float().var(dim=0, unbiased=False).mean().detach(),
            target_variance=z_target.float().var(dim=0, unbiased=False).mean().detach(),
        )
