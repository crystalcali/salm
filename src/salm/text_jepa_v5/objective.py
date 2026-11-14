import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalSIGReg(nn.Module):
    def __init__(self, knots=17, directions=128):
        super().__init__()
        times = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("times", times)
        self.register_buffer("phi", torch.exp(-.5 * times.square()))
        self.register_buffer("weights", weights * self.phi)
        self.directions = directions

    def forward(self, z):
        b, _, d = z.shape
        assert b >= 2
        directions = F.normalize(torch.randn(d, self.directions, device=z.device), dim=0)
        projections = torch.einsum("btd,dk->tbk", z.float(), directions)
        phase = projections.unsqueeze(-1) * self.times
        err = (phase.cos().mean(1) - self.phi).square() + phase.sin().mean(1).square()
        return (err @ self.weights).mean() * b


def objective(pred, target, regularizer, weight):
    mse = F.mse_loss(pred.float(), target.float())
    reg = .5 * (regularizer(pred) + regularizer(target))
    return (1 - weight) * mse + weight * reg, mse, reg


def entropy_rank(centered):
    spectrum = torch.linalg.svdvals(centered.float()).square()
    total = spectrum.sum(-1, keepdim=True)
    p = spectrum / total.clamp_min(1e-20)
    rank = torch.exp(-(p * p.clamp_min(1e-20).log()).sum(-1))
    return torch.where(total.squeeze(-1) > 1e-12, rank, torch.zeros_like(rank))


@torch.no_grad()
def diagnostics(pred, target, positions=8):
    result = {}
    for name, z in (("pred", pred.float()), ("target", target.float())):
        # Remove each position's cross-document mean BEFORE pooling.
        residual = z - z.mean(0, keepdim=True)
        ix = torch.linspace(0, z.shape[1] - 1, min(positions, z.shape[1]),
                            device=z.device).long()
        ranks = entropy_rank(residual[:, ix].transpose(0, 1))
        result[name + "_rank_position"] = float(ranks.mean())
        result[name + "_rank_residual"] = float(entropy_rank(residual.reshape(-1, z.shape[-1])))
        result[name + "_content_std"] = float(residual.square().mean().sqrt())
    result["rank_position_limit"] = min(len(target) - 1, target.shape[-1])
    return result


@dataclass
class SmoothController:
    weight: float
    ema_rank_fraction: float | None = None
    updates: int = 0

    def update(self, cfg, metrics, tokens_seen):
        if tokens_seen < cfg.controller_warmup_tokens:
            return self.weight
        value = min(metrics["pred_rank_position"], metrics["target_rank_position"])
        value /= metrics["rank_position_limit"]
        if not math.isfinite(value):
            raise ValueError("Non-finite controller diagnostic")
        self.ema_rank_fraction = value if self.ema_rank_fraction is None else (
            cfg.controller_ema * self.ema_rank_fraction + (1 - cfg.controller_ema) * value)
        deficit = (cfg.rank_target_fraction - self.ema_rank_fraction) / cfg.rank_target_fraction
        if deficit > 0:
            delta = cfg.controller_gain * deficit
        else:
            delta = .1 * (cfg.sigreg_weight - self.weight)
        delta = max(-cfg.controller_max_delta, min(cfg.controller_max_delta, delta))
        self.weight = max(cfg.sigreg_min, min(cfg.sigreg_max, self.weight + delta))
        self.updates += 1
        return self.weight

    def state_dict(self):
        return asdict(self)
