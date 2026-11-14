import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    arm: str = "fixed"
    seed: int = 42
    preset: str = "base"
    seq_len: int = 1024
    vocab_size: int = 50_000
    proj_dim: int = 256
    global_batch: int = 128
    micro_batch: int = 8
    span_frac: float = .25
    suffix_probability: float = .5
    token_budget: int = 100_000_000
    lr: float = 5e-5
    weight_decay: float = .05
    warmup_tokens: int = 5_000_000
    sigreg_weight: float = .25
    sigreg_min: float = .20
    sigreg_max: float = .45
    sigreg_directions: int = 128
    sigreg_knots: int = 17
    rank_target_fraction: float = .20
    controller_gain: float = .02
    controller_max_delta: float = .005
    controller_ema: float = .9
    controller_warmup_tokens: int = 5_000_000
    diagnostic_every: int = 25
    rank_positions: int = 8
    eval_every: int = 50
    val_documents: int = 128
    save_every: int = 100
    max_grad_norm: float = 1.

    @property
    def dimensions(self):
        return {"base": (12, 768, 12), "large": (19, 1024, 16),
                "tiny": (2, 32, 4)}[self.preset]

    @property
    def tokens_per_step(self):
        return self.global_batch * self.seq_len

    @property
    def steps(self):
        return (self.token_budget + self.tokens_per_step - 1) // self.tokens_per_step

    def validate(self):
        assert self.arm in ("fixed", "adaptive", "mlm")
        assert self.preset in ("base", "large", "tiny")
        assert self.seq_len >= 8 and 0 < self.span_frac < .5
        assert 0 <= self.suffix_probability <= 1
        assert self.global_batch >= 2 and self.global_batch % self.micro_batch == 0
        assert self.val_documents >= 2 and self.val_documents % self.micro_batch == 0
        assert self.vocab_size > 5 and self.proj_dim > 1
        assert 0 < self.sigreg_min <= self.sigreg_weight <= self.sigreg_max < 1
        assert 0 < self.rank_target_fraction < 1 and 0 <= self.controller_ema < 1
        assert self.controller_gain > 0 and self.controller_max_delta > 0
        assert self.token_budget > 0 and self.lr > 0 and self.warmup_tokens >= 0
        assert self.sigreg_knots >= 2 and self.sigreg_directions > 0
        assert min(self.diagnostic_every, self.eval_every, self.save_every,
                   self.micro_batch, self.rank_positions) > 0
        return self

    def fingerprint(self):
        # Batch partition is a memory choice; it does not change the global loss.
        values = asdict(self)
        values.pop("micro_batch")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    @classmethod
    def read(cls, path):
        with open(path) as f:
            return cls(**json.load(f)).validate()
