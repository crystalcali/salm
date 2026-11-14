import torch
from torch import nn
from torch.nn import functional as F


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.normal_(module.in_proj_weight, std=.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        depth, width, heads = cfg.dimensions
        self.word = nn.Embedding(cfg.vocab_size, width)
        self.position = nn.Embedding(cfg.seq_len, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0., activation="gelu",
            batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        # Independently initialize cloned Transformer layers before constructing heads.
        self.apply(initialize)

    def forward(self, ids, positions=None, padding_mask=None):
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        h = self.word(ids) + self.position(positions)
        return self.norm(self.layers(h, src_key_padding_mask=padding_mask))


def view_indices(length, span, start, device):
    assert 0 <= start <= length - span
    pos = torch.arange(length, device=device)
    hidden = (pos >= start) & (pos < start + span)
    return pos[~hidden], pos[hidden]


class ExperimentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Backbone(cfg)
        _, width, heads = cfg.dimensions
        if cfg.arm == "mlm":
            self.mlm_transform = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                               nn.LayerNorm(width))
            self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
            self.mlm_transform.apply(initialize)
        else:
            self.query = nn.Embedding(cfg.seq_len, width)
            self.cross = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
            self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, cfg.proj_dim),
                                        nn.LayerNorm(cfg.proj_dim, elementwise_affine=False))
            self.predictor = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                nn.Linear(width, cfg.proj_dim),
                nn.LayerNorm(cfg.proj_dim, elementwise_affine=False))
            for m in (self.query, self.cross, self.project, self.predictor):
                m.apply(initialize)

    @property
    def span(self):
        return max(2, round(self.cfg.seq_len * self.cfg.span_frac))

    def jepa(self, ids, start):
        visible, hidden = view_indices(ids.shape[1], self.span, start, ids.device)
        cp = visible[None].expand(len(ids), -1)
        tp = hidden[None].expand(len(ids), -1)
        context = self.encoder(ids[:, visible], cp)
        # Target is ONLY the hidden block, never a full document containing later answers.
        # Both branches receive gradients. No EMA and no target stop-gradient.
        target_h = self.encoder(ids[:, hidden], tp)
        query = self.query(tp)
        attended, _ = self.cross(query, context, context, need_weights=False)
        return self.predictor(query + attended), self.project(target_h)

    def mlm_loss(self, ids, start, mask_id):
        _, hidden = view_indices(ids.shape[1], self.span, start, ids.device)
        masked = ids.clone()
        masked[:, hidden] = mask_id
        h = self.mlm_transform(self.encoder(masked)[:, hidden])
        # Tied decoder; logits exist only for masked positions.
        logits = F.linear(h, self.encoder.word.weight, self.mlm_bias)
        return F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size),
                               ids[:, hidden].reshape(-1))


def build_model(cfg):
    # Common encoder initialization is identical across arms of the same seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return ExperimentModel(cfg)
