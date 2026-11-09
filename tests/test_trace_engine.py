import pytest


def test_differentiable_layer_capture_and_objective():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from salm.trace_jepa.engine import LayerCapture, compute_batch
    from salm.trace_jepa.objective import TraceJEPAObjective

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = torch.nn.Linear(8, 8)
        def forward(self, x):
            return x + self.ff(x)

    class ToyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(50, 8)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(), Block()])
            self.proj = torch.nn.Linear(8, 50)
            self.config = SimpleNamespace(hidden_size=8, layer_types=["conv", "conv"])
        def forward(self, input_ids, attention_mask=None, labels=None,
                    use_cache=False, logits_to_keep=0):
            h = self.embed(input_ids)
            for l in self.model.layers:
                h = l(h)
            logits = self.proj(h)
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 50),
                    labels[:, 1:].reshape(-1), ignore_index=-100
                )
            else:
                loss = None
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(1)
    model = ToyCausal()
    heads = torch.nn.ModuleDict({
        "0": TraceJEPAObjective(8, latent_size=8, predictor_width=16,
                                projections=12, knots=9),
    })
    batch = {
        "full_ids": torch.tensor([[1,2,3,4,5], [5,4,3,2,1]]),
        "full_mask": torch.ones(2,5,dtype=torch.long),
        "labels": torch.tensor([[-100,-100,3,4,5], [-100,-100,3,2,1]]),
        "target_ids": torch.tensor([[3,4,5], [3,2,1]]),
        "target_mask": torch.ones(2,3,dtype=torch.long),
        "cut_pos": torch.tensor([1,1]),
    }
    with LayerCapture(model, [0]) as c:
        loss, metrics = compute_batch(model, heads, c, batch, 0.2)
        loss.backward()
    assert model.embed.weight.grad is not None
    assert model.embed.weight.grad.abs().sum() > 0
    assert heads["0"].projector[1].weight.grad.abs().sum() > 0
    assert metrics["jepa"] > 0
    assert len(model.model.layers[0]._forward_hooks) == 0


def test_lora_target_names_are_from_real_lfm3_code():
    # Prevent accidentally swapping to llama-style up_proj/down_proj.
    from pathlib import Path
    source = (Path(__file__).parent.parent / "src/salm/trace_jepa/engine.py").read_text()
    assert '"w1", "w2", "w3"' in source
