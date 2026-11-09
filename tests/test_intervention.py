import pytest


def test_intervention_changes_outputs_without_mutating_weights():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from salm.intervention import run_intervention

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Linear(4, 4, bias=False)
            self.feed_forward = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return x + self.conv(x) + self.feed_forward(x)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 4)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])
            self.lm_head = torch.nn.Linear(4, 32)
            self.config = SimpleNamespace(layer_types=["conv", "conv"])

        def forward(self, input_ids, use_cache=False):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return SimpleNamespace(logits=self.lm_head(x))

    class Tok:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids):
            return str(ids[0])

    torch.manual_seed(4)
    model = ToyModel().eval()
    tokens = torch.tensor([[1, 2, 3]])
    baseline = model(tokens).logits.clone()
    result = run_intervention(model, Tok(), "abc", 0, "ffn", 0, 128, True)
    assert result["experiment"] == "module_ablation"
    assert len(result["rows"]) == 2
    assert all(r["kl_base_to_modified"] >= 0 for r in result["rows"])
    assert torch.allclose(baseline, model(tokens).logits)



def test_corpus_intervention_aggregates_across_tokens_without_mutating_weights():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from salm.corpus_intervention import run_corpus_intervention

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Linear(5, 5, bias=False)
            self.feed_forward = torch.nn.Linear(5, 5, bias=False)
        def forward(self, h):
            return h + self.conv(h) + self.feed_forward(h)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50, 5)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.head = torch.nn.Linear(5, 50)
            self.config = SimpleNamespace(layer_types=["conv", "conv"], _name_or_path="tiny")
        def forward(self, input_ids, use_cache=False):
            h = self.embedding(input_ids)
            for layer in self.model.layers:
                h = layer(h)
            return SimpleNamespace(logits=self.head(h))

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1,2,3,4,5]])}

    torch.manual_seed(7)
    model = Tiny().eval()
    before = model(torch.tensor([[1,2,3,4,5]])).logits.detach().clone()
    result = run_corpus_intervention(
        model, Tokenizer(), ["First prompt", "Second prompt"],
        which="ffn", scale=0.0, max_tokens=16, skip_prefix=1
    )
    assert result["experiment"] == "corpus_module_ablation"
    assert result["prompts_scored"] == 2
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["scored_positions"] == 6
        assert row["kl_base_to_modified"] >= 0
        assert 0 <= row["top1_change_rate"] <= 1
    assert torch.allclose(model(torch.tensor([[1,2,3,4,5]])).logits, before)
