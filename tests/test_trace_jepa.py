"""CPU tests; no LFM weights or HF network required."""
import json
import random

import pytest

from salm.trace_jepa.data import read_traces, split_tasks, verify_disjoint, build_pair, collate_pairs
from salm.trace_jepa.prepare import prepare_arithmetic


class Tok:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [1 + (ord(char) % 90) for char in text]


def test_verified_smoke_data_and_split(tmp_path):
    train = tmp_path/"train.jsonl"
    val = tmp_path/"val.jsonl"
    report = prepare_arithmetic(str(train), str(val), tasks=32, seed=12)
    assert report["train"] + report["val"] == 32
    a, b = read_traces(str(train)), read_traces(str(val))
    verify_disjoint(a, b)
    assert all(row["verification"]["arithmetic"] is True for row in a+b)


def test_causal_random_cut_no_leakage():
    pytest.importorskip('torch')
    row = {
        "task_id": "t1", "prompt": "Solve 2+3",
        "steps": ["First identify the sum", "Then add the numbers"],
        "final_answer": "5",
    }
    tok = Tok()
    a = build_pair(row, tok, random.Random(0), 256)
    b = collate_pairs([a], tok.pad_token_id)
    assert a["full_ids"][:a["cut_pos"]+1] != a["target_ids"]
    assert a["full_ids"][a["cut_pos"]+1:] == a["target_ids"]
    assert b["cut_pos"][0].item() == a["cut_pos"]
    assert b["target_ids"].shape[1] == len(a["target_ids"])
    assert all(t == -100 for t in a["labels"][:len(tok.encode("Task:\nSolve 2+3\nSolution:\n"))])


def test_no_split_leakage_by_prompt_and_task():
    base = {"task_id": "x", "prompt": "Same prompt", "steps": ["a"]}
    with pytest.raises(ValueError, match="task_id"):
        verify_disjoint([base], [base])
    with pytest.raises(ValueError, match="duplicate prompts"):
        verify_disjoint([base], [{**base, "task_id": "y"}])


def test_reproducible_cut():
    row = {"task_id": "x", "prompt": "Problem", "steps": ["a", "b", "c"]}
    rng1, rng2 = random.Random(33), random.Random(33)
    assert build_pair(row, Tok(), rng1)["cut"] == build_pair(row, Tok(), rng2)["cut"]


def test_sigreg_rejects_bad_sizes():
    torch = pytest.importorskip("torch")
    from salm.trace_jepa.sigreg import SIGReg
    with pytest.raises(ValueError):
        SIGReg()(torch.zeros(1, 8))
    with pytest.raises(ValueError):
        SIGReg()(torch.zeros(2, 3, 4))


def test_sigreg_distribution_and_backward():
    torch = pytest.importorskip("torch")
    from salm.trace_jepa.sigreg import SIGReg
    torch.manual_seed(17)
    reg = SIGReg(projections=32, knots=9)
    rand = torch.randn(128, 16, requires_grad=True)
    loss = reg(rand)
    loss.backward()
    assert torch.isfinite(loss)
    assert rand.grad is not None and torch.isfinite(rand.grad).all()
    torch.manual_seed(17)
    null = reg(torch.zeros(128, 16))
    assert null > loss


def test_shared_jepa_no_stopgrad():
    torch = pytest.importorskip("torch")
    from salm.trace_jepa.objective import TraceJEPAObjective
    torch.manual_seed(10)
    obj = TraceJEPAObjective(hidden_size=12, latent_size=6,
                             predictor_width=16, projections=12, knots=9)
    online = torch.randn(16, 12, requires_grad=True)
    target = torch.randn(16, 12, requires_grad=True)
    result = obj(online, target)
    result.total.backward()
    assert online.grad is not None and online.grad.abs().sum().item() > 0
    assert target.grad is not None and target.grad.abs().sum().item() > 0
    assert torch.isfinite(result.total)
