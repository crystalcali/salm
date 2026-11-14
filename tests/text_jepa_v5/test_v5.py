import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from salm.text_jepa_v5.config import Config
from salm.text_jepa_v5.corpus import (
    PreparedCorpus,
    digest_text,
    directory_hash,
    extract_reasoning,
    partition,
    sha_file,
)
from salm.text_jepa_v5.hub import PrivateHub, safe_path
from salm.text_jepa_v5.model import build_model, view_indices
from salm.text_jepa_v5.objective import (
    ConditionalSIGReg,
    SmoothController,
    diagnostics,
    objective,
)
from salm.text_jepa_v5.runner import train, train_step


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(arm="fixed", **kw):
    cfg = Config(arm=arm, preset="tiny", seq_len=16, vocab_size=32, proj_dim=8,
        global_batch=4, micro_batch=2, val_documents=4, sigreg_directions=4, sigreg_knots=5,
        token_budget=192, eval_every=3, save_every=1, diagnostic_every=1,
        controller_warmup_tokens=0, warmup_tokens=0)
    return replace(cfg, **kw).validate()


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    files = {}
    rng = np.random.default_rng(3)
    for source in ("general", "reasoning"):
        for part in ("train", "val"):
            name = f"{source}_{part}.bin"
            rows = 40 if part == "train" else 4
            rng.integers(5, 32, (rows, 16), dtype="uint32").tofile(root / name)
            files[name] = {"rows": rows, "sha256": sha_file(root / name)}
    (root / "manifest.json").write_text(json.dumps({"seq_len": 16, "vocab_size": 32,
        "files": files, "tokenizer_hash": directory_hash(root / "tokenizer")}))
    return PreparedCorpus(root)


def test_common_encoder_initialization_and_distinct_layers():
    fixed = build_model(tiny())
    mlm = build_model(tiny("mlm"))
    for key, value in fixed.encoder.state_dict().items():
        assert torch.equal(value, mlm.encoder.state_dict()[key])
    assert not torch.equal(fixed.encoder.layers.layers[0].linear1.weight,
                           fixed.encoder.layers.layers[1].linear1.weight)


def test_target_gradients_and_gradcache_match_direct_global_loss():
    cfg = tiny(max_grad_norm=1e8)
    direct, cached = build_model(cfg), build_model(cfg)
    reg = ConditionalSIGReg(5, 4)
    x = torch.randint(5, 32, (4, 16))
    torch.manual_seed(123)
    p, t = direct.jepa(x, 5)
    t.retain_grad()
    loss, _, _ = objective(p, t, reg, .25)
    loss.backward()
    assert t.grad.abs().sum() > 0
    assert direct.project[1].weight.grad.abs().sum() > 0
    torch.manual_seed(123)
    train_step(cached, torch.optim.SGD(cached.parameters(), lr=0), reg, x, 5, .25, 4)
    for (name, p), q in zip(direct.named_parameters(), cached.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=3e-6, rtol=3e-4), name


def test_hidden_text_cannot_leak_to_suffix_prediction():
    model = build_model(tiny()).eval()
    x = torch.randint(5, 32, (4, 16))
    changed = x.clone()
    changed[:, 12:] = (changed[:, 12:] + 7) % 32
    with torch.no_grad():
        p, t = model.jepa(x, 12)
        pp, tt = model.jepa(changed, 12)
    assert torch.equal(p, pp)
    assert not torch.equal(t, tt)


def test_target_encoding_is_independent_of_visible_context():
    model = build_model(tiny()).eval()
    x = torch.randint(5, 32, (4, 16))
    changed = x.clone()
    visible, _ = view_indices(16, 4, 5, x.device)
    changed[:, visible] = (changed[:, visible] + 7) % 32
    with torch.no_grad():
        _, t = model.jepa(x, 5)
        _, tt = model.jepa(changed, 5)
    assert torch.equal(t, tt)


def test_mlm_masks_all_hidden_tokens():
    model = build_model(tiny("mlm"))
    observed = []
    handle = model.encoder.register_forward_pre_hook(lambda _, args: observed.append(args[0].clone()))
    x = torch.randint(5, 32, (4, 16))
    model.mlm_loss(x, 12, mask_id=4)
    handle.remove()
    assert (observed[0][:, 12:] == 4).all()
    assert torch.equal(observed[0][:, :12], x[:, :12])


def test_position_only_signal_has_zero_conditional_rank_and_variance():
    z = torch.randn(1, 8, 16).expand(8, -1, -1).clone()
    metrics = diagnostics(z, z)
    assert metrics["target_rank_position"] == 0
    assert metrics["target_content_std"] < 1e-6
    assert metrics["rank_position_limit"] == 7
    random_metrics = diagnostics(torch.randn(8, 8, 16), torch.randn(8, 8, 16))
    assert 1 < random_metrics["target_rank_position"] <= 7.0001
    torch.manual_seed(42)
    reg = ConditionalSIGReg(17, 64)
    assert reg(z) > 3 * reg(torch.randn_like(z))


def test_controller_is_smooth_bounded_and_resumable():
    cfg = tiny()
    controller = SmoothController(.25)
    low = {"pred_rank_position": 0., "target_rank_position": 0., "rank_position_limit": 127}
    previous = controller.weight
    for _ in range(200):
        controller.update(cfg, low, 100)
        assert abs(controller.weight - previous) <= cfg.controller_max_delta + 1e-12
        assert cfg.sigreg_min <= controller.weight <= cfg.sigreg_max
        previous = controller.weight
    assert controller.weight == cfg.sigreg_max
    restored = SmoothController(**controller.state_dict())
    high = {"pred_rank_position": 100., "target_rank_position": 100., "rank_position_limit": 127}
    for _ in range(200):
        assert restored.update(cfg, high, 100) == controller.update(cfg, high, 100)
    assert controller.weight < cfg.sigreg_max


def test_sampler_exact_schedule_and_resume(corpus):
    sampler = corpus.sampler(42)
    first = sampler.batch(4)
    cursor = sampler.cursor
    second = sampler.batch(4)
    resumed = corpus.sampler(42, cursor)
    assert torch.equal(second, resumed.batch(4))
    assert not torch.equal(first, second)
    # 7 general and 3 reasoning rows over every ten positions.
    sources = ["reasoning" if (i + 1) * 3 // 10 > i * 3 // 10 else "general" for i in range(10)]
    assert sources.count("reasoning") == 3


def test_reasoning_families_and_verification_flags():
    row = {"problem": "What is 2 + 2?", "generations": ["bad", "good"],
           "correctness_math_verify": [False, True], "is_reasoning_complete": [True, True]}
    text, group = extract_reasoning(row)
    assert text.endswith("good")
    assert group == digest_text("  WHAT is 2 + 2?\n")
    assert partition(group) == partition(digest_text("What is 2 + 2?"))
    assert extract_reasoning({"problem": "x", "generations": ["unverified"]}) is None


def test_modified_corpus_is_rejected(corpus):
    path = corpus.root / "general_train.bin"
    with path.open("r+b") as f:
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="integrity"):
        PreparedCorpus(corpus.root)


def test_prepare_bilingual_tokenizer_and_disjoint_document_windows(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from salm.text_jepa_v5 import corpus as module
    specs = {k: {"language": k} for k in ("en", "ru", "reasoning")}
    monkeypatch.setattr(module, "resolve_sources", lambda: specs)

    def synthetic_source(spec, reasoning=False):
        import random
        language = spec["language"]
        rng = random.Random({"en": 1, "ru": 2, "reasoning": 3}[language])
        alphabet = "абвгдежзийклмнопрстуфхцчшщэюя" if language == "ru" else "abcdefghijklmnopqrstuvwxyz"
        for i in range(3000):
            text = f"{language} document {i}: " + " ".join(
                "".join(rng.choices(alphabet, k=8)) for _ in range(30))
            yield text, digest_text(text)

    monkeypatch.setattr(module, "iter_source", synthetic_source)
    prepared = module.prepare(tmp_path / "prepared", token_budget=128, seq_len=16,
        val_documents=4, tokenizer_documents=20, vocab_size=400)
    tok = AutoTokenizer.from_pretrained(prepared.root / "tokenizer")
    assert len(tok) == 400 and tok.mask_token_id == 4
    assert tok.decode(tok.encode("Привет world", add_special_tokens=False)) == "Привет world"
    groups = {"train": set(), "val": set()}
    contents = set()
    for path in prepared.root.glob("*_metadata.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert partition(row["group"]) == row["part"]
            assert row["content"] not in contents
            contents.add(row["content"])
            groups[row["part"]].add(row["group"])
            assert row["language"] != "ru"  # English-only model pilot.
    assert not groups["train"] & groups["val"]


@pytest.mark.parametrize("arm", ["fixed", "adaptive", "mlm"])
def test_training_resume_matches_uninterrupted(corpus, tmp_path, arm):
    cfg = tiny(arm)
    full = tmp_path / (arm + "_full")
    split = tmp_path / (arm + "_split")
    reference, _ = train(cfg, corpus, full, 4, device="cpu")
    train(cfg, corpus, split, 4, device="cpu", max_steps=2)
    resumed, _ = train(cfg, corpus, split, 4, device="cpu")
    for key, value in reference.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    a = torch.load(full / "latest.pt", weights_only=False)
    b = torch.load(split / "latest.pt", weights_only=False)
    assert a["cursor"] == b["cursor"] == 12
    assert a["controller"] == b["controller"]
    for key in ("loss", "mse", "sigreg", "mlm_ce"):
        if key in a["history"][-1]:
            assert a["history"][-1][key] == b["history"][-1][key]
    with pytest.raises(ValueError, match="configuration mismatch"):
        train(replace(cfg, lr=.002), corpus, split, 4, device="cpu")


def test_private_hub_rejects_public_destination_and_unsafe_paths():
    hub = object.__new__(PrivateHub)
    class FakeApi:
        def repo_info(self, *_, **__):
            class Info:
                private = False
            return Info()
    hub.api, hub.repo_id = FakeApi(), "owner/repo"
    with pytest.raises(ValueError, match="private"):
        hub.ensure_private()
    for value in ("../private", "/root", "id/../other", "a//b"):
        with pytest.raises(ValueError):
            safe_path(value)
