"""Prepared token windows: indexed resume, no lossy streaming shuffle buffer."""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch


def digest_text(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def partition(group):
    bucket = int(group[:16], 16) % 100
    return "val" if bucket < 2 else "train"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path):
    values = {p.name: sha_file(p) for p in sorted(Path(path).glob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class PreparedCorpus:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.fingerprint = sha_file(self.root / "manifest.json")
        self.seq_len = self.manifest["seq_len"]
        self.arrays = {}
        for name, item in self.manifest["files"].items():
            if not re.fullmatch(r"(?:general|reasoning)_(?:train|val)\.bin", name):
                raise ValueError("Unexpected corpus file name")
            path = self.root / name
            expected = item["rows"] * self.seq_len * np.dtype("uint32").itemsize
            if path.stat().st_size != expected or (verify and sha_file(path) != item["sha256"]):
                raise ValueError(f"Corpus integrity mismatch: {name}")
            self.arrays[name[:-4]] = np.memmap(path, dtype="uint32", mode="r",
                                              shape=(item["rows"], self.seq_len))
        if verify and directory_hash(self.root / "tokenizer") != self.manifest["tokenizer_hash"]:
            raise ValueError("Tokenizer integrity mismatch")

    def sampler(self, seed, cursor=0):
        return MixedSampler(self, seed, cursor)

    def validation(self, source, count):
        arr = self.arrays[source + "_val"]
        if len(arr) < count:
            raise ValueError("Insufficient prepared validation documents")
        return torch.from_numpy(np.array(arr[:count], dtype=np.int64))


class MixedSampler:
    """Exact 7:3 schedule over fixed-length windows; each source is shuffled by epoch."""
    def __init__(self, corpus, seed, cursor=0):
        self.corpus, self.seed, self.cursor = corpus, seed, cursor
        self._permutations = {}

    def _row(self, source, rank):
        arr = self.corpus.arrays[source + "_train"]
        epoch, offset = divmod(rank, len(arr))
        key = (source, epoch)
        if key not in self._permutations:
            # Keep one permutation per source, bounded memory on long runs.
            self._permutations = {k: v for k, v in self._permutations.items() if k[0] != source}
            salt = 0 if source == "general" else 1_000_003
            self._permutations[key] = np.random.default_rng(
                self.seed + salt + 10_007 * epoch).permutation(len(arr))
        return int(self._permutations[key][offset])

    def batch(self, size):
        if min(len(self.corpus.arrays[s + "_train"]) for s in ("general", "reasoning")) < size:
            raise ValueError("Prepare at least global_batch documents per source")
        rows, keys = [], set()
        for i in range(self.cursor, self.cursor + size):
            before, after = i * 3 // 10, (i + 1) * 3 // 10
            source, rank = ("reasoning", before) if after > before else ("general", i - before)
            row = self._row(source, rank)
            if (source, row) in keys:
                # No silent repeats at an epoch boundary: prepare more unique windows.
                raise ValueError("Repeated document in a batch; increase prepared corpus size")
            keys.add((source, row))
            rows.append(self.corpus.arrays[source + "_train"][row])
        self.cursor += size
        return torch.from_numpy(np.array(rows, dtype=np.int64))


def extract_reasoning(row):
    """One problem family, one verified, complete generation; fail closed on missing flags."""
    for i, text in enumerate(row.get("generations") or []):
        flags = row.get("correctness_math_verify") or []
        complete = row.get("is_reasoning_complete") or []
        if i < len(flags) and flags[i] is True and i < len(complete) and complete[i] is True:
            problem = str(row["problem"])
            return problem + "\n\n" + text, digest_text(problem)
    return None


def source_stream(repo, config, revision, reasoning=False):
    from datasets import load_dataset
    stream = load_dataset(repo, config, revision=revision, split="train", streaming=True)
    for row in stream:
        if reasoning:
            extracted = extract_reasoning(row)
            if extracted is not None:
                yield extracted
        else:
            text = str(row.get("text", ""))
            if text.strip():
                yield text, digest_text(text)


def resolve_sources():
    from huggingface_hub import HfApi
    specs = {
        "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        "ru": ("HuggingFaceFW/fineweb-2", "rus_Cyrl"),
        "reasoning": ("open-r1/OpenR1-Math-220k", "default"),
    }
    api = HfApi()
    resolved = {k: {"repo": repo, "config": config, "revision": api.dataset_info(repo).sha}
                for k, (repo, config) in specs.items()}
    resolved["evaluation"] = {repo: api.dataset_info(repo).sha for repo in
                              ("sentence-transformers/stsb", "stanfordnlp/sst2")}
    return resolved


def iter_source(spec, reasoning=False):
    return source_stream(spec["repo"], spec["config"], spec["revision"], reasoning)


def train_tokenizer(destination, specs, documents=10_000, vocab_size=50_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    core = Tokenizer(models.BPE(unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()

    def samples():
        # English and Russian are used only for fitting vocabulary; model pilots remain EN.
        for language in ("en", "ru"):
            n = 0
            for text, group in iter_source(specs[language]):
                if partition(group) != "train":
                    continue
                yield text[:16_000]
                n += 1
                if n >= documents:
                    break
            if n < documents:
                raise ValueError("Insufficient tokenizer training documents")

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"])
    core.train_from_iterator(samples(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>",
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", mask_token="<mask>",
        model_max_length=1024)
    if len(tokenizer) != vocab_size:
        raise ValueError("Tokenizer corpus too small to reach requested vocabulary")
    tokenizer.save_pretrained(destination)
    return tokenizer


def prepare(root, token_budget, seq_len=1024, val_documents=128,
            tokenizer_documents=10_000, vocab_size=50_000, seed=42, ru_fraction=0.):
    """One deterministic random window per unique document/problem family.

    Prepared arrays intentionally replace streaming *during training*. Source revisions,
    tokenizer, windows and file hashes are immutable; an integer cursor resumes exactly.
    ru_fraction is a fraction of the GENERAL 70%, not of all training tokens.
    """
    from transformers import AutoTokenizer
    assert 0 <= ru_fraction <= 1
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("Corpus already exists; use it or choose a new preparation ID")
    root.mkdir(parents=True, exist_ok=True)
    specs_path = root / "sources.json"
    specs = json.loads(specs_path.read_text()) if specs_path.exists() else resolve_sources()
    specs_path.write_text(json.dumps(specs, indent=2))
    tok_dir = root / "tokenizer"
    if (tok_dir / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tok_dir.mkdir(exist_ok=True)
        tokenizer = train_tokenizer(tok_dir, specs, tokenizer_documents, vocab_size)
    if len(tokenizer) != vocab_size:
        raise ValueError("Vocabulary differs from configuration")
    steps = (token_budget + 128 * seq_len - 1) // (128 * seq_len)
    windows = steps * 128
    counts = {"general": max(256, windows - windows * 3 // 10 + 256),
              "reasoning": max(256, windows * 3 // 10 + 256)}
    files, global_hashes, prepared_languages = {}, set(), {}
    for source in ("general", "reasoning"):
        specs_selected = ("en", "ru") if source == "general" else ("reasoning",)
        streams = {k: iter_source(specs[k], k == "reasoning") for k in specs_selected}
        received = {"train": 0, "val": 0}
        language_counts = {"en": 0, "ru": 0, "reasoning": 0}
        goals = {"train": counts[source], "val": val_documents}
        names = {p: f"{source}_{p}.bin" for p in goals}
        handles = {p: (root / names[p]).open("wb") for p in goals}
        seen_groups = set()
        metadata = (root / f"{source}_metadata.jsonl").open("w")
        rng = random.Random(seed + (source == "reasoning"))
        attempt = 0
        try:
            while any(received[p] < goals[p] for p in goals):
                lang = "reasoning" if source == "reasoning" else (
                    "ru" if int((attempt + 1) * ru_fraction) > int(attempt * ru_fraction) else "en")
                attempt += 1
                try:
                    text, group = next(streams[lang])
                except StopIteration as e:
                    raise ValueError(f"Exhausted {lang}; reduce budget or expand source") from e
                part = partition(group)
                if received[part] >= goals[part] or group in seen_groups:
                    continue
                content_hash = digest_text(text)
                if content_hash in global_hashes:
                    continue
                ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
                if len(ids) < seq_len:
                    continue
                offset = rng.randrange(len(ids) - seq_len + 1)
                np.asarray(ids[offset:offset + seq_len], dtype="uint32").tofile(handles[part])
                seen_groups.add(group)
                global_hashes.add(content_hash)
                metadata.write(json.dumps({"group": group, "content": content_hash,
                    "part": part, "offset": offset, "language": lang}) + "\n")
                received[part] += 1
                language_counts[lang] += 1
                if sum(received.values()) % 2000 == 0:
                    print(source, received, flush=True)
        finally:
            for f in handles.values():
                f.close()
            metadata.close()
        for p, name in names.items():
            files[name] = {"rows": received[p], "sha256": sha_file(root / name)}
        prepared_languages[source] = language_counts
        print(source, "prepared", received, language_counts, flush=True)
    manifest = {"schema": 5, "seed": seed, "seq_len": seq_len,
        "files": files, "sources": specs, "vocab_size": len(tokenizer),
        "tokenizer_hash": directory_hash(tok_dir), "ru_fraction_general": ru_fraction,
        "prepared_language_counts": prepared_languages,
        "mixture": {"general": .7, "reasoning": .3},
        "group_split": "normalized document/problem SHA256; bucket <2/100 is val",
        "note": "Exact normalized duplicates excluded; semantic/web benchmark overlap not ruled out."}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return PreparedCorpus(root)
