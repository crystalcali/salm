import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Text-JEPA v5: fixed / smooth SIGReg / span-MLM")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--tokens", type=int, default=100_000_000)
    prep.add_argument("--tokenizer-documents", type=int, default=10_000)
    prep.add_argument("--ru-fraction", type=float, default=0., help="RU fraction of GENERAL texts, large stage only")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--config", default="configs/text_jepa_v5/base.json")
    bench.add_argument("--output", default="runs/text_jepa_v5/benchmark.json")
    bench.add_argument("--micro-batches", default="8,16,32")
    run = commands.add_parser("train")
    run.add_argument("--config", default="configs/text_jepa_v5/base.json")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--arm", choices=["fixed", "adaptive", "mlm"])
    run.add_argument("--seed", type=int)
    run.add_argument("--repo-id")
    run.add_argument("--run-id")
    run.add_argument("--resume-hub", action="store_true")
    run.add_argument("--max-hours", type=float)
    ext = commands.add_parser("external")
    ext.add_argument("--checkpoint", required=True)
    ext.add_argument("--data", required=True)
    ext.add_argument("--final-sts-test", action="store_true")
    report = commands.add_parser("compare")
    report.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .corpus import prepare
        prepare(args.data, args.tokens, tokenizer_documents=args.tokenizer_documents,
                ru_fraction=args.ru_fraction)
    elif args.command == "benchmark":
        from .runner import benchmark
        cfg = Config.read(args.config)
        result = benchmark(cfg, tuple(map(int, args.micro_batches.split(","))))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    elif args.command == "train":
        from transformers import AutoTokenizer

        from .corpus import PreparedCorpus
        from .runner import train
        cfg = Config.read(args.config)
        overrides = {k: getattr(args, k) for k in ("arm", "seed") if getattr(args, k) is not None}
        cfg = replace(cfg, **overrides).validate()
        corpus = PreparedCorpus(args.data)
        tokenizer = AutoTokenizer.from_pretrained(corpus.root / "tokenizer")
        hub = None
        if args.repo_id:
            from .hub import PrivateHub
            if not args.run_id:
                parser.error("--repo-id requires --run-id")
            hub = PrivateHub(args.repo_id, args.run_id)
        if args.resume_hub:
            if hub is None:
                parser.error("--resume-hub requires --repo-id and --run-id")
            hub.download_run(args.output)
        train(cfg, corpus, args.output, tokenizer.mask_token_id, hub=hub, max_hours=args.max_hours)
    elif args.command == "external":
        from .corpus import PreparedCorpus
        from .external import evaluate_external
        result = evaluate_external(args.checkpoint, PreparedCorpus(args.data), final_test=args.final_sts_test)
        print(json.dumps(result, indent=2))
    else:
        from .report import compare
        print(json.dumps(compare(args.root), indent=2))


if __name__ == "__main__":
    main()
