"""Command-line entry point: python -m salm.cli [inspect|demo|capture|gradient|probe|dashboard]."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "LiquidAI/LFM3-1.7B-Base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SALM — study LFM3 internals")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inspect", help="Architecture facts and cache/parameter estimates")
    p.add_argument("--offline", action="store_true", help="No HF request")
    p.add_argument("--model", default=DEFAULT_MODEL)
    commands.add_parser("doctor", help="Check Python.h, compiler, PyTorch, CUDA and Triton before GPU capture")
    p = commands.add_parser("inspect-file", help="Verify actual capture kind, shape and prompt")
    p.add_argument("--path", required=True)
    p = commands.add_parser("demo", help="Generate labeled synthetic capture, no model download")
    p.add_argument("--out", default="runs/demo.npz")
    p = commands.add_parser("capture", help="Capture LFM3 activations via forward hooks")
    p.add_argument("--model", default=DEFAULT_MODEL)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Text to capture")
    group.add_argument("--prompt-file", help="Read multi-line UTF-8 prompt from a file")
    p.add_argument("--out", default="runs/capture.npz")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--modules", default="", help="Comma-separated: ffn,operator")
    p = commands.add_parser("batch", help="Capture one prompt per line, loading model only once")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--out-dir", default="runs/batch")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--modules", default="ffn,operator")
    p = commands.add_parser("intervene", help="Scale FFN or operator output; measure next-token KL")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", required=True)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--all", action="store_true", help="Sweep all decoder blocks")
    p.add_argument("--module", choices=["ffn", "operator"], default="ffn")
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", default="runs/intervention.json")
    p = commands.add_parser("gradient", help="Backpropagate causal LM loss into layer states")
    p.add_argument("--model", default=DEFAULT_MODEL)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Text to evaluate")
    group.add_argument("--prompt-file", help="Read multi-line UTF-8 prompt from a file")
    p.add_argument("--out", default="runs/gradients.json")
    p.add_argument("--max-tokens", type=int, default=128)
    p = commands.add_parser("probe", help="Supervised depth predictor from multiple capture files")
    p.add_argument("--input", default="runs")
    p.add_argument("--source", type=int, default=7)
    p.add_argument("--target", type=int, default=11)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--out", default="runs/depth_probe.json")
    p = commands.add_parser("dashboard", help="Launch local Streamlit visualization")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        from .model import report
        if args.offline:
            result = report()
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            result = report(config=config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "doctor":
        from .diagnostics import system_report
        result = system_report()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["triton_requires_python_headers"] and not result["python_h_exists"]:
            print("\nFIX: sudo apt update && sudo apt install -y python3.12-dev build-essential")
            raise SystemExit(2)
        if result["torch_cuda_available"] is False:
            print("\nWARNING: PyTorch does not detect CUDA; check the ARM64 GPU build.")
    elif args.command == "inspect-file":
        from .capture_info import inspect_capture
        print(json.dumps(inspect_capture(args.path), indent=2, ensure_ascii=False))
    elif args.command == "demo":
        from .capture import demo_capture
        print("Synthetic capture:", demo_capture(args.out))
    elif args.command == "batch":
        from .capture import capture_activations
        from .model import load_model
        prompts = [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]
        if not prompts:
            raise SystemExit("Empty prompt file")
        model, tokenizer = load_model(args.model)
        for i, prompt in enumerate(prompts):
            dest = Path(args.out_dir) / f"prompt_{i:04d}.npz"
            mods = tuple(x.strip() for x in args.modules.split(",") if x.strip())
            print(f"[{i+1}/{len(prompts)}]", capture_activations(
                model, tokenizer, prompt, str(dest), args.max_tokens, mods
            ))
    elif args.command == "intervene":
        from .model import load_model
        from .intervention import run_intervention
        model, tokenizer = load_model(args.model)
        result = run_intervention(model, tokenizer, args.prompt, args.layer,
                                  args.module, args.scale, args.max_tokens, args.all)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print("Saved:", target)
    elif args.command in {"capture", "gradient"}:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
        if not prompt_text.strip():
            raise SystemExit("Prompt must not be empty")
        from .model import load_model
        from .capture import capture_activations, capture_gradients
        model, tokenizer = load_model(args.model)
        if args.command == "capture":
            modules = tuple(x.strip() for x in args.modules.split(",") if x.strip())
            path = capture_activations(model, tokenizer, prompt_text, args.out,
                                       args.max_tokens, modules)
        else:
            path = capture_gradients(model, tokenizer, prompt_text, args.out,
                                     args.max_tokens)
        print("Saved:", path)
    elif args.command == "probe":
        from .probe import train_depth_probe
        result = train_depth_probe(args.input, args.source, args.target,
                                   args.out, args.epochs, args.width)
        print(json.dumps(result, indent=2))
    elif args.command == "dashboard":
        app = Path(__file__).with_name("app.py")
        command = [
            sys.executable, "-m", "streamlit", "run", str(app),
            "--server.address", args.host, "--server.port", str(args.port),
            "--server.headless", "true",
        ]
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
