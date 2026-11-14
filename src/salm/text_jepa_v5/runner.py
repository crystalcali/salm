import json
import math
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_model
from .objective import ConditionalSIGReg, SmoothController, diagnostics, objective


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision(device):
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This configuration requires a BF16-capable CUDA device")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def mask_start(cfg, step, mode=None):
    rng = random.Random(cfg.seed + 1009 * step)
    span = max(2, round(cfg.seq_len * cfg.span_frac))
    if mode is None:
        mode = "suffix" if rng.random() < cfg.suffix_probability else "infill"
    start = cfg.seq_len - span if mode == "suffix" else rng.randrange(1, cfg.seq_len - span)
    return mode, start


def train_step(model, optimizer, regularizer, x_cpu, start, weight, mask_id, diagnose=False):
    cfg = model.cfg
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    if cfg.arm == "mlm":
        total = 0.
        for xb in x_cpu.split(cfg.micro_batch):
            with precision(device):
                loss = model.mlm_loss(xb.to(device), start, mask_id)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MLM loss")
            (loss * len(xb) / len(x_cpu)).backward()
            total += float(loss.detach()) * len(xb) / len(x_cpu)
        metrics["mlm_ce"] = total
    else:
        predictions, targets = [], []
        with torch.no_grad():
            for xb in x_cpu.split(cfg.micro_batch):
                with precision(device):
                    p, t = model.jepa(xb.to(device), start)
                predictions.append(p.float())
                targets.append(t.float())
        p = torch.cat(predictions).detach().requires_grad_(True)
        t = torch.cat(targets).detach().requires_grad_(True)
        loss, mse, reg = objective(p, t, regularizer, weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite JEPA loss")
        gp, gt = torch.autograd.grad(loss, (p, t))
        for offset, xb in enumerate(x_cpu.split(cfg.micro_batch)):
            with precision(device):
                pp, tt = model.jepa(xb.to(device), start)
            sl = slice(offset * cfg.micro_batch, offset * cfg.micro_batch + len(xb))
            torch.autograd.backward((pp.float(), tt.float()), (gp[sl], gt[sl]))
        metrics.update(loss=float(loss.detach()), mse=float(mse.detach()),
                       sigreg=float(reg.detach()), sigreg_weight=weight)
        if diagnose:
            metrics.update(diagnostics(p, t, cfg.rank_positions))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm,
                                             error_if_nonfinite=True)
    optimizer.step()
    metrics["grad_norm"] = float(grad_norm)
    return metrics


@torch.no_grad()
def evaluate(model, corpus, mask_id):
    cfg = model.cfg
    device = next(model.parameters()).device
    model.eval()
    result = {}
    for source in ("general", "reasoning"):
        x = corpus.validation(source, cfg.val_documents)
        for mode in ("infill", "suffix"):
            _, start = mask_start(cfg, 0, mode)
            if mode == "infill":
                start = (cfg.seq_len - model.span) // 2
            prefix = f"{source}_{mode}_"
            if cfg.arm == "mlm":
                losses = []
                for xb in x.split(cfg.micro_batch):
                    with precision(device):
                        losses.append(float(model.mlm_loss(xb.to(device), start, mask_id)))
                result[prefix + "mlm_ce"] = float(np.mean(losses))
                continue
            predictions, targets, shuffled = [], [], []
            for i in range(0, len(x), cfg.micro_batch):
                xb = x[i:i + cfg.micro_batch].to(device)
                wrong = x.roll(1, 0)[i:i + cfg.micro_batch].to(device)
                with precision(device):
                    p, t = model.jepa(xb, start)
                    pw, _ = model.jepa(wrong, start)
                predictions.append(p.float())
                targets.append(t.float())
                shuffled.append(pw.float())
            p, t, wrong = map(torch.cat, (predictions, targets, shuffled))

            def retrieval(query, truth=t):
                scores = torch.einsum("btd,ctd->tbc", F.normalize(query, dim=-1),
                                      F.normalize(truth, dim=-1))
                labels = torch.arange(len(query), device=device)[None]
                return float((scores.argmax(-1) == labels).float().mean())

            result[prefix + "mse"] = float(F.mse_loss(p, t))
            result[prefix + "retrieval"] = retrieval(p)
            result[prefix + "shuffled_retrieval"] = retrieval(wrong)
            result[prefix + "shuffled_mse"] = float(F.mse_loss(wrong, t))
            result[prefix + "chance"] = 1 / len(x)
            result.update({prefix + k: v for k, v in diagnostics(p, t, cfg.rank_positions).items()})
    return result


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, cfg, corpus, model, optimizer):
    # Only load own checkpoints from the configured private repository.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config_fingerprint"] != cfg.fingerprint():
        raise ValueError("Resume configuration mismatch; use a new run ID")
    if state["corpus_fingerprint"] != corpus.fingerprint:
        raise ValueError("Resume corpus/tokenizer mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state


def environment(device):
    import platform
    return {"python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported()}


def train(cfg, corpus, output, mask_id, hub=None, max_steps=None, max_hours=None,
          device=None, resume=True):
    cfg.validate()
    if corpus.seq_len != cfg.seq_len or corpus.manifest["vocab_size"] != cfg.vocab_size:
        raise ValueError("Prepared corpus does not match model configuration")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "latest.pt"
    if path.exists() and not resume:
        raise FileExistsError("Existing run; resume or choose a new directory")
    model = build_model(cfg).to(device)
    regularizer = ConditionalSIGReg(cfg.sigreg_knots, cfg.sigreg_directions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    controller = SmoothController(cfg.sigreg_weight)
    sampler = corpus.sampler(cfg.seed)
    completed, history, elapsed_before, compute_seconds = 0, [], 0., 0.
    if path.exists():
        state = load_checkpoint(path, cfg, corpus, model, opt)
        completed, history = state["step"], state["history"]
        controller = SmoothController(**state["controller"])
        sampler.cursor = state["cursor"]
        elapsed_before, compute_seconds = state["wall_seconds"], state["compute_seconds"]
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output / "environment.json").write_text(json.dumps(environment(device), indent=2))
    print(json.dumps({"encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "resume_step": completed, "planned_steps": cfg.steps, **environment(device)}), flush=True)
    start_time = time.monotonic()
    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {}
    # Notebook calls also run in the main thread; tests may use other threads.
    import threading
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, request_stop)

    def save():
        atomic_checkpoint(path, {"schema": 5, "config": asdict(cfg),
            "config_fingerprint": cfg.fingerprint(), "corpus_fingerprint": corpus.fingerprint,
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "controller": controller.state_dict(), "cursor": sampler.cursor,
            "step": completed, "history": history, "rng": capture_rng(),
            "wall_seconds": elapsed_before + time.monotonic() - start_time,
            "compute_seconds": compute_seconds})
        (output / "metrics.json").write_text(json.dumps({"history": history,
            "step": completed, "complete": completed == cfg.steps,
            "tokens_seen": completed * cfg.tokens_per_step,
            "compute_seconds": compute_seconds,
            "corpus_fingerprint": corpus.fingerprint,
            "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
        from .report import plot_run
        plot_run(output)
        if hub is not None:
            hub.upload_run(output)

    # If a first update fails, there is still a consistent pre-update recovery point.
    if not path.exists():
        save()
    limit = min(cfg.steps, max_steps) if max_steps is not None else cfg.steps
    try:
        for step in range(completed + 1, limit + 1):
            if stop_requested or (max_hours is not None and
                    elapsed_before + time.monotonic() - start_time >= max_hours * 3600):
                break
            tokens = step * cfg.tokens_per_step
            if tokens <= cfg.warmup_tokens:
                factor = max(.01, tokens / max(1, cfg.warmup_tokens))
            else:
                fraction = min(1., (tokens - cfg.warmup_tokens) /
                               max(1, cfg.steps * cfg.tokens_per_step - cfg.warmup_tokens))
                factor = .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
            for group in opt.param_groups:
                group["lr"] = cfg.lr * factor
            mode, start = mask_start(cfg, step)
            x = sampler.batch(cfg.global_batch)
            diagnose = step % cfg.diagnostic_every == 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.monotonic()
            # Any failure leaves the last fully committed checkpoint intact.
            metrics = train_step(model, opt, regularizer, x, start,
                controller.weight, mask_id, diagnose=diagnose)
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_seconds += time.monotonic() - tick
            completed = step
            if cfg.arm == "adaptive" and diagnose:
                controller.update(cfg, metrics, tokens)
            entry = {"step": step, "tokens_seen": tokens, "mode": mode,
                "lr": cfg.lr * factor, "controller_next_weight": controller.weight,
                "compute_seconds": compute_seconds,
                "wall_seconds": elapsed_before + time.monotonic() - start_time,
                **metrics}
            if step == 1 or step % cfg.eval_every == 0 or step == limit:
                entry.update({"val_" + k: v for k, v in evaluate(model, corpus, mask_id).items()})
                print(json.dumps(entry), flush=True)
            history.append(entry)
            if step % cfg.save_every == 0 or step == limit:
                save()
        if completed != limit:
            save()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return model, history


def benchmark(cfg, micro_batches=(4, 8, 16, 32), warmup=1, repeats=2):
    """Measure complete optimizer updates, not encoder-only forward speed."""
    import gc
    from dataclasses import replace
    if not torch.cuda.is_available():
        raise RuntimeError("Run the G4 benchmark in a GPU Colab runtime")
    device = torch.device("cuda")
    results = []
    for arm in ("fixed", "adaptive", "mlm"):
        for micro in micro_batches:
            trial = replace(cfg, arm=arm, micro_batch=micro).validate()
            model = opt = reg = x = None
            try:
                seed_all(cfg.seed)
                model = build_model(trial).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
                reg = ConditionalSIGReg(cfg.sigreg_knots, cfg.sigreg_directions).to(device)
                x = torch.randint(5, cfg.vocab_size, (cfg.global_batch, cfg.seq_len))
                torch.cuda.reset_peak_memory_stats()
                times = []
                for i in range(warmup + repeats):
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    train_step(model, opt, reg, x, (cfg.seq_len - model.span) // 2,
                               cfg.sigreg_weight, 4)
                    torch.cuda.synchronize()
                    if i >= warmup:
                        times.append(time.monotonic() - tick)
                row = {"arm": arm, "micro_batch": micro,
                    "tokens_per_second": cfg.tokens_per_step / float(np.median(times)),
                    "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            except torch.cuda.OutOfMemoryError:
                row = {"arm": arm, "micro_batch": micro, "oom": True}
            finally:
                del model, opt, reg, x
                gc.collect()
                torch.cuda.empty_cache()
            results.append(row)
            print(json.dumps(row), flush=True)
    return {"environment": environment(device), "results": results}
