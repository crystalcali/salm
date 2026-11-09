"""Preflight for native CUDA/Triton compilation on DGX Spark."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
import sysconfig
from pathlib import Path


def python_header() -> Path:
    """Python development headers are outside venv site-packages."""
    return Path(sysconfig.get_path("include")) / "Python.h"


def require_triton_headers(cuda_available: bool, *, triton_present: bool | None = None,
                           header_path: Path | None = None):
    """Give an actionable error *before* downloading/loading 3+ GB of weights."""
    if triton_present is None:
        triton_present = importlib.util.find_spec("triton") is not None
    path = header_path if header_path is not None else python_header()
    if cuda_available and triton_present and not path.is_file():
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"CUDA Triton cannot compile native code: missing {path}. "
            f"Install Python development headers for Python {version}; "
            f"on Ubuntu: sudo apt update && sudo apt install -y "
            f"python{version}-dev build-essential. "
            f"After installation run python -m salm.cli doctor. "
            f"Your Hugging Face model cache does not need to be downloaded again."
        )


def system_report() -> dict:
    """Inspect local environment without loading any model weights."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_header": str(python_header()),
        "python_h_exists": python_header().is_file(),
        "gcc": shutil.which("gcc"),
        "triton_installed": importlib.util.find_spec("triton") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torch_cuda_available": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "gpu_name": None,
    }
    if info["torch_installed"]:
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["torch_cuda_version"] = torch.version.cuda
            info["torch_cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    info["triton_requires_python_headers"] = bool(
        info["triton_installed"] and info["torch_cuda_available"]
    )
    return info
