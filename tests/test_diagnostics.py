from pathlib import Path

import pytest

from salm.diagnostics import require_triton_headers


def test_missing_headers_detected_before_model_load(tmp_path):
    missing = tmp_path / "Python.h"
    with pytest.raises(RuntimeError, match=r"python3\.12-dev") if __import__("sys").version_info[:2] == (3, 12) else pytest.raises(RuntimeError, match="development headers"):
        require_triton_headers(True, triton_present=True, header_path=missing)


def test_not_required_for_cpu(tmp_path):
    require_triton_headers(False, triton_present=True, header_path=tmp_path / "Python.h")


def test_not_required_without_triton(tmp_path):
    require_triton_headers(True, triton_present=False, header_path=tmp_path / "Python.h")


def test_header_present(tmp_path):
    file = tmp_path / "Python.h"
    file.touch()
    require_triton_headers(True, triton_present=True, header_path=file)
