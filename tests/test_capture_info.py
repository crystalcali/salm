import json
import os

import numpy as np

from salm.capture_info import choose_capture, inspect_capture
from salm.capture import demo_capture


def test_inspect_demo_is_not_real(tmp_path):
    p = demo_capture(str(tmp_path / "demo.npz"))
    info = inspect_capture(p)
    assert info["kind"] == "synthetic_demo"
    assert info["hidden_shape"] == [25, 48, 64]
    assert not info["is_real_lfm3_shape"]


def test_real_is_preferred_to_newer_demo(tmp_path):
    real = tmp_path / "newton.npz"
    np.savez_compressed(real, hidden=np.zeros((25, 4, 2048), dtype=np.float16),
                        metadata=np.array(json.dumps({"kind": "activation_capture", "prompt": "Newton"})))
    demo = demo_capture(str(tmp_path / "demo.npz"))
    assert choose_capture([demo, real]) == 1
    assert inspect_capture(real)["is_real_lfm3_shape"]


def test_choose_latest_when_only_demo(tmp_path):
    a = demo_capture(str(tmp_path / "a.npz"))
    b = demo_capture(str(tmp_path / "b.npz"))
    os.utime(a, (100, 100))
    os.utime(b, (200, 200))
    assert choose_capture([a, b]) == 1
