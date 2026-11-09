import json

import numpy as np

from salm.capture import demo_capture
from salm.cli import build_parser
from salm.metrics import layer_rows


def test_demo_snapshot_roundtrip(tmp_path):
    p = demo_capture(str(tmp_path / "demo.npz"))
    with np.load(p, allow_pickle=False) as item:
        hidden = item["hidden"]
        meta = json.loads(str(item["metadata"].item()))
        assert hidden.shape == (25, 48, 64)
        assert item["ffn"].shape == (24, 48, 64)
        assert item["operator"].shape == (24, 48, 64)
        assert item["operator_effective"].shape == (24, 48, 64)
        assert item["ffn_effective"].shape == (24, 48, 64)
        assert len(item["token_strings"]) == 48
    assert meta["kind"] == "synthetic_demo"
    assert len(layer_rows(hidden)) == 24


def test_cli_parser():
    assert build_parser().parse_args(["inspect", "--offline"]).offline is True
    assert build_parser().parse_args(["probe", "--source", "3", "--target", "8"]).target == 8
    assert build_parser().parse_args(["capture", "--prompt-file", "example.txt"]).prompt_file == "example.txt"
