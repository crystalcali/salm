"""Exercise the actual notebook activation code in an isolated running Python."""
import ast
import json
import subprocess
import sys
from pathlib import Path


def test_colab_current_kernel_and_restart_can_import_src_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/text_jepa_v5_colab.ipynb").read_text())
    activation_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "activate_source":
                activation_cells.append(ast.get_source_segment(source, node))
    assert len(activation_cells) == 2  # installation and independently rerun HF cell
    for activation in activation_cells:
        script = f"""
import sys
assert 'salm' not in sys.modules
try:
    import salm
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('This test must start without the checkout on sys.path')
{activation}
root = activate_source({str(root)!r})
activate_source(root)  # idempotent when a cell is rerun
from salm.text_jepa_v5.hub import PrivateHub, safe_path
from salm.text_jepa_v5.config import Config
assert Config().seq_len == 1024
assert safe_path('pilot/run') == 'pilot/run'
assert sys.path.count(str(root / 'src')) == 1
try:
    activate_source({str(tmp_path)!r})
except RuntimeError as error:
    assert 'установочную ячейку' in str(error)
else:
    raise AssertionError('Missing source must fail before authentication')
print('Current kernel imports work')
"""
        # -I ignores cwd/PYTHONPATH; -S ignores site-packages and .pth files.
        result = subprocess.run([sys.executable, "-I", "-S", "-c", script],
                                cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "Current kernel imports work" in result.stdout
