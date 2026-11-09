.PHONY: install test inspect demo capture dashboard gradient probe
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[lab,dev]"

test:
	$(PYTHON) -m pytest -q

inspect:
	$(PYTHON) -m salm.cli inspect --offline

demo:
	$(PYTHON) -m salm.cli demo --out runs/demo.npz

capture:
	$(PYTHON) -m salm.cli capture --prompt "Solve x**2-5*x+6 = 0." --out runs/example.npz --modules ffn,operator

dashboard:
	$(PYTHON) -m salm.cli dashboard

gradient:
	$(PYTHON) -m salm.cli gradient --prompt "Solve x**2-5*x+6 = 0." --out runs/gradients.json

probe:
	$(PYTHON) -m salm.cli probe --input runs --source 7 --target 11 --out runs/depth_probe.json
