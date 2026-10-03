#!/usr/bin/env bash
# Create the two Python environments the code expects, inside the repository:
#   .venv_eval    Python 3.11: environment, tools, evaluator     (requirements.txt)
#   .venv_render  Python 3.13: the Blender `bpy` module for renders (requirements-render.txt)
# The agent prompt names `.venv_eval/bin/python -m ppbench.v2.session`, so keep that path.
#   PY311=python3.11 PY313=python3.13 bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
"${PY311:-python3.11}" -m venv .venv_eval
.venv_eval/bin/pip install -q --upgrade pip
.venv_eval/bin/pip install -q -r requirements.txt
"${PY313:-python3.13}" -m venv .venv_render
.venv_render/bin/pip install -q --upgrade pip
.venv_render/bin/pip install -q -r requirements-render.txt
PYTHONPATH=. .venv_eval/bin/python scripts/smoke_test.py
