#!/usr/bin/env bash
# Create (or refresh) the `acdb-active` conda environment and verify the install.
#
#   bash scripts/setup_env.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="acdb-active"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    echo "  \$HOME/miniconda3/bin/conda init bash && exec bash"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' exists — updating from environment.yml"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "[setup] creating env '$ENV_NAME'"
    conda env create -f environment.yml
fi

PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
echo "[setup] python = $PY"

echo "[setup] verifying imports and PC backend ..."
"$PY" - <<'PYCODE'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
import numpy, pandas, requests, dotenv                      # noqa: F401
from causallearn.search.ConstraintBased.PC import pc        # noqa: F401
from causal_discovery.active import LEVELS, build_instance  # noqa: F401
from causal_discovery.active.episode import run_pc
inst = build_instance(LEVELS[1], 12345, 300, 150)
graph = run_pc(inst.observational_data, 0.05)
print(f"  ok: numpy {numpy.__version__}, pandas {pandas.__version__}")
print(f"  ok: PC on a d={inst.config.d} instance -> "
      f"{graph.num_directed_edges} directed / {graph.num_undirected_edges} undirected edges")
PYCODE

if [ ! -f .env ]; then
    echo
    echo "[setup] WARNING: no .env file. Create one with your OpenRouter key:"
    echo "    echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env"
else
    "$PY" - <<'PYCODE'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
from causal_discovery.active.llm_client import resolve_api_key
key = resolve_api_key(".env")
print(f"[setup] API key loaded from .env ({key[:10]}...{key[-4:]})")
PYCODE
fi

echo
echo "[setup] running the offline test suite ..."
"$PY" -m pytest tests/test_active.py -q

echo
echo "[setup] done. Activate with:  conda activate $ENV_NAME"
