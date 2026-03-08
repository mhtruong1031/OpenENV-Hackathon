#!/bin/bash
# Register a Python 3.12 Jupyter kernel from the project's uv venv.
# Run from project root: bash scripts/register_kernel_312.sh

set -e
cd "$(dirname "$0")/.."

echo "Ensuring Python 3.12 venv..."
uv python install 3.12
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}" uv sync --extra train

echo "Registering Jupyter kernel 'OpenEnv Bio (Python 3.12)'..."
uv run python -m ipykernel install --user \
  --name openenv-bio-312 \
  --display-name "OpenEnv Bio (Python 3.12)"

echo "Done. Select 'OpenEnv Bio (Python 3.12)' in the notebook kernel picker."
