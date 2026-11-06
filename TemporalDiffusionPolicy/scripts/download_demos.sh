#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT_DIR/src/temporal_diffusion_policy/tools/dataset_upload.py" --download --category demonstration "$@"
