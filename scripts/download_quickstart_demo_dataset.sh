#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory containing this script, then one level up)
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$ROOT_DIR/data"

mkdir -p "$DATA_DIR"

huggingface-cli download erdrsvhr9/temporally_conditioned_diffusion_policy \
    --repo-type dataset \
    --include "demonstration/wait_at_goal_dataset_v4.zarr.zip" \
    --local-dir "$DATA_DIR"