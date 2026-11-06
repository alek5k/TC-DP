#!/bin/bash
set -euo pipefail

# Resolve repo root (directory containing this script, then one level up)
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$ROOT_DIR/data"

# ensure folders exist
mkdir -p "$DATA_DIR/demonstration" "$DATA_DIR/inference" "$DATA_DIR/inference_extended"

for parent in demonstration inference inference_extended; do
  [ -d "$DATA_DIR/$parent" ] || continue
  for z in "$DATA_DIR/$parent"/*.zarr.zip; do
    [ -f "$z" ] || continue
    echo "Unzipping: $(basename "$z") -> $DATA_DIR/$parent"
    unzip -o "$z" -d "$DATA_DIR/$parent" && rm -f "$z"
  done
done