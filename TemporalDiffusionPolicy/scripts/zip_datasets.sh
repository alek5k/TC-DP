#!/bin/bash
set -euo pipefail

# Resolve repo root (directory containing this script, then one level up)
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$ROOT_DIR/data"

for parent in demonstration inference inference_extended; do
  parent_dir="$DATA_DIR/$parent"
  [ -d "$parent_dir" ] || continue

  # Work inside the parent so the zip doesn't include "demonstration/" or "inference/" prefixes
  (
    cd "$parent_dir"
    for d in *.zarr/; do
      [ -d "$d" ] || continue
      echo "Zipping: $(basename "$d")"
      zip -9 -r "${d%/}.zip" "${d%/}" -x '*.mkv' '*/full_image_raw/*'
    done
  )
done

