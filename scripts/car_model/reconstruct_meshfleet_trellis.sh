#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/data2/car_meshes/MeshFleet_TRELLIS}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/car_meshes/MeshFleet_TRELLIS_RECONSTRUCTED}"

mkdir -p "$OUTPUT_DIR/train" "$OUTPUT_DIR/test"

python "$DATASET_ROOT/reconstruct_data.py" \
  --shard_dir "$DATASET_ROOT/train" \
  --output_dir "$OUTPUT_DIR/train"

python "$DATASET_ROOT/reconstruct_data.py" \
  --shard_dir "$DATASET_ROOT/test" \
  --output_dir "$OUTPUT_DIR/test"

if [ -f "$DATASET_ROOT/train/metadata.csv" ]; then
  cp "$DATASET_ROOT/train/metadata.csv" "$OUTPUT_DIR/train/metadata.csv"
fi

if [ -f "$DATASET_ROOT/test/metadata.csv" ]; then
  cp "$DATASET_ROOT/test/metadata.csv" "$OUTPUT_DIR/test/metadata.csv"
fi
