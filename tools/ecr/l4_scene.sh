#!/bin/bash
# GEMS Stage-4 GOAL #E-05 (ladder L4): routed per-pixel fusion row for one
# scene, riding the CURRENT incumbent's cache (hardlink clone; the frozen
# net + rewritten manifest live in the clone).
# Usage: l3_scene.sh <scene> <gpu> <src_cache_dirname> <checkpoint> <row_tag>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2; SRC=$G1/ecr_cache/$3; CKPT=$4; TAG=$5
DST=$G1/ecr_cache/${SCENE}_${TAG}_l4routed

echo "=== L4 $SCENE: hardlink cache clone ==="
mkdir -p $DST
for d in renders gt depths; do
  [ -d $DST/$d ] || cp -al $SRC/$d $DST/$d
done
cp -f $SRC/camera_index.json $DST/camera_index.json
cp -f $SRC/manifest.json $DST/manifest.json

echo "=== L4 $SCENE: train fusion net (train-only LOO) ==="
$PY -m tools.ecr.train_fusion --cache $DST --gpu $GPU --routed

echo "=== L4 $SCENE: ECR row + audit ==="
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/l4_${SCENE}_${TAG}_routed_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $DST
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/l4_${SCENE}_${TAG}_routed_audit \
  --ecr --ecr-cache $DST --fast
echo "=== L4 $SCENE DONE ==="
