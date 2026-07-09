#!/bin/bash
# GEMS Stage-4: run the FINAL incumbent stack on one (scene, base) — used
# for suite completion (SS3DM/toy) and the L6 compact rows after the ladder
# closes. Pipeline: clone base cache -> frozen multiband (K,alpha)
# calibration -> frozen fusion-net training (learned|routed per the final
# incumbent) -> ECR row -> --ecr audit.
# Usage: final_stack_scene.sh <scene> <gpu> <src_cache_dirname> <checkpoint> <row_tag> <learned|routed>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2; SRC=$G1/ecr_cache/$3; CKPT=$4; TAG=$5; MODE=$6
DST=$G1/ecr_cache/${SCENE}_${TAG}_final
ROUTED_FLAG=""
[ "$MODE" = "routed" ] && ROUTED_FLAG="--routed"

echo "=== FINAL $SCENE ($TAG, $MODE): clone + multiband calibration ==="
mkdir -p $DST
for d in renders gt depths; do
  [ -d $DST/$d ] || cp -al $SRC/$d $DST/$d
done
cp -f $SRC/camera_index.json $DST/camera_index.json
$PY -m tools.ecr.build_cache --checkpoint $CKPT --scene $SCENE \
  --out $DST --gpu $GPU --fuse multiband --k-grid 2,4,8 --bands 4

echo "=== FINAL $SCENE: fusion net ($MODE) ==="
$PY -m tools.ecr.train_fusion --cache $DST --gpu $GPU $ROUTED_FLAG

echo "=== FINAL $SCENE: ECR row + audit ==="
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/final_${SCENE}_${TAG}_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $DST
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/final_${SCENE}_${TAG}_audit \
  --ecr --ecr-cache $DST --fast
echo "=== FINAL $SCENE DONE ==="
