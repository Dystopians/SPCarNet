#!/bin/bash
# GEMS Stage-4 GOAL #E-03 (ladder L2): multiband K-source transport row for
# one scene, riding the CURRENT incumbent's base checkpoint + cache data.
# Cache data files are hardlink-shared with the incumbent's cache (same
# renders/gt/depths); only the manifest/config differs.
# Usage: l2_scene.sh <scene> <gpu> <src_cache_dirname> <checkpoint> <row_tag>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2; SRC=$G1/ecr_cache/$3; CKPT=$4; TAG=$5
DST=$G1/ecr_cache/${SCENE}_${TAG}_l2mb

echo "=== L2 $SCENE: hardlink cache clone + multiband (K,alpha) calibration ==="
mkdir -p $DST
for d in renders gt depths; do
  [ -d $DST/$d ] || cp -al $SRC/$d $DST/$d
done
cp -f $SRC/camera_index.json $DST/camera_index.json
$PY -m tools.ecr.build_cache --checkpoint $CKPT --scene $SCENE \
  --out $DST --gpu $GPU --fuse multiband --k-grid 2,4,8 --bands 4

echo "=== L2 $SCENE: ECR row + audit ==="
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/l2_${SCENE}_${TAG}_multiband_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $DST
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/l2_${SCENE}_${TAG}_multiband_audit \
  --ecr --ecr-cache $DST --fast
echo "=== L2 $SCENE DONE ==="
