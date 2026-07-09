#!/bin/bash
# GEMS Stage-4 GOAL #E-06 (L5 cache Pareto): one (scene, variant) point.
# Usage: l5_scene_variant.sh <scene> <gpu> <variant>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2; V=$3
SRC=$G1/ecr_cache/${SCENE}_cleanfixed30k_l2mb
DST=$G1/ecr_cache/${SCENE}_l5_${V}
CKPT=$G1/models/${SCENE}_cleanfixed30k/point_cloud/iteration_30000/point_cloud_state_dict.pt

echo "=== L5 $SCENE $V: variant build + frozen recalibration ==="
$PY -m tools.ecr.l5_compress --src $SRC --out $DST --variant $V --gpu $GPU
echo "=== L5 $SCENE $V: routed fusion net (frozen recipe) ==="
$PY -m tools.ecr.train_fusion --cache $DST --gpu $GPU --routed
echo "=== L5 $SCENE $V: ECR row + audit ==="
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/l5_${SCENE}_${V}_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $DST
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $CKPT --scene $SCENE --out $G1/eval/l5_${SCENE}_${V}_audit \
  --ecr --ecr-cache $DST --fast
echo "=== L5 $SCENE $V DONE ==="
