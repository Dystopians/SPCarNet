#!/bin/bash
# TOPCONF EXP-T2B (GOAL #E-11): one T&T/DB scene through the full frozen
# pipeline: clean30k anchor training -> base row -> PJ-2026 cache/row/audit
# -> final stack (multiband + routed net) row/audit. NO new mechanisms; the
# ladder is closed — these are transfer/external-validity rows.
# Usage: t2b_scene.sh <scene> <gpu>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2
case $SCENE in
  tandt_truck)  SRC=/data/peilincai/mesh_datasets/tandt_db/tandt/truck ;;
  tandt_train)  SRC=/data/peilincai/mesh_datasets/tandt_db/tandt/train ;;
  db_drjohnson) SRC=/data/peilincai/mesh_datasets/tandt_db/db/drjohnson ;;
  db_playroom)  SRC=/data/peilincai/mesh_datasets/tandt_db/db/playroom ;;
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac
MODEL=$G1/models/${SCENE}_clean30k
CKPT=$MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt
CACHE=$G1/ecr_cache/${SCENE}_clean30k

echo "=== T2B $SCENE: clean30k anchor training (frozen recipe, seed 0) ==="
if [ -f $CKPT ]; then
  echo "checkpoint exists, skipping training"
else
  env WANDB_MODE=online CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    -s $SRC -m $MODEL --images images -r -1 --eval \
    --iterations 30000 --seed 0 \
    --enable_wandb --wandb_project mesh-splatting \
    --wandb_name gems_${SCENE}_clean30k --wandb_disable_fixed_views
fi
test -f $CKPT || { echo "MISSING CKPT after training: $CKPT"; exit 3; }

echo "=== T2B $SCENE: base anchor row ==="
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/${SCENE}_clean30k_v1 --gpu $GPU \
  --skip-geometry --skip-downstream

echo "=== T2B $SCENE: PJ-2026 cache + row + audit ==="
$PY -m tools.ecr.build_cache --checkpoint $CKPT --scene $SCENE \
  --out $CACHE --gpu $GPU
$PY run_eval.py --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/e0_${SCENE}_clean30k_pj2026_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $CACHE --skip-geometry --skip-downstream
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $CKPT --scene $SCENE \
  --out $G1/eval/e0_${SCENE}_clean30k_pj2026_audit \
  --ecr --ecr-cache $CACHE --fast

echo "=== T2B $SCENE: final stack (frozen l2mb + routed) ==="
bash tools/ecr/final_stack_scene.sh $SCENE $GPU ${SCENE}_clean30k \
  $CKPT clean30k routed
echo "=== T2B $SCENE DONE ==="
