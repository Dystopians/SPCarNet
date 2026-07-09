#!/bin/bash
# GEMS Stage-4 GOAL #E-01 (ladder L1b): ONE full-budget E3-style distillation
# per scene on the PRIMARY anchor, then PJ transport on top.
# Frozen recipe (pre-registered in LEDGER #E-01, one config across scenes):
#   teacher = frozen Phase-J ELA at density x3 (kout 0.12, jitter 1.5,
#   interp 1.5, seed 0); distill channel = features + SH
#   (--feature_rest_lr_mult 1.0), weights/positions frozen, 30k -> 40k.
# Usage: l1_distill_scene.sh <scene> <gpu>
set -e
cd /data/peilincai/mesh-splatting
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
G1=/data/peilincai/gems_stage1
SCENE=$1; GPU=$2
CKPT=$G1/models/${SCENE}_cleanfixed30k/point_cloud/iteration_30000/point_cloud_state_dict.pt
SRCMODEL=$G1/models/${SCENE}_cleanfixed30k
AUG=$G1/datasets_aug/${SCENE}_cleanfixed_teacher_d3
NAME=l1_${SCENE}_cleanfixed_distill
D=$G1/models/$NAME

echo "=== L1 $SCENE: teacher bake ==="
env CUDA_VISIBLE_DEVICES=$GPU $PY -m tools.gems_train.teacher_factory \
  --scene $SCENE --checkpoint $CKPT --out-root $AUG --gpu 0 \
  --kout-frac 0.12 --jitter-count-frac 1.5 --interp-count-frac 1.5 \
  --seed 0 --iteration-tag 30000

echo "=== L1 $SCENE: features+SH distillation FT 30k->40k ==="
rm -rf $D; mkdir -p $D/point_cloud
cp -r $SRCMODEL/point_cloud/iteration_30000 $D/point_cloud/
env CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=online $PY train.py -s $AUG -m $D \
  --images images -r -1 --eval \
  --split_strategy file --split_file $AUG/split.json \
  --load_iteration 30000 --iterations 40000 --seed 0 \
  --densify_until_iter 30000 \
  --skip_restricted_delaunay --freeze_topology_updates \
  --test_iterations -1 --save_iterations 40000 \
  --weight_lr 0.0 --lr_triangles_points_init 0.0 --feature_rest_lr_mult 1.0 \
  --wandb_disable_fixed_views --enable_wandb --wandb_project mesh-splatting \
  --wandb_group gems_stage4_l1 --wandb_name gems_$NAME
rm -rf $D/point_cloud/iteration_30000   # retention: final iterate only

DCKPT=$D/point_cloud/iteration_40000/point_cloud_state_dict.pt
echo "=== L1 $SCENE: distilled BASE row (diagnostic reference) ==="
env $PY run_eval.py --checkpoint $DCKPT --scene $SCENE \
  --out $G1/eval/l1_${SCENE}_distillbase_v1 --gpu $GPU \
  --skip-geometry --skip-downstream

echo "=== L1 $SCENE: evidence cache + ECR row + audit ==="
$PY -m tools.ecr.build_cache --checkpoint $DCKPT --scene $SCENE \
  --out $G1/ecr_cache/${SCENE}_l1distill --gpu $GPU
$PY run_eval.py --checkpoint $DCKPT --scene $SCENE \
  --out $G1/eval/l1_${SCENE}_distill_pj2026_v1 --gpu $GPU \
  --renderer ecr --ecr-cache $G1/ecr_cache/${SCENE}_l1distill
env CUDA_VISIBLE_DEVICES=$GPU $PY tools/audit_test_path.py \
  --checkpoint $DCKPT --scene $SCENE \
  --out $G1/eval/l1_${SCENE}_distill_pj2026_audit \
  --ecr --ecr-cache $G1/ecr_cache/${SCENE}_l1distill --fast
echo "=== L1 $SCENE DONE ==="
