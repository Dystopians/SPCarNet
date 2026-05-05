#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${ITERATIONS:-7000}"
export MODEL_ROOT="${MODEL_ROOT:-outputs/carnet/meshprior/stageF55_parking_delayed_robust_calibrated_gate_ratio0015_7000/parking_${ITERATIONS}iter_ratio0015_delayed_robust_calibrated_gate}"
export WANDB_GROUP="${WANDB_GROUP:-final_stageF55_parking_delayed_robust_calibrated_gate_ratio0015_7000}"
export WANDB_NAME="${WANDB_NAME:-F55_parking_ratio0015_${ITERATIONS}_delayed_robust_calibrated_gate}"
export SCENE_NAME="${SCENE_NAME:-F55_parking_ratio0015_${ITERATIONS}_delayed_robust_calibrated_gate}"
export PRISM_GEOMETRY_ACQ_UNTIL_ITER="${PRISM_GEOMETRY_ACQ_UNTIL_ITER:-1400}"
export PRISM_STATS_COLLECTION_ITERS="${PRISM_STATS_COLLECTION_ITERS:-100}"
export PRISM_CANDIDATE_PRUNE_RATIO="${PRISM_CANDIDATE_PRUNE_RATIO:-0.015}"

exec bash scripts/car_model/final_run_stageF51_parking_robust_calibrated_gate_ratio004_7000.sh
