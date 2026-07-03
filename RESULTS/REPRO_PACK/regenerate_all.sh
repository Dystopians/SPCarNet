#!/bin/bash
# GEMS Stage-2 evidence pack — regenerate all CPU deliverables from the corpus.
# (The GPU fps bench is separate: tools/gems/run_supervised.sh fpsbench -- \
#    $PY tools/gems/report/fps_bench.py --gpu 4)
set -euo pipefail
PY=${PY:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
$PY tools/gems/report/collect.py
$PY tools/gems/report/tables.py
$PY tools/gems/report/figures.py
echo "[regenerate_all] done — RESULTS/aggregate + RESULTS/figures refreshed"
