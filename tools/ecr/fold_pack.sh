#!/bin/bash
# GEMS Stage-4 evidence-pack folding (prompt §5/§9): copy the canonical ECR
# artifact set from gems_stage1 into the repo pack at RESULTS/STAGE4_ECR/,
# then byte-verify every copy (sha256). Re-runnable; missing optional
# artifacts (e.g. L5 Pareto while E-06 is in flight) are listed as PENDING.
set -e
G1=/data/peilincai/gems_stage1
REPO=/data/peilincai/mesh-splatting
DST=$REPO/RESULTS/STAGE4_ECR
mkdir -p $DST/gates $DST/tables $DST/difix

declare -A COPIES=(
  [$G1/analysis/final_stack/final_stack_tables.md]=$DST/tables/final_stack_tables.md
  [$G1/analysis/final_stack/final_stack_summary.json]=$DST/tables/final_stack_summary.json
  [$G1/analysis/final_stack/e07_matched_total_3dgs.md]=$DST/tables/e07_matched_total_3dgs.md
  [$G1/analysis/final_stack/e07_matched_total_3dgs.json]=$DST/tables/e07_matched_total_3dgs.json
  [$G1/analysis/final_stack/ecr_failure_cases.md]=$DST/tables/ecr_failure_cases.md
  [$G1/analysis/final_stack/ecr_failure_cases.json]=$DST/tables/ecr_failure_cases.json
  [$G1/analysis/final_stack/l5_pareto.md]=$DST/tables/l5_pareto.md
  [$G1/analysis/final_stack/l5_pareto.json]=$DST/tables/l5_pareto.json
  [$G1/analysis/e0_pj2026/e0_primary_table.md]=$DST/tables/e0_primary_table.md
  [$G1/analysis/e0_pj2026/e0_b50_table.md]=$DST/tables/e0_b50_table.md
  [$G1/analysis/e0_pj2026/l1_gate.json]=$DST/gates/l1_gate.json
  [$G1/analysis/e0_pj2026/l2_gate.json]=$DST/gates/l2_gate.json
  [$G1/analysis/e0_pj2026/l3_gate.json]=$DST/gates/l3_gate.json
  [$G1/analysis/e0_pj2026/l4_gate.json]=$DST/gates/l4_gate.json
  [$G1/analysis/e0_pj2026/l3_vs_floor.json]=$DST/gates/l3_vs_floor.json
  [$G1/analysis/e0_pj2026/l4_vs_floor.json]=$DST/gates/l4_vs_floor.json
  [$G1/analysis/difix_cell/difix_table.md]=$DST/difix/difix_table.md
  [$G1/analysis/difix_cell/attempt_log.md]=$DST/difix/attempt_log.md
  [$G1/analysis/final_stack/hierarchical_cis.md]=$DST/tables/hierarchical_cis.md
  [$G1/analysis/final_stack/hierarchical_cis.json]=$DST/tables/hierarchical_cis.json
  [$G1/analysis/final_stack/t2b_tandt_db.md]=$DST/tables/t2b_tandt_db.md
  [$G1/analysis/final_stack/t2b_tandt_db.json]=$DST/tables/t2b_tandt_db.json
  [$G1/analysis/temporal/temporal_summary.md]=$DST/tables/temporal_summary.md
  [$G1/analysis/temporal/temporal_summary.json]=$DST/tables/temporal_summary.json
  [$G1/analysis/ibr_cell/ibr_table.md]=$DST/ibr/ibr_table.md
  [$G1/ibr_cell/attempt_log.md]=$DST/ibr/attempt_log.md
)
mkdir -p $DST/ibr

MANIFEST=$DST/sha256_manifest.txt
: > $MANIFEST
PENDING=()
for SRC in "${!COPIES[@]}"; do
  OUT=${COPIES[$SRC]}
  if [ ! -f "$SRC" ]; then PENDING+=("$SRC"); continue; fi
  cp -f "$SRC" "$OUT"
  S1=$(sha256sum "$SRC" | cut -d' ' -f1)
  S2=$(sha256sum "$OUT" | cut -d' ' -f1)
  if [ "$S1" != "$S2" ]; then echo "BYTE MISMATCH: $SRC"; exit 1; fi
  echo "$S2  ${OUT#$REPO/}  <= $SRC" >> $MANIFEST
done
sort -k2 -o $MANIFEST $MANIFEST
{
  echo "# Stage-4 ECR pack — folded $(cd $REPO && git rev-parse --short HEAD)"
  echo "# Byte verification: every line above is sha256 of BOTH source and copy (asserted equal at fold time)."
  echo "# Qual grids live in RESULTS/figures/ecr_qual/ (generated in-repo, not copied)."
  for p in "${PENDING[@]}"; do echo "# PENDING (not yet banked): $p"; done
} >> $MANIFEST
echo "folded $(grep -c '^[0-9a-f]' $MANIFEST) artifacts; ${#PENDING[@]} pending"
cat $MANIFEST | tail -5
