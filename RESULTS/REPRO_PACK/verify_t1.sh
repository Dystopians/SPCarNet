#!/bin/bash
# REPRO_PACK verification: regenerate T1 from scratch in a fresh process
# (fresh corpus walk -> fresh tables) into a temp dir and diff against the
# shipped tables. PASS iff byte-identical modulo the .md timestamp line.
set -uo pipefail
PY=${PY:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gems_repro_t1.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
cd "$REPO"

$PY tools/gems/report/collect.py --out "$TMP/all_rows.json" >"$TMP/collect.log" 2>&1 \
  || { echo "FAIL: collect.py (see $TMP/collect.log)"; exit 1; }
$PY tools/gems/report/tables.py --rows "$TMP/all_rows.json" --outdir "$TMP" \
  >"$TMP/tables.log" 2>&1 \
  || { echo "FAIL: tables.py (see $TMP/tables.log)"; exit 1; }

RESULT="$REPO/RESULTS/REPRO_PACK/verify_t1_result.txt"
{
  echo "verify_t1 run: $(date -u +%Y-%m-%dT%H:%M:%SZ) (fresh process, fresh corpus walk)"
  fail=0
  for f in T1_main_pareto.md T1_main_pareto.csv T1_per_scene_detail.csv; do
    if diff <(grep -v '^_generated' "$REPO/RESULTS/aggregate/$f") \
            <(grep -v '^_generated' "$TMP/$f") > /dev/null; then
      echo "  $f: IDENTICAL (modulo timestamp line)"
    else
      echo "  $f: DIFFERS"
      fail=1
    fi
  done
  [ $fail -eq 0 ] && echo "VERDICT: PASS" || echo "VERDICT: FAIL"
} | tee "$RESULT"
grep -q "VERDICT: PASS" "$RESULT"
