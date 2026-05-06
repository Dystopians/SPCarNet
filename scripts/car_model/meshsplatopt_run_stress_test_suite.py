#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.stress_test_defects import synthetic_method_scores, write_stress_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic SCE14 stress-test scoring suite.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = synthetic_method_scores(manifest)
    write_stress_results(rows, args.output_dir)
    print({"methods": len(rows), "passing": sum(int(r["passes_gate"]) for r in rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

