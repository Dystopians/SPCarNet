#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.stress_test_defects import make_stress_test_manifest, write_stress_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic SCE14 stress-test defect manifest.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    manifest = make_stress_test_manifest(seed=int(args.seed), split=str(args.split))
    write_stress_manifest(manifest, args.output_dir)
    print({"defects": len(manifest["defects"]), "all_reversible": all(manifest["reversibility"].values())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

