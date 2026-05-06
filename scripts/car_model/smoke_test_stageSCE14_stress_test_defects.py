#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.stress_test_defects import (  # noqa: E402
    DEFECT_FAMILIES,
    make_stress_test_manifest,
    synthetic_method_scores,
    write_stress_manifest,
    write_stress_results,
)


def main() -> int:
    manifest = make_stress_test_manifest(seed=3, split="train")
    assert len(manifest["defects"]) == len(DEFECT_FAMILIES)
    assert all(manifest["reversibility"].values()), manifest["reversibility"]
    rows = synthetic_method_scores(manifest)
    planner = next(row for row in rows if row["method"] == "sce_certificate_planner")
    assert planner["defects_repaired"] >= 5
    assert planner["false_repair_rate"] == 0.0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_stress_manifest(manifest, out)
        write_stress_results(rows, out)
        assert (out / "stress_test_manifest.json").is_file()
        assert (out / "stress_test_results.json").is_file()
        assert (out / "stress_test_results.csv").is_file()
    print("SCE14 stress-test defects smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

