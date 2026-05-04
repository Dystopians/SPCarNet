#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


SECTIONS: dict[str, str] = {
    "meshsplatopt_neurips_manuscript_skeleton.md": """# MeshSplatOpt NeurIPS Manuscript Skeleton

## Title
MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting

## Abstract Checklist
- State CSEF as evidence-debt field.
- State reversible bidirectional edits.
- State counterfactual render/geometry certificates.
- State clean-to-compact R53/R55 result honestly.
- State current repair/fill limitations.

## Main Claims
1. Clean-to-compact recovery dominates the strongest clean long parking baseline with 70% fewer triangles.
2. CSEF/edit/gate machinery provides an auditable repair optimizer, but real-scene bidirectional edit gains are still weaker than the clean-to-compact route.
3. Negative results are part of the contribution boundary.
""",
    "meshsplatopt_neurips_method.md": """# Method

## CSEF
Define positive surface evidence, negative free-space evidence, explanation debt, prior support, topology cost, and uncertainty.

## Edit Calculus
DELETE, COLLAPSE, SNAP, SPLIT, FILL, APPEARANCE_RESET are represented as reversible edit records.

## Counterfactual Certificates
Each accepted edit must pass render, sparse-depth, normal, free-space, changed-pixel, topology, and budget/state gates.

## Clean-to-Compact Recovery
The current strongest route is clean 22k -> smallest-area triangle compaction -> strict fixed-topology recovery.
""",
    "meshsplatopt_neurips_experiments.md": """# Experiments

## Headline Parking Result
Use `outputs/carnet/meshsplatopt/clean_to_compact_tables/clean_to_compact_results.md`.

## Cross-Scene Sparse Recovery
Use `outputs/carnet/meshsplatopt/sparse_recovery_tables/sparse_recovery_results.md`.

## Required Remaining Work
Run full-budget sweep on at least two additional scenes before claiming broad superiority.
""",
    "meshsplatopt_neurips_ablation.md": """# Ablations

Use `outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_contract.md`.

Required completion before main-conference submission:
- no CSEF debt
- no free-space
- no render gate
- no sparse geometry gate
- no rollback
- delete-only
- snap-only
- fill-only
- no topology freeze
""",
    "meshsplatopt_neurips_related_work.md": """# Related Work

Position against Mesh Splatting, 3DGS compression/pruning, surface-aligned splatting, classical mesh repair, COLMAP/MVS, and geometry priors.
""",
    "meshsplatopt_neurips_reviewer_risk_checklist.md": """# Reviewer Risk Checklist

| risk | status | mitigation |
|---|---|---|
| Looks like pruning only | active | Lead with CSEF/edit calculus, but be honest that R53 is clean-to-compact. |
| Single-scene headline | active | Need full-budget cross-scene replication. |
| Hiding negative edit results | controlled | R17-R28, R45-R52, R56 are explicitly logged. |
| Training metrics mixed with independent metrics | controlled | Tables use render.py + metrics.py + geometry JSON. |
| Prior hallucination | active | Giant-hole prior-only fills must stay diagnostic until real evidence passes. |
""",
    "meshsplatopt_neurips_final_go_no_go.md": """# Final Go / No-Go

Current decision: `WORKSHOP_OR_ARXIV_UNTIL_R15_R16_COMPLETE`.

R53/R55 are strong enough to justify continued full-budget work. They are not by themselves enough for a NeurIPS main claim because the original prompt requires multi-scene full-budget validation, ablations, and real bidirectional repair evidence.

Upgrade to `GO_NEURIPS` only after:
1. R53-like clean-to-compact dominance replicates on at least 2/3 scenes, or another CSEF edit route produces stronger cross-scene gains.
2. Ablation suite shows at least three core components are empirically necessary.
3. Giant-hole repair has synthetic plus at least one real/realistic non-hallucinated example.
""",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/car_model/reports")
    parser.add_argument("--manifest", default="outputs/carnet/meshsplatopt/neurips_package/manifest.json")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for filename, text in SECTIONS.items():
        path = out_dir / filename
        path.write_text(text.strip() + "\n", encoding="utf-8")
        written.append(str(path.relative_to(ROOT)))
    manifest = {
        "status": "PACKAGE_SCAFFOLD_READY",
        "written": written,
        "required_tables": [
            "outputs/carnet/meshsplatopt/clean_to_compact_tables/clean_to_compact_results.md",
            "outputs/carnet/meshsplatopt/sparse_recovery_tables/sparse_recovery_results.md",
            "outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_contract.md",
        ],
    }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

