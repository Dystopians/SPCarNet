#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    component_removed: str
    expected_failure: str
    synthetic_required: bool
    public_scene_required: bool
    status: str
    evidence: str = ""
    command_hint: str = ""


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("A01_no_csef_debt", "CSEF explanation debt", "misses synthetic holes / voids", True, False, "INTERFACE_READY"),
    AblationSpec("A02_no_free_space", "negative free-space evidence", "accepts unsupported fill/snap", True, False, "INTERFACE_READY"),
    AblationSpec("A03_no_render_gate", "counterfactual render gate", "harmful edits can pass geometry-only tests", True, True, "INTERFACE_READY"),
    AblationSpec("A04_no_sparse_geometry_gate", "sparse COLMAP geometry gate", "render-only rows hide geometry regression", True, True, "PARTIAL_EVIDENCE", "R43/R44 split shows render/geometry tradeoff."),
    AblationSpec("A05_no_changed_pixel_gate", "changed-pixel gate", "global drift can masquerade as local repair", True, False, "INTERFACE_READY"),
    AblationSpec("A06_no_rollback", "rollback", "rejected edits corrupt state", True, False, "SMOKE_COVERED"),
    AblationSpec("A07_no_teacher_recovery", "teacher recovery", "edited regions keep appearance damage", True, True, "PARTIAL_EVIDENCE", "R45/R46 teacher variants are negative controls, not final positives."),
    AblationSpec("A08_delete_collapse_only", "bidirectional snap/fill", "method collapses to pruning", True, True, "PARTIAL_EVIDENCE"),
    AblationSpec("A09_snap_only", "delete/fill portfolio", "tiny or unstable gains", True, True, "NEGATIVE_EVIDENCE", "R17-R21 snap lines show small/mixed effects."),
    AblationSpec("A10_fill_only", "delete/snap portfolio", "medium recovery fails", True, True, "NEGATIVE_EVIDENCE", "R22/R26/R28 fill lines fail against matched sparse recovery."),
    AblationSpec("A11_giant_fill_no_certificate", "giant-hole certificate", "fills unknown voids", True, False, "SMOKE_COVERED"),
    AblationSpec("A12_object_prior_no_scene_gate", "scene gate for object prior", "prior hallucination", True, False, "SMOKE_COVERED"),
    AblationSpec("A13_budget_controller_disabled", "budget controller", "topology blow-up", False, True, "NEGATIVE_EVIDENCE", "R25 grows parking to 5.89M triangles and loses render."),
    AblationSpec("A14_densification_freeze_disabled", "strict topology freeze", "triangle count changes during recovery", False, True, "NEGATIVE_EVIDENCE", "R49 shows legacy controls drop to 934205 triangles; R50 fixes it."),
)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MeshSplatOpt Ablation Suite Contract",
        "",
        "| id | removed component | expected failure | synthetic | public scene | status | evidence |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ablation_id']} | {row['component_removed']} | {row['expected_failure']} | "
            f"{row['synthetic_required']} | {row['public_scene_required']} | {row['status']} | {row['evidence']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/ablation_suite")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(spec) for spec in ABLATIONS]
    (out_dir / "ablation_suite_contract.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_md(out_dir / "ablation_suite_contract.md", rows)
    complete = [row for row in rows if row["status"] in {"NEGATIVE_EVIDENCE", "SMOKE_COVERED", "PARTIAL_EVIDENCE"}]
    summary = {
        "total": len(rows),
        "evidence_backed": len(complete),
        "interface_ready_only": len(rows) - len(complete),
        "gate": "PARTIAL_PASS_NEEDS_PUBLIC_SCENE_COMPLETION",
    }
    (out_dir / "ablation_suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

