#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType
from ss3dm_prior.meshsplatopt.teacher_recovery import run_teacher_recovery_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR11_teacher_recovery_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    edit = MeshEdit(edit_id="recovery_smoke_edit", edit_type=MeshSplatOptEditType.APPEARANCE_RESET.value, defect_id="smoke")
    edit_json = out / "edit.json"
    edit_json.write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
    plan = run_teacher_recovery_contract(
        model_path=out / "missing_model",
        edit_json=edit_json,
        output_dir=out / "recovery",
        iterations=200,
    )
    cache_files_exist = all(Path(p).exists() for p in plan.teacher_cache_files)
    checks = {
        "cache_files_written": cache_files_exist,
        "missing_render_path_documented": plan.status == "SOFT_PASS_MISSING_RENDER_PATH",
        "edited_unedited_metrics_distinguished": "edited_region_loss_available" in plan.recovery_metrics
        and "unedited_teacher_distillation_available" in plan.recovery_metrics,
    }
    report = {"status": "SOFT PASS" if all(checks.values()) else "FAIL", "checks": checks, "plan": plan.to_dict()}
    (out / "teacher_recovery_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MeshSplatOpt Stage R11 Teacher Recovery Smoke", "", f"Status: `{report['status']}`", "", "## Checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    (out / "teacher_recovery_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] == "FAIL":
        raise SystemExit(f"Stage R11 teacher recovery smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
