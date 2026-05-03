from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherRecoveryPlan:
    model_path: str
    edit_json: str
    output_dir: str
    iterations: int
    status: str
    teacher_cache_files: list[str]
    recovery_metrics: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_teacher_recovery_contract(
    *,
    model_path: str | Path,
    edit_json: str | Path,
    output_dir: str | Path,
    iterations: int = 200,
) -> TeacherRecoveryPlan:
    out = Path(output_dir)
    cache = out / "teacher_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    edit_json = Path(edit_json)
    notes: list[str] = []
    cache_files: list[str] = []

    edit_payload = json.loads(edit_json.read_text(encoding="utf-8")) if edit_json.is_file() else {}
    (cache / "teacher_metadata.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "edit_json": str(edit_json),
                "edit_id": edit_payload.get("edit_id", "unknown"),
                "cache_kind": "contract_placeholder",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cache_files.append(str(cache / "teacher_metadata.json"))
    for name in ["rgb", "depth", "normal", "alpha", "visibility_mask", "edit_region_mask"]:
        path = cache / f"{name}.npz"
        np.savez(path, values=np.zeros((1, 1), dtype=np.float32), available=False)
        cache_files.append(str(path))

    renderable = model_path.exists() and (model_path / "point_cloud").exists()
    if renderable:
        status = "PASS_CONTRACT_ONLY_REAL_RENDER_NOT_INVOKED"
        notes.append("Renderable-looking model path exists, but R11 contract runner does not invoke training.")
    else:
        status = "SOFT_PASS_MISSING_RENDER_PATH"
        notes.append("No renderable Mesh Splatting model path found; wrote cache/recovery contract only.")

    plan = TeacherRecoveryPlan(
        model_path=str(model_path),
        edit_json=str(edit_json),
        output_dir=str(out),
        iterations=int(iterations),
        status=status,
        teacher_cache_files=cache_files,
        recovery_metrics={
            "real_recovery_run": False,
            "edited_region_loss_available": False,
            "unedited_teacher_distillation_available": False,
        },
        notes=notes,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher_recovery_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    lines = ["# Teacher Recovery Report", "", f"- status: `{status}`", f"- iterations: `{iterations}`", "", "## Notes", ""]
    for note in notes:
        lines.append(f"- {note}")
    (out / "teacher_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan
