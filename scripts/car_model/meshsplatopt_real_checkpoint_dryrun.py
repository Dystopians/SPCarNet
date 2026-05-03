#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import apply_edit_to_checkpoint_copy, load_checkpoint_state, validate_checkpoint_schema
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_model",
        default="outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model",
    )
    parser.add_argument("--iteration", type=int, default=200)
    parser.add_argument(
        "--output_root",
        default="outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_model = Path(args.source_model)
    output_root = Path(args.output_root)
    model_out = output_root / "model"
    iter_dir = model_out / "point_cloud" / f"iteration_{args.iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    ckpt = source_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    payload = load_checkpoint_state(ckpt)
    valid_in, in_errors = validate_checkpoint_schema(payload)
    if not valid_in:
        raise SystemExit(f"Invalid input checkpoint: {in_errors}")
    # Remove one face as a path validation edit. This is not a method-quality claim.
    edit = MeshEdit(
        edit_id="real_checkpoint_dryrun_delete_one_face",
        edit_type=MeshSplatOptEditType.DELETE_TRIANGLES.value,
        defect_id="dryrun",
        affected_faces=[0],
        evidence_summary={"dryrun": True},
        risk_summary={"method_claim": False},
    )
    edit_json = output_root / "edit.json"
    output_root.mkdir(parents=True, exist_ok=True)
    edit_json.write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
    report = apply_edit_to_checkpoint_copy(ckpt, edit, iter_dir)
    for name in ["cfg_args", "cameras.json", "input.ply"]:
        src = source_model / name
        if src.exists():
            shutil.copy2(src, model_out / name)
    edited_payload = load_checkpoint_state(iter_dir / "point_cloud_state_dict.pt")
    valid_out, out_errors = validate_checkpoint_schema(edited_payload)
    commands = {
        "render": (
            f"CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py "
            f"-m {model_out} --iteration {args.iteration} --skip_train"
        ),
        "metrics": f"CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m {model_out}",
        "geometry": (
            f"CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py "
            f"--model_path {model_out} --iteration {args.iteration}"
        ),
    }
    summary = {
        "status": "PASS" if valid_out and report.supported else "FAIL",
        "source_model": str(source_model),
        "output_model": str(model_out),
        "iteration": args.iteration,
        "input_schema_valid": valid_in,
        "output_schema_valid": valid_out,
        "output_schema_errors": out_errors,
        "checkpoint_report": report.to_dict(),
        "commands": commands,
        "claim_note": "dry-run path validation only; not a method-quality result",
    }
    (output_root / "real_checkpoint_dryrun_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Real Checkpoint Dry-Run Report", "", f"- status: `{summary['status']}`", "", "## Commands", ""]
    for name, cmd in commands.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
        lines.append("")
    (output_root / "real_checkpoint_dryrun_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit("real checkpoint dry-run failed")


if __name__ == "__main__":
    main()
