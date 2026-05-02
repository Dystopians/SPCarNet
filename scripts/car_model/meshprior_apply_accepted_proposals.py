"""Apply scene-gated MeshPrior proposals to a safe mesh copy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.apply_proposals import apply_accepted_proposals


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _proposal_rows(payload: dict) -> list[dict]:
    return list(payload.get("proposals", [])) if "proposals" in payload else [payload]


def _write_recovery_plan(args: argparse.Namespace, manifest: dict) -> None:
    if not args.write_recovery_plan:
        return
    out = Path(args.output_dir)
    commands = out / "recovery_commands.sh"
    scene_source = args.scene_source or "<scene_source>"
    recovery_model = args.recovery_model or str(out / "recovery_model")
    commands.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Review application_manifest.json before running these commands.",
                f"# Applied mesh: {manifest['applied_mesh']}",
                f"CUDA_VISIBLE_DEVICES=<gpu_id> WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online \\",
                f"  python train.py -s {scene_source} -m {recovery_model} --eval --enable_wandb \\",
                "  --wandb_project spcarnet_meshprior --wandb_group meshprior_scene_recovery \\",
                "  --wandb_name meshprior_scene_recovery_review_required",
                "",
                f"python evaluate_geometry_colmap.py -s {scene_source} -m {recovery_model} --images images --eval \\",
                f"  --output {recovery_model}/geometry_eval_colmap/recovery_eval.json",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    proposals = _proposal_rows(_load_json(args.proposals))
    gate_report = _load_json(args.gate_report)
    manifest = apply_accepted_proposals(
        proposals=proposals,
        gate_report=gate_report,
        output_dir=args.output_dir,
        initial_mesh=args.initial_mesh or None,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "application_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (out / "application_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Accepted Proposal Application\n\n")
        f.write(f"status: `{manifest['status']}`\n\n")
        f.write(f"accepted: `{manifest['accepted_count']}`\n\n")
        f.write(f"applied: `{manifest['applied_count']}`\n\n")
        f.write(f"applied mesh: `{manifest['applied_mesh']}`\n\n")
        if manifest["warnings"]:
            f.write("## Warnings\n\n")
            for warning in manifest["warnings"]:
                f.write(f"- {warning}\n")
    _write_recovery_plan(args, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply accepted MeshPrior proposals to a safe mesh copy.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--initial_mesh", default="")
    parser.add_argument("--write_recovery_plan", action="store_true")
    parser.add_argument("--scene_source", default="")
    parser.add_argument("--recovery_model", default="")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
