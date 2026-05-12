#!/usr/bin/env python3
"""Blend a checkpoint-level residual delta into a source MeshSplat model.

The script materializes

    output = source + gamma * (candidate - source)

for same-topology checkpoint tensors.  It is intentionally small and auditable:
the blend is applied only to explicit tensor keys, defaults to appearance SH
features, and writes a policy-compatible audit JSON that downstream train-val
gates can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import copy_model_metadata, checkpoint_path, validate_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--candidate_model", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--blend_keys",
        nargs="+",
        default=["features_dc", "features_rest"],
        help="Checkpoint tensor keys to blend. Default keeps geometry fixed.",
    )
    parser.add_argument(
        "--allow_topology_copy",
        action="store_true",
        help="Allow identical face tensors to be copied from the candidate. "
        "This is still same-topology; it does not support extra local vertices.",
    )
    return parser.parse_args()


def _shape(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return [int(x) for x in value.shape]
    return None


def main() -> int:
    args = parse_args()
    gamma = float(args.gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("--gamma must be in [0, 1]")

    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    candidate_checkpoint = checkpoint_path(args.candidate_model, args.iteration)
    output_checkpoint = (
        args.output_model
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "point_cloud_state_dict.pt"
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    source = torch.load(source_checkpoint, map_location="cpu")
    candidate = torch.load(candidate_checkpoint, map_location="cpu")
    blend_keys = [str(k) for k in args.blend_keys]

    if "_triangle_indices" not in source or "_triangle_indices" not in candidate:
        raise KeyError("both checkpoints must contain _triangle_indices")
    source_faces = source["_triangle_indices"].detach().cpu()
    candidate_faces = candidate["_triangle_indices"].detach().cpu()
    if tuple(source_faces.shape) != tuple(candidate_faces.shape):
        raise ValueError("topology mismatch: face tensor shapes differ")
    if not torch.equal(source_faces, candidate_faces):
        raise ValueError(
            "topology mismatch: face tensors differ; this script only handles same-topology blends"
        )

    out: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for key, value in source.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        src = value.detach().cpu()
        cand = candidate.get(key)
        if key in blend_keys:
            if cand is None or not torch.is_tensor(cand):
                raise KeyError(f"candidate checkpoint is missing tensor key {key!r}")
            cand_cpu = cand.detach().cpu()
            if tuple(src.shape) != tuple(cand_cpu.shape):
                raise ValueError(f"shape mismatch for blended key {key!r}")
            delta = cand_cpu.to(dtype=torch.float32) - src.to(dtype=torch.float32)
            blended = src.to(dtype=torch.float32) + gamma * delta
            out[key] = blended.to(dtype=src.dtype)
            stats[key] = {
                "shape": _shape(src),
                "delta_abs_mean": float(delta.abs().mean().item()) if delta.numel() else 0.0,
                "delta_abs_max": float(delta.abs().max().item()) if delta.numel() else 0.0,
                "blended_abs_mean": float((gamma * delta).abs().mean().item()) if delta.numel() else 0.0,
                "blended_abs_max": float((gamma * delta).abs().max().item()) if delta.numel() else 0.0,
            }
        elif key == "_triangle_indices" and bool(args.allow_topology_copy):
            out[key] = candidate_faces.to(dtype=src.dtype).clone()
        else:
            out[key] = src.clone()

    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    torch.save(out, output_checkpoint)

    audit = {
        "operator": "checkpoint_delta_blend",
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "candidate_model": str(args.candidate_model),
        "candidate_checkpoint": str(candidate_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "gamma": gamma,
        "blend_keys": blend_keys,
        "accepted": gamma > 0.0,
        "no_op_copy": gamma == 0.0,
        "topology_same": True,
        "topology_after": {
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "blend_stats": stats,
    }
    (args.output_model / "checkpoint_delta_blend_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Checkpoint Delta Blend Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- source model: `{args.source_model}`",
        f"- candidate model: `{args.candidate_model}`",
        f"- output model: `{args.output_model}`",
        f"- iteration: `{int(args.iteration)}`",
        f"- gamma: `{gamma:.6f}`",
        f"- blend keys: `{', '.join(blend_keys)}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- degenerate faces: `{degenerate}`",
        f"- invalid indices: `{invalid}`",
        "",
        "This blend uses no held-out test residuals. A separate train-heldout render gate must certify it.",
    ]
    (args.output_model / "checkpoint_delta_blend_audit.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_model": str(args.output_model), "gamma": gamma}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
