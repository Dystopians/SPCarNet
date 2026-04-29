"""Lightweight diagnostic eval for CarNet checkpoints.

Runs a checkpoint over the test split with a configurable corruption override
and writes a summary JSON + per-patch CSV. Optionally records per-point
(r_in, r_out, ratio) arrays for residual-direction analysis.

Replaces the full ``ss3dm_prior.eval`` entrypoint for diagnostic sweeps: skips
all viz / gallery / sequence-map rendering so a single run takes ~1-2 minutes
on a single GPU instead of 10-15 minutes.

Profiles recognised by ``--profile``:
    default          - keep the checkpoint's embedded corruption config as-is
    zero             - disable every corruption (sanity check: clean -> clean)
    only_<name>      - enable exactly one of:
                       point_dropout | gaussian_jitter | normal_noise |
                       local_hole_mask | outlier_cluster | density_imbalance

Outputs under ``output_dir/eval_name/``:
    summary.json            - scalar metrics
    patch_predictions.csv   - per-patch metrics
    probe_points.npz        - (optional) per-point r_in/r_out/ratio arrays
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.eval import _collate_samples, _move_eval_batch_to_device, _load_model_from_checkpoint
from ss3dm_prior.metrics import (
    recon_chamfer_l1,
    recon_chamfer_l1_or_nan,
    recon_normal_cosine_or_nan,
    _cdist_fp32_safe,
)
from ss3dm_prior.utils.io import load_yaml


CLASSICAL_CORRUPTIONS = (
    "point_dropout",
    "gaussian_jitter",
    "normal_noise",
    "local_hole_mask",
    "outlier_cluster",
    "density_imbalance",
)


def build_corruption_config(base_config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return a new corruption dict per ``profile`` by tweaking enabled flags."""
    cfg = copy.deepcopy(base_config)
    if profile == "default":
        return cfg
    # Always disable lidar block (we do not re-enable it here).
    if "lidar" in cfg and isinstance(cfg["lidar"], dict):
        for key, value in cfg["lidar"].items():
            if isinstance(value, dict) and "enabled" in value:
                value["enabled"] = False
    if profile == "zero":
        for name in CLASSICAL_CORRUPTIONS:
            if name in cfg and isinstance(cfg[name], dict):
                cfg[name]["enabled"] = False
        return cfg
    if profile.startswith("only_"):
        keep = profile[len("only_") :]
        if keep not in CLASSICAL_CORRUPTIONS:
            raise ValueError(f"Unknown corruption name in profile: {profile!r}")
        for name in CLASSICAL_CORRUPTIONS:
            if name in cfg and isinstance(cfg[name], dict):
                cfg[name]["enabled"] = name == keep
        return cfg
    raise ValueError(f"Unknown profile: {profile!r}")


def records_for_subset(patch_index_path: Path, split_config_path: str | Path, subset: str) -> list[dict[str, Any]]:
    records = read_patch_index_jsonl(patch_index_path)
    split = load_yaml(split_config_path)
    key = f"{subset}_towns"
    towns = set(split.get(key, []))
    if not towns:
        raise ValueError(f"Split has no entry {key!r}.")
    selected = [record for record in records if str(record["town_id"]) in towns]
    if not selected:
        raise ValueError(f"No {subset} records found in split.")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic eval for CarNet checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch_cache_dir", required=True)
    parser.add_argument("--split_config", required=True)
    parser.add_argument("--output_dir", required=True, help="Root output directory.")
    parser.add_argument("--eval_name", required=True, help="Subfolder name under output_dir.")
    parser.add_argument("--profile", default="default", help="Corruption profile (default / zero / only_<name>).")
    parser.add_argument("--subset", default="test", choices=["test", "val", "train"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--probe_points",
        action="store_true",
        help="Dump per-point (r_in, r_out, ratio) arrays for residual direction analysis.",
    )
    parser.add_argument(
        "--probe_max_patches",
        type=int,
        default=20,
        help="Number of patches to record per-point data for when --probe_points is set.",
    )
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() / args.eval_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] checkpoint={checkpoint_path}", flush=True)
    print(f"[diag] output_dir={output_dir}", flush=True)
    print(f"[diag] profile={args.profile} subset={args.subset}", flush=True)

    device = torch.device(args.device)
    model, run_config = _load_model_from_checkpoint(checkpoint_path, device)
    base_corruption = copy.deepcopy(run_config["model_config"]["corruptions"])
    corruption_config = build_corruption_config(base_corruption, args.profile)

    # Persist the effective corruption so we can audit later.
    (output_dir / "corruption_config_used.json").write_text(
        json.dumps(corruption_config, indent=2, default=str)
    )

    records = records_for_subset(patch_cache_dir / "patch_index.jsonl", args.split_config, args.subset)
    print(f"[diag] {args.subset}_records={len(records)}", flush=True)

    dataset = TeacherPatchTrainDataset(
        patch_index_path=patch_cache_dir / "patch_index.jsonl",
        records=records,
        split_config=None,
        corruption_config=corruption_config,
        seed=int(run_config["train_config"].get("seed", 0)) + 2000,
        dynamic_corruption=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_samples,
    )

    patch_rows: list[dict[str, Any]] = []
    probe_r_in_list: list[np.ndarray] = []
    probe_r_out_list: list[np.ndarray] = []
    probe_patch_ids: list[str] = []
    probe_n_patches_captured = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=f"[{args.profile}]", dynamic_ncols=True):
            moved = _move_eval_batch_to_device(batch, device)
            outputs = model(
                corrupted_points=moved["corrupted_points"].float(),
                corrupted_normals=moved["corrupted_normals"].float(),
                observed_points=moved["observed_points"].float(),
                clean_points=moved["clean_points"].float(),
                clean_normals=moved["clean_normals"].float(),
                query_points_all=moved["query_points_all"].float(),
                visible_clean_points=moved.get("visible_clean_points"),
                visible_clean_normals=moved.get("visible_clean_normals"),
                hidden_clean_points=moved.get("hidden_clean_points"),
                hidden_clean_normals=moved.get("hidden_clean_normals"),
                sample_latent_candidates_k=0,
                stochastic_flow_steps=None,
            )
            batch_size = moved["clean_points"].shape[0]
            for sample_idx in range(batch_size):
                clean_points = moved["clean_points"][sample_idx : sample_idx + 1]
                corrupted_points = moved["corrupted_points"][sample_idx : sample_idx + 1]
                recon_points = outputs["recon_points"][sample_idx : sample_idx + 1]
                recon_normals = outputs["recon_normals"][sample_idx : sample_idx + 1]
                clean_normals = moved["clean_normals"][sample_idx : sample_idx + 1]

                visible_clean_points = _optional(batch.get("visible_clean_points"), sample_idx, device)
                hidden_clean_points = _optional(batch.get("hidden_clean_points"), sample_idx, device)
                visible_clean_normals = _optional(batch.get("visible_clean_normals"), sample_idx, device)

                chamfer_before = float(recon_chamfer_l1(corrupted_points, clean_points).detach().cpu())
                chamfer_after = float(recon_chamfer_l1(recon_points, clean_points).detach().cpu())
                visible_chamfer = (
                    recon_chamfer_l1_or_nan(recon_points, visible_clean_points)
                    if visible_clean_points is not None
                    else float("nan")
                )
                hidden_chamfer = (
                    recon_chamfer_l1_or_nan(recon_points, hidden_clean_points)
                    if hidden_clean_points is not None
                    else float("nan")
                )
                visible_normal = (
                    recon_normal_cosine_or_nan(recon_points, recon_normals, visible_clean_points, visible_clean_normals)
                    if visible_clean_points is not None and visible_clean_normals is not None
                    else float("nan")
                )

                patch_rows.append(
                    {
                        "patch_id": batch["patch_id"][sample_idx],
                        "town_id": batch["town_id"][sample_idx],
                        "sequence_id": batch["sequence_id"][sample_idx],
                        "chamfer_before": chamfer_before,
                        "chamfer_after": chamfer_after,
                        "denoise_gain": chamfer_before - chamfer_after,
                        "visible_recon_chamfer_l1": visible_chamfer,
                        "hidden_completion_chamfer_l1": hidden_chamfer,
                        "visible_recon_normal_cosine": visible_normal,
                    }
                )

                if args.probe_points and probe_n_patches_captured < args.probe_max_patches:
                    # Per-point nearest-neighbor distances from corrupted / recon to clean.
                    r_in = _per_point_distance_to_set(corrupted_points[0], clean_points[0])
                    r_out = _per_point_distance_to_set(recon_points[0], clean_points[0])
                    probe_r_in_list.append(r_in.cpu().numpy())
                    probe_r_out_list.append(r_out.cpu().numpy())
                    probe_patch_ids.append(batch["patch_id"][sample_idx])
                    probe_n_patches_captured += 1

    # Summary
    def _mean_col(key: str) -> float:
        values = [row[key] for row in patch_rows if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "profile": args.profile,
        "subset": args.subset,
        "num_patches": len(patch_rows),
        "recon_chamfer_l1": _mean_col("chamfer_after"),
        "chamfer_before": _mean_col("chamfer_before"),
        "denoise_gain_chamfer": _mean_col("denoise_gain"),
        "visible_recon_chamfer_l1": _mean_col("visible_recon_chamfer_l1"),
        "hidden_completion_chamfer_l1": _mean_col("hidden_completion_chamfer_l1"),
        "visible_recon_normal_cosine": _mean_col("visible_recon_normal_cosine"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Per-patch CSV
    csv_path = output_dir / "patch_predictions.csv"
    if patch_rows:
        keys = list(patch_rows[0].keys())
        with csv_path.open("w") as handle:
            handle.write(",".join(keys) + "\n")
            for row in patch_rows:
                handle.write(",".join(str(row[key]) for key in keys) + "\n")

    if args.probe_points and probe_r_in_list:
        probe_path = output_dir / "probe_points.npz"
        np.savez_compressed(
            probe_path,
            r_in=np.stack(probe_r_in_list, axis=0),
            r_out=np.stack(probe_r_out_list, axis=0),
            patch_ids=np.asarray(probe_patch_ids),
        )
        print(f"[diag] saved probe arrays: {probe_path}", flush=True)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[diag] summary: {summary_path}", flush=True)
    print(f"[diag] csv:     {csv_path}", flush=True)
    return 0


def _optional(value, index, device):
    if value is None:
        return None
    if isinstance(value, list):
        if index >= len(value) or not isinstance(value[index], torch.Tensor):
            return None
        return value[index].to(device).unsqueeze(0)
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return value.to(device).unsqueeze(0)
        return value[index : index + 1].to(device)
    return None


def _per_point_distance_to_set(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """For each point in ``x`` (N, 3), the L2 distance to the nearest point in ``y`` (M, 3)."""
    if x.ndim == 2:
        x2 = x.unsqueeze(0)
    else:
        x2 = x
    if y.ndim == 2:
        y2 = y.unsqueeze(0)
    else:
        y2 = y
    d = _cdist_fp32_safe(x2, y2, p=2)
    return d.min(dim=2).values.squeeze(0)


if __name__ == "__main__":
    raise SystemExit(main())
