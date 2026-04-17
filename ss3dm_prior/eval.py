"""Standardized evaluation entrypoint for SS3DM prior checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import warnings

from ss3dm_prior.utils.cuda_env import configure_isolated_mps_pipe_if_needed

configured_mps_pipe = configure_isolated_mps_pipe_if_needed()

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.metrics import (
    free_space_violation_rate,
    intrinsic_difficulty_spearman,
    occupancy_iou_visible,
    point_defect_mae,
    prototype_usage_entropy,
    recon_chamfer_l1,
    recon_normal_cosine,
    retrieval_top1_cross_sequence,
    retrieval_top1_nonself,
    retrieval_top1_self_aligned,
    retrieval_top5_nonself,
    retrieval_top5_self_aligned,
    score_spearman,
)
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser
from ss3dm_prior.reporting import (
    aggregate_global_metrics,
    aggregate_per_group,
    write_csv,
    write_report_md,
    write_summary_json,
)
from ss3dm_prior.tools.audit_run_protocol import audit_run_protocol
from ss3dm_prior.utils.io import load_yaml
from ss3dm_prior.viz.render_patch_panels import (
    render_hybrid_reconstruction_panel,
    render_patch_triptych,
    render_prototype_usage_gallery,
    render_visibility_panel,
)
from ss3dm_prior.viz.render_textured_mesh_panels import render_textured_whole_mesh_triptych
from ss3dm_prior.viz.render_sequence_maps import render_sequence_visibility_map


def _collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    tensor_keys = {
        "clean_points",
        "clean_normals",
        "observed_points",
        "corrupted_points",
        "corrupted_normals",
        "point_defect_target",
        "corruption_score_target",
        "patch_center_world",
        "surface_query_points",
        "surface_query_labels",
        "free_query_points",
        "free_query_labels",
        "unknown_query_points",
        "query_points_all",
        "query_labels_all",
        "query_ignore_mask",
        "camera_support_count",
        "lidar_support_count",
        "visible_surface_fraction",
        "free_space_fraction",
        "unknown_fraction",
        "intrinsic_patch_difficulty_target",
    }
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        collated[key] = default_collate(values) if key in tensor_keys else values
    return collated


class _NullRun:
    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None

    def save(self, *_args, **_kwargs) -> None:
        return None


def _init_wandb(eval_name: str, output_dir: Path, project: str, mode: str):
    if mode == "disabled":
        return None, _NullRun()
    try:
        import wandb  # type: ignore
    except Exception as exc:
        warnings.warn(f"wandb unavailable during eval, disabling logging: {exc}", stacklevel=2)
        return None, _NullRun()
    run = wandb.init(project=project, name=eval_name, mode=mode, dir=str(output_dir), job_type="eval", reinit=True)
    return wandb, run


def _safe_float(value: Any, *, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(scalar):
        return default
    return scalar


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _free_query_violation_scores(
    query_occupancy_logits: torch.Tensor | None,
    query_labels_all: torch.Tensor,
    query_ignore_mask: torch.Tensor,
    expected_count: int,
) -> np.ndarray | None:
    if query_occupancy_logits is None or expected_count <= 0:
        return None
    logits = query_occupancy_logits.reshape(-1).detach().cpu()
    labels = query_labels_all.reshape(-1).detach().cpu()
    ignore = query_ignore_mask.reshape(-1).detach().cpu().bool()
    free_mask = (labels <= 0.5) & (~ignore)
    free_scores = torch.sigmoid(logits[free_mask]).numpy()
    if free_scores.shape[0] == expected_count:
        return free_scores.astype(np.float32)
    if free_scores.shape[0] > expected_count:
        return free_scores[:expected_count].astype(np.float32)
    return None


def _select_sequence_ids_for_visualization(per_sequence_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    if not per_sequence_rows:
        return []
    selected: list[tuple[str, str]] = []
    selectors = [
        ("hardest", max, "mean_intrinsic_difficulty"),
        ("best_gain", max, "mean_denoise_gain"),
        ("worst_gain", min, "mean_denoise_gain"),
    ]
    seen: set[str] = set()
    for label, reducer, key in selectors:
        valid_rows = [row for row in per_sequence_rows if np.isfinite(float(row.get(key, float("nan"))))]
        if not valid_rows:
            continue
        chosen = reducer(valid_rows, key=lambda row: float(row[key]))
        sequence_id = str(chosen["sequence_id"])
        if sequence_id in seen:
            continue
        seen.add(sequence_id)
        selected.append((label, sequence_id))
    if not selected:
        first = per_sequence_rows[0]
        selected.append(("sequence", str(first["sequence_id"])))
    target_count = min(3, len(per_sequence_rows))
    if len(selected) < target_count:
        for row in sorted(per_sequence_rows, key=lambda item: int(item.get("patch_count", 0)), reverse=True):
            sequence_id = str(row["sequence_id"])
            if sequence_id in seen:
                continue
            seen.add(sequence_id)
            selected.append((f"additional_{len(selected) + 1}", sequence_id))
            if len(selected) >= target_count:
                break
    return selected


def _select_gallery_rows(
    patch_rows: list[dict[str, Any]],
    *,
    examples_per_gallery: int,
) -> list[tuple[str, dict[str, Any] | None]]:
    selectors: list[tuple[str, str, bool]] = [
        ("best_hidden_completion", "hidden_completion_score", True),
        ("worst_free_space_violation", "free_space_violation_rate", True),
        ("largest_intrinsic_score_error", "intrinsic_difficulty_abs_error", True),
    ]
    chosen: list[tuple[str, dict[str, Any] | None]] = []
    seen_patch_ids: set[str] = set()
    for gallery_name, metric_key, reverse in selectors:
        valid_rows = [
            row
            for row in patch_rows
            if np.isfinite(float(row.get(metric_key, float("nan")))) and str(row.get("patch_id", "")) not in seen_patch_ids
        ]
        ranked_rows = sorted(valid_rows, key=lambda row: float(row[metric_key]), reverse=reverse)
        for rank_idx, row in enumerate(ranked_rows[: max(1, examples_per_gallery)], start=1):
            patch_id = str(row["patch_id"])
            seen_patch_ids.add(patch_id)
            chosen.append((f"{gallery_name}__rank{rank_idx}", row))
    return chosen


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SS3DM prior checkpoint on a test split.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--manifest_path", required=True, help="Manifest path metadata.")
    parser.add_argument("--patch_cache_dir", required=True, help="Teacher patch cache root.")
    parser.add_argument("--split_config", required=True, help="Split YAML file.")
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--eval_name", required=True, help="Evaluation name.")
    parser.add_argument("--wandb_project", default="ss3dm_prior_eval", help="wandb project.")
    parser.add_argument("--wandb_mode", default="offline", help="wandb mode.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Execution device for eval. Default to cpu to avoid flaky CUDA probing during report/render generation.",
    )
    parser.add_argument(
        "--fail_on_protocol_warning",
        action="store_true",
        help="Raise an error if the saved run metadata indicates split-protocol warnings.",
    )
    parser.add_argument(
        "--examples_per_gallery",
        type=int,
        default=2,
        help="Representative examples to render per gallery category.",
    )
    return parser


def _records_for_test_split(patch_index_path: Path, split_config_path: str | Path) -> list[dict[str, Any]]:
    records = read_patch_index_jsonl(patch_index_path)
    split = load_yaml(split_config_path)
    test_towns = set(split.get("test_towns", []))
    selected = [record for record in records if str(record["town_id"]) in test_towns]
    if not selected:
        raise ValueError(f"No test patch records found for test_towns={sorted(test_towns)} in {patch_index_path}")
    return selected


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[LocalPatchDenoiser, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = payload["run_config"]
    model_cfg = run_config["model_config"]
    model = LocalPatchDenoiser(**model_cfg["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, run_config


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)
    if configured_mps_pipe:
        _print_progress(f"[eval] using isolated CUDA_MPS_PIPE_DIRECTORY={configured_mps_pipe}")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() / args.eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_panels_dir = output_dir / "patch_panels"
    sequence_maps_dir = output_dir / "sequence_maps"
    prototype_dir = output_dir / "prototype_gallery"
    for directory in (patch_panels_dir, sequence_maps_dir, prototype_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _print_progress(f"[eval] output_dir={output_dir}")
    wandb_module, wandb_run = _init_wandb(args.eval_name, output_dir, args.wandb_project, args.wandb_mode)
    if wandb_run is not None and hasattr(wandb_run, "url"):
        run_url = getattr(wandb_run, "url", None)
        if isinstance(run_url, str) and run_url:
            _print_progress(f"[eval] wandb_url={run_url}")

    _print_progress("[eval] auditing checkpoint protocol")
    protocol_audit = audit_run_protocol(checkpoint_path)
    for protocol_warning in protocol_audit["protocol_warnings"]:
        warnings.warn(f"Protocol audit warning: {protocol_warning}", stacklevel=2)
    if args.fail_on_protocol_warning and protocol_audit["protocol_warnings"]:
        raise ValueError(
            "Protocol audit failed: "
            + "; ".join(str(item) for item in protocol_audit["protocol_warnings"])
        )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        _print_progress("[eval] probing device=auto")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_progress(f"[eval] loading checkpoint on device={device}")
    model, run_config = _load_model_from_checkpoint(checkpoint_path, device)
    _print_progress("[eval] reading test records")
    records = _records_for_test_split(patch_cache_dir / "patch_index.jsonl", args.split_config)
    _print_progress(f"[eval] test_records={len(records)}")
    dataset = TeacherPatchTrainDataset(
        patch_index_path=patch_cache_dir / "patch_index.jsonl",
        records=records,
        split_config=None,
        corruption_config=run_config["model_config"]["corruptions"],
        seed=int(run_config["train_config"].get("seed", 0)) + 2000,
        dynamic_corruption=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(run_config["train_config"].get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_samples,
    )
    _print_progress(f"[eval] dataloader_batches={len(loader)}")

    patch_rows: list[dict[str, Any]] = []
    qualitative_paths: list[Path] = []
    sequence_map_bank: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    clean_latents: list[torch.Tensor] = []
    corrupted_latents: list[torch.Tensor] = []
    clean_bank_patch_ids: list[str] = []
    clean_bank_sequence_ids: list[str] = []
    prototype_code_indices: list[int] = []
    prototype_examples_by_code: dict[int, dict[str, object]] = {}

    with torch.no_grad():
        eval_iter = tqdm(loader, total=len(loader), desc="eval batches", dynamic_ncols=True)
        for batch_idx, batch in enumerate(eval_iter, start=1):
            moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                corrupted_points=moved["corrupted_points"].float(),
                corrupted_normals=moved["corrupted_normals"].float(),
                observed_points=moved["observed_points"].float(),
                clean_points=moved["clean_points"].float(),
                clean_normals=moved["clean_normals"].float(),
                query_points_all=moved["query_points_all"].float(),
            )
            point_defect_pred_raw = torch.expm1(torch.clamp(outputs["point_defect_pred"], min=0.0))
            patch_score_pred_raw = torch.expm1(torch.clamp(outputs["patch_score_pred"], min=0.0))
            corrupted_latents.append(outputs["retrieval_embedding"].detach().cpu())
            if outputs["clean_retrieval_embedding"] is not None:
                clean_latents.append(outputs["clean_retrieval_embedding"].detach().cpu())
            if outputs.get("code_indices") is not None:
                prototype_code_indices.extend(int(index) for index in outputs["code_indices"].detach().cpu().tolist())

            batch_size = moved["clean_points"].shape[0]
            for sample_idx in range(batch_size):
                patch_id = batch["patch_id"][sample_idx]
                town_id = batch["town_id"][sample_idx]
                sequence_id = batch["sequence_id"][sample_idx]
                clean_points = moved["clean_points"][sample_idx : sample_idx + 1]
                corrupted_points = moved["corrupted_points"][sample_idx : sample_idx + 1]
                recon_points = outputs["recon_points"][sample_idx : sample_idx + 1]
                recon_normals = outputs["recon_normals"][sample_idx : sample_idx + 1]
                clean_normals = moved["clean_normals"][sample_idx : sample_idx + 1]

                chamfer_before = float(recon_chamfer_l1(corrupted_points, clean_points).detach().cpu())
                chamfer_after = float(recon_chamfer_l1(recon_points, clean_points).detach().cpu())
                normal_cos = float(recon_normal_cosine(recon_points, recon_normals, clean_points, clean_normals).detach().cpu())
                denoise_gain = chamfer_before - chamfer_after
                pred_score = float(patch_score_pred_raw[sample_idx].detach().cpu())
                target_score = float(moved["corruption_score_target"][sample_idx].detach().cpu())
                defect_mae = float(
                    point_defect_mae(
                        point_defect_pred_raw[sample_idx],
                        moved["point_defect_target"][sample_idx],
                    )
                    .detach()
                    .cpu()
                )
                score_abs_error = abs(pred_score - target_score)
                patch_center_world = moved["patch_center_world"][sample_idx].detach().cpu().numpy()
                camera_support = _safe_float(moved["camera_support_count"][sample_idx].detach().cpu())
                lidar_support = _safe_float(moved["lidar_support_count"][sample_idx].detach().cpu())
                visible_surface_fraction = _safe_float(moved["visible_surface_fraction"][sample_idx].detach().cpu())
                free_space_fraction = _safe_float(moved["free_space_fraction"][sample_idx].detach().cpu())
                unknown_fraction = _safe_float(moved["unknown_fraction"][sample_idx].detach().cpu())
                intrinsic_target = _safe_float(moved["intrinsic_patch_difficulty_target"][sample_idx].detach().cpu())
                intrinsic_pred = (
                    _safe_float(outputs["intrinsic_difficulty_pred"][sample_idx].detach().cpu())
                    if outputs.get("intrinsic_difficulty_pred") is not None
                    else float("nan")
                )
                intrinsic_abs_error = abs(intrinsic_pred - intrinsic_target) if np.isfinite(intrinsic_pred) and np.isfinite(intrinsic_target) else float("nan")
                query_logits = (
                    outputs["query_occupancy_logits"][sample_idx : sample_idx + 1]
                    if outputs.get("query_occupancy_logits") is not None
                    else None
                )
                query_labels_all = moved["query_labels_all"][sample_idx : sample_idx + 1]
                query_ignore_mask = moved["query_ignore_mask"][sample_idx : sample_idx + 1]
                occupancy_iou_value = (
                    occupancy_iou_visible(
                        query_logits.detach().cpu(),
                        query_labels_all.detach().cpu(),
                        query_ignore_mask.detach().cpu(),
                    )
                    if query_logits is not None and int(query_labels_all.numel()) > 0
                    else float("nan")
                )
                free_space_violation_value = (
                    free_space_violation_rate(
                        query_logits.detach().cpu(),
                        query_labels_all.detach().cpu(),
                        query_ignore_mask.detach().cpu(),
                    )
                    if query_logits is not None and int(query_labels_all.numel()) > 0
                    else float("nan")
                )
                code_index = (
                    int(outputs["code_indices"][sample_idx].detach().cpu().item())
                    if outputs.get("code_indices") is not None
                    else -1
                )
                visible_support_count = camera_support + lidar_support

                row = {
                    "patch_id": patch_id,
                    "town_id": town_id,
                    "sequence_id": sequence_id,
                    "patch_center_x": float(patch_center_world[0]),
                    "patch_center_y": float(patch_center_world[1]),
                    "patch_center_z": float(patch_center_world[2]),
                    "corruption_score_target": target_score,
                    "patch_score_pred": pred_score,
                    "score_abs_error": score_abs_error,
                    "recon_chamfer_l1": chamfer_after,
                    "recon_normal_cosine": normal_cos,
                    "denoise_gain_chamfer": denoise_gain,
                    "point_defect_mae": defect_mae,
                    "chamfer_before": chamfer_before,
                    "chamfer_after": chamfer_after,
                    "occupancy_iou_visible": occupancy_iou_value,
                    "free_space_violation_rate": free_space_violation_value,
                    "visible_support_count": visible_support_count,
                    "camera_support_count": camera_support,
                    "lidar_support_count": lidar_support,
                    "visible_surface_fraction": visible_surface_fraction,
                    "free_space_fraction": free_space_fraction,
                    "unknown_fraction": unknown_fraction,
                    "intrinsic_difficulty_target": intrinsic_target,
                    "intrinsic_difficulty_pred": intrinsic_pred,
                    "intrinsic_difficulty_abs_error": intrinsic_abs_error,
                    "code_index": code_index,
                    "hidden_completion_score": max(denoise_gain, 0.0) * max(unknown_fraction, 0.0),
                }
                patch_rows.append(row)
                sequence_map_bank[sequence_id]["patch_centers"].append(patch_center_world)
                sequence_map_bank[sequence_id]["actual_gains"].append(denoise_gain)
                sequence_map_bank[sequence_id]["visible_surface_fraction"].append(visible_surface_fraction)
                sequence_map_bank[sequence_id]["free_space_fraction"].append(free_space_fraction)
                sequence_map_bank[sequence_id]["intrinsic_targets"].append(intrinsic_target)
                clean_bank_patch_ids.append(patch_id)
                clean_bank_sequence_ids.append(sequence_id)
                if code_index >= 0 and code_index not in prototype_examples_by_code:
                    prototype_examples_by_code[code_index] = {
                        "patch_id": patch_id,
                        "code_index": code_index,
                        "clean_points": clean_points[0].detach().cpu().numpy(),
                        "intrinsic_pred": round(intrinsic_pred, 4) if np.isfinite(intrinsic_pred) else "nan",
                        "intrinsic_target": round(intrinsic_target, 4) if np.isfinite(intrinsic_target) else "nan",
                    }
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
                eval_iter.set_postfix(samples=len(patch_rows))

    if clean_latents and corrupted_latents:
        all_clean = torch.cat(clean_latents, dim=0)
        all_corrupted = torch.cat(corrupted_latents, dim=0)
        retrieval_metrics = {
            "retrieval_top1_self_aligned": retrieval_top1_self_aligned(all_corrupted, all_clean),
            "retrieval_top5_self_aligned": retrieval_top5_self_aligned(all_corrupted, all_clean),
            "retrieval_top1_nonself": retrieval_top1_nonself(
                all_corrupted,
                all_clean,
                query_patch_ids=clean_bank_patch_ids,
                target_patch_ids=clean_bank_patch_ids,
                query_sequence_ids=clean_bank_sequence_ids,
                target_sequence_ids=clean_bank_sequence_ids,
            ),
            "retrieval_top5_nonself": retrieval_top5_nonself(
                all_corrupted,
                all_clean,
                query_patch_ids=clean_bank_patch_ids,
                target_patch_ids=clean_bank_patch_ids,
                query_sequence_ids=clean_bank_sequence_ids,
                target_sequence_ids=clean_bank_sequence_ids,
            ),
            "retrieval_top1_cross_sequence": retrieval_top1_cross_sequence(
                all_corrupted,
                all_clean,
                query_patch_ids=clean_bank_patch_ids,
                target_patch_ids=clean_bank_patch_ids,
                query_sequence_ids=clean_bank_sequence_ids,
                target_sequence_ids=clean_bank_sequence_ids,
            ),
        }
    else:
        warnings.warn("Retrieval embeddings unavailable during eval.", stacklevel=2)
        retrieval_metrics = {
            "retrieval_top1_self_aligned": float("nan"),
            "retrieval_top5_self_aligned": float("nan"),
            "retrieval_top1_nonself": float("nan"),
            "retrieval_top5_nonself": float("nan"),
            "retrieval_top1_cross_sequence": float("nan"),
        }

    if len(patch_rows) >= 2:
        score_spearman_value = score_spearman(
            torch.tensor([row["patch_score_pred"] for row in patch_rows], dtype=torch.float32),
            torch.tensor([row["corruption_score_target"] for row in patch_rows], dtype=torch.float32),
        )
    else:
        warnings.warn("score_spearman unavailable during eval due to too few samples.", stacklevel=2)
        score_spearman_value = float("nan")
    retrieval_metrics["score_spearman"] = score_spearman_value
    if len(patch_rows) >= 2:
        intrinsic_spearman_value = intrinsic_difficulty_spearman(
            torch.tensor([row["intrinsic_difficulty_pred"] for row in patch_rows], dtype=torch.float32),
            torch.tensor([row["intrinsic_difficulty_target"] for row in patch_rows], dtype=torch.float32),
        )
    else:
        intrinsic_spearman_value = float("nan")
    retrieval_metrics["intrinsic_difficulty_spearman"] = intrinsic_spearman_value
    retrieval_metrics["prototype_usage_entropy"] = (
        prototype_usage_entropy(torch.tensor(prototype_code_indices, dtype=torch.int64))
        if prototype_code_indices
        else float("nan")
    )

    summary = aggregate_global_metrics(
        patch_rows,
        {
            "retrieval_top1_self_aligned": retrieval_metrics["retrieval_top1_self_aligned"],
            "retrieval_top5_self_aligned": retrieval_metrics["retrieval_top5_self_aligned"],
            "retrieval_top1_nonself": retrieval_metrics["retrieval_top1_nonself"],
            "retrieval_top5_nonself": retrieval_metrics["retrieval_top5_nonself"],
            "retrieval_top1_cross_sequence": retrieval_metrics["retrieval_top1_cross_sequence"],
            "score_spearman": retrieval_metrics["score_spearman"],
            "intrinsic_difficulty_spearman": retrieval_metrics["intrinsic_difficulty_spearman"],
            "prototype_usage_entropy": retrieval_metrics["prototype_usage_entropy"],
        },
    )
    summary.update(protocol_audit)
    summary_path = write_summary_json(output_dir / "metrics_summary.json", summary)
    per_town_rows = aggregate_per_group(patch_rows, group_key="town_id")
    per_sequence_rows = aggregate_per_group(patch_rows, group_key="sequence_id")
    write_csv(
        output_dir / "metrics_per_town.csv",
        per_town_rows,
        [
            "town_id",
            "patch_count",
            "sequence_count",
            "recon_chamfer_l1",
            "recon_normal_cosine",
            "denoise_gain_chamfer",
            "intrinsic_difficulty_mae",
            "occupancy_iou_visible",
            "free_space_violation_rate",
            "score_mae",
            "point_defect_mae",
            "mean_visible_support",
            "mean_intrinsic_difficulty",
            "prototype_usage_entropy",
        ],
    )
    write_csv(
        output_dir / "metrics_per_sequence.csv",
        per_sequence_rows,
        [
            "sequence_id",
            "town_id",
            "patch_count",
            "mean_corruption_severity",
            "mean_denoise_gain",
            "denoise_gain_chamfer",
            "recon_chamfer_l1",
            "recon_normal_cosine",
            "intrinsic_difficulty_mae",
            "occupancy_iou_visible",
            "free_space_violation_rate",
            "score_mae",
            "point_defect_mae",
            "mean_visible_support",
            "mean_intrinsic_difficulty",
            "prototype_usage_entropy",
        ],
    )
    write_csv(
        output_dir / "patch_predictions.csv",
        patch_rows,
        list(patch_rows[0].keys()) if patch_rows else ["patch_id"],
    )

    patch_dataset = {record["patch_id"]: dataset[idx] for idx, record in enumerate(dataset.records)}
    gallery_specs = _select_gallery_rows(
        patch_rows,
        examples_per_gallery=max(1, int(args.examples_per_gallery)),
    )
    _print_progress(f"[eval] rendering representative panels count={len(gallery_specs)}")
    for gallery_name, row in tqdm(gallery_specs, desc="render panels", dynamic_ncols=True):
        if row is None:
            continue
        sample = patch_dataset[row["patch_id"]]
        sample_device = {key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        with torch.no_grad():
            outputs = model(
                corrupted_points=sample_device["corrupted_points"].float(),
                corrupted_normals=sample_device["corrupted_normals"].float(),
                observed_points=sample_device["observed_points"].float(),
                clean_points=sample_device["clean_points"].float(),
                clean_normals=sample_device["clean_normals"].float(),
                query_points_all=sample_device["query_points_all"].float(),
            )
        code_index = int(outputs["code_indices"][0].detach().cpu().item()) if outputs.get("code_indices") is not None else -1
        prototype_summary_lines = [
            f"prototype_code: {code_index}",
            f"legacy_top1: {_safe_float(summary.get('retrieval_top1_self_aligned')):.4f}",
            f"nonself_top1: {_safe_float(summary.get('retrieval_top1_nonself')):.4f}",
        ]
        triptych_path = render_patch_triptych(
            corrupted_points=sample["corrupted_points"].numpy(),
            recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
            clean_points=sample["clean_points"].numpy(),
            info_lines=[
                f"gallery: {gallery_name}",
                f"town: {row['town_id']}",
                f"sequence: {row['sequence_id']}",
                f"patch: {row['patch_id']}",
                f"view: corrupt / repaired / ground_truth",
                f"gain: {row['denoise_gain_chamfer']:.4f}",
                f"chamfer_before: {_safe_float(row.get('chamfer_before')):.4f}",
                f"chamfer_after: {_safe_float(row.get('chamfer_after')):.4f}",
            ],
            output_path=patch_panels_dir / f"{gallery_name}__{row['patch_id']}__triptych.png",
        )
        qualitative_paths.append(triptych_path)
        source_mesh_path = str(sample.get("patch_metadata", {}).get("source_mesh_path", "")).strip()
        if source_mesh_path:
            try:
                textured_triptych_path = render_textured_whole_mesh_triptych(
                    source_mesh_path=source_mesh_path,
                    clean_points=sample["clean_points"].numpy(),
                    corrupted_points=sample["corrupted_points"].numpy(),
                    recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
                    info_lines=[
                        f"gallery: {gallery_name}",
                        f"patch: {row['patch_id']}",
                        f"asset: {sample.get('patch_metadata', {}).get('asset_id', 'n/a')}",
                        "view: hero/front/top-down/low-angle x corrupt/repaired/ground_truth",
                        f"gain: {row['denoise_gain_chamfer']:.4f}",
                        f"chamfer_before: {_safe_float(row.get('chamfer_before')):.4f}",
                        f"chamfer_after: {_safe_float(row.get('chamfer_after')):.4f}",
                    ],
                    output_path=patch_panels_dir / f"{gallery_name}__{row['patch_id']}__textured_triptych.png",
                )
                qualitative_paths.append(textured_triptych_path)
            except Exception as exc:
                warnings.warn(
                    f"Failed to render textured whole-mesh triptych for patch {row['patch_id']}: {exc}",
                    stacklevel=2,
                )
        hybrid_path = render_hybrid_reconstruction_panel(
            corrupted_points=sample["corrupted_points"].numpy(),
            recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
            clean_points=sample["clean_points"].numpy(),
            free_query_points=sample["free_query_points"].numpy(),
            free_query_violation_scores=_free_query_violation_scores(
                outputs.get("query_occupancy_logits"),
                sample_device["query_labels_all"],
                sample_device["query_ignore_mask"],
                expected_count=int(sample["free_query_points"].shape[0]),
            ),
            intrinsic_pred=_safe_float(
                outputs["intrinsic_difficulty_pred"][0].detach().cpu()
                if outputs.get("intrinsic_difficulty_pred") is not None
                else None
            ),
            intrinsic_target=_safe_float(sample["intrinsic_patch_difficulty_target"].item()),
            prototype_summary_lines=prototype_summary_lines,
            info_lines=[
                f"gallery: {gallery_name}",
                f"town: {row['town_id']}",
                f"sequence: {row['sequence_id']}",
                f"patch: {row['patch_id']}",
                f"gain: {row['denoise_gain_chamfer']:.4f}",
                f"free_violation: {_safe_float(row.get('free_space_violation_rate')):.4f}",
                f"intrinsic_abs_err: {_safe_float(row.get('intrinsic_difficulty_abs_error')):.4f}",
            ],
            output_path=patch_panels_dir / f"{gallery_name}__{row['patch_id']}.png",
        )
        qualitative_paths.append(hybrid_path)
        if gallery_name == "best_hidden_completion":
            visibility_path = render_visibility_panel(
                clean_points=sample["clean_points"].numpy(),
                observed_points=sample["observed_points"].numpy(),
                surface_query_points=sample["surface_query_points"].numpy(),
                free_query_points=sample["free_query_points"].numpy(),
                unknown_query_points=sample["unknown_query_points"].numpy(),
                info_lines=[
                    f"gallery: {gallery_name}",
                    f"patch: {row['patch_id']}",
                    f"visible_surface_fraction: {_safe_float(row.get('visible_surface_fraction')):.4f}",
                    f"free_space_fraction: {_safe_float(row.get('free_space_fraction')):.4f}",
                    f"unknown_fraction: {_safe_float(row.get('unknown_fraction')):.4f}",
                ],
                output_path=patch_panels_dir / f"{gallery_name}__{row['patch_id']}__visibility.png",
            )
            qualitative_paths.append(visibility_path)

    selected_sequences = _select_sequence_ids_for_visualization(per_sequence_rows)
    _print_progress(f"[eval] rendering sequence maps count={len(selected_sequences)}")
    for map_label, sequence_id in tqdm(selected_sequences, desc="render sequence maps", dynamic_ncols=True):
        bank = sequence_map_bank[sequence_id]
        seq_map_path = render_sequence_visibility_map(
            patch_centers_world=np.asarray(bank["patch_centers"], dtype=np.float32),
            visible_surface_fraction=np.asarray(bank["visible_surface_fraction"], dtype=np.float32),
            free_space_fraction=np.asarray(bank["free_space_fraction"], dtype=np.float32),
            intrinsic_targets=np.asarray(bank["intrinsic_targets"], dtype=np.float32),
            actual_gains=np.asarray(bank["actual_gains"], dtype=np.float32),
            sequence_id=sequence_id,
            map_title=map_label.replace("_", " ").title(),
            output_path=sequence_maps_dir / f"{map_label}__{sequence_id}.png",
        )
        qualitative_paths.append(seq_map_path)

    if prototype_examples_by_code:
        _print_progress(f"[eval] rendering prototype gallery count={min(len(prototype_examples_by_code), 8)}")
        prototype_gallery_path = render_prototype_usage_gallery(
            prototype_examples=[
                prototype_examples_by_code[code_index]
                for code_index in sorted(prototype_examples_by_code.keys())[:8]
            ],
            output_path=prototype_dir / "prototype_gallery.png",
        )
        qualitative_paths.append(prototype_gallery_path)

    report_path = write_report_md(
        output_dir / "report.md",
        checkpoint_path=str(checkpoint_path),
        split_config_path=args.split_config,
        summary_metrics=summary,
        protocol_audit=protocol_audit,
        per_town_rows=per_town_rows,
        per_sequence_rows=per_sequence_rows,
        qualitative_paths=qualitative_paths,
    )

    if wandb_module is not None:
        wandb_run.log({f"eval/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
        for path in qualitative_paths[:10]:
            wandb_run.log({f"eval_images/{path.stem}": wandb_module.Image(str(path))})
        wandb_run.save(str(summary_path))
        wandb_run.save(str(report_path))
        wandb_run.finish()

    _print_progress("[eval] completed")
    print(f"metrics_summary_json: {summary_path}")
    print(f"report_md: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
