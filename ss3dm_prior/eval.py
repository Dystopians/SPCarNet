"""Standardized evaluation entrypoint for SS3DM prior checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.metrics import (
    point_defect_mae,
    recon_chamfer_l1,
    recon_normal_cosine,
    retrieval_top1,
    retrieval_top5,
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
from ss3dm_prior.utils.io import load_yaml
from ss3dm_prior.viz.render_patch_panels import render_patch_triptych, render_retrieval_gallery
from ss3dm_prior.viz.render_sequence_maps import render_sequence_improvement_map


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
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() / args.eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_panels_dir = output_dir / "patch_panels"
    sequence_maps_dir = output_dir / "sequence_maps"
    retrieval_dir = output_dir / "retrieval_gallery"
    for directory in (patch_panels_dir, sequence_maps_dir, retrieval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run_config = _load_model_from_checkpoint(checkpoint_path, device)
    records = _records_for_test_split(patch_cache_dir / "patch_index.jsonl", args.split_config)
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
    wandb_module, wandb_run = _init_wandb(args.eval_name, output_dir, args.wandb_project, args.wandb_mode)

    patch_rows: list[dict[str, Any]] = []
    qualitative_paths: list[Path] = []
    sequence_map_bank: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    clean_latents: list[torch.Tensor] = []
    corrupted_latents: list[torch.Tensor] = []
    clean_bank_points: list[np.ndarray] = []
    clean_bank_patch_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                corrupted_points=moved["corrupted_points"].float(),
                corrupted_normals=moved["corrupted_normals"].float(),
                observed_points=moved["observed_points"].float(),
                clean_points=moved["clean_points"].float(),
                clean_normals=moved["clean_normals"].float(),
            )
            point_defect_pred_raw = torch.expm1(torch.clamp(outputs["point_defect_pred"], min=0.0))
            patch_score_pred_raw = torch.expm1(torch.clamp(outputs["patch_score_pred"], min=0.0))
            corrupted_latents.append(outputs["retrieval_embedding"].detach().cpu())
            if outputs["clean_retrieval_embedding"] is not None:
                clean_latents.append(outputs["clean_retrieval_embedding"].detach().cpu())

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
                }
                patch_rows.append(row)
                sequence_map_bank[sequence_id]["patch_centers"].append(patch_center_world)
                sequence_map_bank[sequence_id]["pred_scores"].append(pred_score)
                sequence_map_bank[sequence_id]["actual_gains"].append(denoise_gain)
                clean_bank_points.append(clean_points[0].detach().cpu().numpy())
                clean_bank_patch_ids.append(patch_id)

    if clean_latents and corrupted_latents:
        all_clean = torch.cat(clean_latents, dim=0)
        all_corrupted = torch.cat(corrupted_latents, dim=0)
        retrieval_metrics = {
            "retrieval_top1": retrieval_top1(all_corrupted, all_clean),
            "retrieval_top5": retrieval_top5(all_corrupted, all_clean),
        }
    else:
        warnings.warn("Retrieval embeddings unavailable during eval.", stacklevel=2)
        retrieval_metrics = {"retrieval_top1": float("nan"), "retrieval_top5": float("nan")}

    if len(patch_rows) >= 2:
        score_spearman_value = score_spearman(
            torch.tensor([row["patch_score_pred"] for row in patch_rows], dtype=torch.float32),
            torch.tensor([row["corruption_score_target"] for row in patch_rows], dtype=torch.float32),
        )
    else:
        warnings.warn("score_spearman unavailable during eval due to too few samples.", stacklevel=2)
        score_spearman_value = float("nan")
    retrieval_metrics["score_spearman"] = score_spearman_value

    summary = aggregate_global_metrics(
        patch_rows,
        {
            "retrieval_top1": retrieval_metrics["retrieval_top1"],
            "retrieval_top5": retrieval_metrics["retrieval_top5"],
            "score_spearman": retrieval_metrics["score_spearman"],
        },
    )
    summary_path = write_summary_json(output_dir / "metrics_summary.json", summary)
    per_town_rows = aggregate_per_group(patch_rows, group_key="town_id")
    per_sequence_rows = aggregate_per_group(patch_rows, group_key="sequence_id")
    write_csv(
        output_dir / "metrics_per_town.csv",
        per_town_rows,
        ["town_id", "patch_count", "sequence_count", "recon_chamfer_l1", "recon_normal_cosine", "denoise_gain_chamfer", "score_mae", "point_defect_mae"],
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
            "score_mae",
            "point_defect_mae",
        ],
    )
    write_csv(
        output_dir / "patch_predictions.csv",
        patch_rows,
        list(patch_rows[0].keys()) if patch_rows else ["patch_id"],
    )

    by_best_gain = sorted(patch_rows, key=lambda row: row["denoise_gain_chamfer"], reverse=True)
    by_worst_gain = sorted(patch_rows, key=lambda row: row["denoise_gain_chamfer"])
    by_score_error = sorted(patch_rows, key=lambda row: row["score_abs_error"], reverse=True)
    patch_dataset = {record["patch_id"]: dataset[idx] for idx, record in enumerate(dataset.records)}
    gallery_specs = [
        ("best_gain", by_best_gain[0] if by_best_gain else None),
        ("worst_gain", by_worst_gain[0] if by_worst_gain else None),
        ("largest_score_error", by_score_error[0] if by_score_error else None),
    ]
    for gallery_name, row in gallery_specs:
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
            )
        triptych_path = render_patch_triptych(
            corrupted_points=sample["corrupted_points"].numpy(),
            recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
            clean_points=sample["clean_points"].numpy(),
            info_lines=[
                f"gallery: {gallery_name}",
                f"town: {row['town_id']}",
                f"sequence: {row['sequence_id']}",
                f"patch: {row['patch_id']}",
                f"score_target: {row['corruption_score_target']:.4f}",
                f"score_pred: {row['patch_score_pred']:.4f}",
                f"gain: {row['denoise_gain_chamfer']:.4f}",
                f"score_abs_err: {row['score_abs_error']:.4f}",
            ],
            output_path=patch_panels_dir / f"{gallery_name}__{row['patch_id']}.png",
        )
        qualitative_paths.append(triptych_path)

    town_to_sequences: dict[str, list[str]] = defaultdict(list)
    for row in patch_rows:
        if row["sequence_id"] not in town_to_sequences[row["town_id"]]:
            town_to_sequences[row["town_id"]].append(row["sequence_id"])
    for town_id, sequence_ids in sorted(town_to_sequences.items()):
        best_sequence = max(
            sequence_ids,
            key=lambda seq_id: sum(1 for row in patch_rows if row["sequence_id"] == seq_id),
        )
        bank = sequence_map_bank[best_sequence]
        seq_map_path = render_sequence_improvement_map(
            patch_centers_world=np.asarray(bank["patch_centers"], dtype=np.float32),
            predicted_scores=np.asarray(bank["pred_scores"], dtype=np.float32),
            actual_gains=np.asarray(bank["actual_gains"], dtype=np.float32),
            sequence_id=best_sequence,
            output_path=sequence_maps_dir / f"{best_sequence}.png",
        )
        qualitative_paths.append(seq_map_path)

    if clean_latents and corrupted_latents:
        all_clean = torch.cat(clean_latents, dim=0)
        all_corrupted = torch.cat(corrupted_latents, dim=0)
        similarity = torch.matmul(
            torch.nn.functional.normalize(all_corrupted, dim=-1),
            torch.nn.functional.normalize(all_clean, dim=-1).transpose(0, 1),
        )
        query_idx = int(similarity.shape[0] // 2)
        nearest_idx = int(similarity[query_idx].argmax().item())
        query_row = patch_rows[query_idx]
        query_sample = patch_dataset[query_row["patch_id"]]
        nearest_clean = clean_bank_points[nearest_idx]
        retrieval_path = render_retrieval_gallery(
            query_corrupted_points=query_sample["corrupted_points"].numpy(),
            target_clean_points=query_sample["clean_points"].numpy(),
            nearest_clean_points=nearest_clean,
            info_lines=[
                f"query_patch: {query_row['patch_id']}",
                f"nearest_patch: {clean_bank_patch_ids[nearest_idx]}",
                f"pred_score: {query_row['patch_score_pred']:.4f}",
                f"actual_gain: {query_row['denoise_gain_chamfer']:.4f}",
            ],
            output_path=retrieval_dir / f"{query_row['patch_id']}_retrieval.png",
        )
        qualitative_paths.append(retrieval_path)

    report_path = write_report_md(
        output_dir / "report.md",
        checkpoint_path=str(checkpoint_path),
        split_config_path=args.split_config,
        summary_metrics=summary,
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

    print(f"metrics_summary_json: {summary_path}")
    print(f"report_md: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
