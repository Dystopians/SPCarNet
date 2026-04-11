"""Training loop for SS3DM prior local patch learning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.engine.checkpoint import load_checkpoint, save_checkpoint
from ss3dm_prior.losses import compute_patch_losses
from ss3dm_prior.metrics import (
    denoise_gain_chamfer,
    point_defect_mae,
    recon_chamfer_l1,
    recon_normal_cosine,
    retrieval_top1,
    retrieval_top5,
    score_mae,
    score_spearman,
)
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser
from ss3dm_prior.utils.io import load_yaml
from ss3dm_prior.viz.render_patch_panels import (
    render_patch_denoise_panel,
    render_patch_triptych,
    render_retrieval_gallery,
)
from ss3dm_prior.viz.render_sequence_maps import render_sequence_improvement_map


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_grad_norm(parameters) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total += grad_norm**2
        found = True
    return float(total**0.5) if found else 0.0


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
        if key in tensor_keys:
            collated[key] = default_collate(values)
        else:
            collated[key] = values
    return collated


class _NullRun:
    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


def _init_wandb(train_config: dict[str, Any], output_dir: Path, run_name: str):
    if not bool(train_config.get("wandb_enable", True)):
        return None, _NullRun()
    try:
        import wandb  # type: ignore
    except Exception as exc:
        warnings.warn(f"wandb unavailable, falling back to disabled logging: {exc}", stacklevel=2)
        return None, _NullRun()

    mode = str(train_config.get("wandb_mode", "offline"))
    project = str(train_config.get("wandb_project", "ss3dm_prior"))
    run = wandb.init(
        project=project,
        name=run_name,
        mode=mode,
        dir=str(output_dir),
        config=train_config,
        reinit=True,
    )
    return wandb, run


@dataclass
class DatasetBundle:
    train_dataset: TeacherPatchTrainDataset
    val_dataset: TeacherPatchTrainDataset


def _split_records_for_debug(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    if len(indices) <= 1:
        return records, records
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_idx = set(indices[:val_count].tolist())
    train_records = [records[i] for i in range(len(records)) if i not in val_idx]
    val_records = [records[i] for i in range(len(records)) if i in val_idx]
    if not train_records:
        train_records = list(val_records)
    return train_records, val_records


def build_datasets(
    *,
    patch_index_path: str | Path,
    split_config: str | Path | dict[str, Any],
    corruption_config: dict[str, Any],
    train_config: dict[str, Any],
    seed: int,
) -> DatasetBundle:
    patch_index_path = Path(patch_index_path).expanduser().resolve()
    all_records = read_patch_index_jsonl(patch_index_path)
    debug_use_all = bool(train_config.get("debug_use_all_patches_for_train_val", False))
    val_fraction = float(train_config.get("debug_val_fraction", 0.25))
    allow_debug_override = bool(train_config.get("allow_debug_split_override", False))
    if isinstance(split_config, (str, Path)):
        split_data = load_yaml(split_config)
    else:
        split_data = split_config

    if debug_use_all and not allow_debug_override:
        train_towns = set(split_data.get("train_towns", []))
        val_towns = set(split_data.get("val_towns", []))
        test_towns = set(split_data.get("test_towns", []))
        if val_towns or test_towns:
            raise ValueError(
                "debug_use_all_patches_for_train_val=True bypasses town holdout and is only allowed "
                "for explicit debug runs. Set allow_debug_split_override=True to opt in."
            )

    if debug_use_all:
        train_records, val_records = _split_records_for_debug(all_records, seed=seed, val_fraction=val_fraction)
    else:
        train_dataset = TeacherPatchTrainDataset(
            patch_index_path=patch_index_path,
            split_config=split_config,
            subsets=("train",),
            corruption_config=corruption_config,
            seed=seed,
            dynamic_corruption=True,
        )
        val_dataset = TeacherPatchTrainDataset(
            patch_index_path=patch_index_path,
            split_config=split_config,
            subsets=("val",),
            corruption_config=corruption_config,
            seed=seed + 1000,
            dynamic_corruption=False,
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            if bool(train_config.get("allow_split_fallback", False)):
                warnings.warn(
                    "Train/val split produced empty dataset; falling back to debug all-patch split.",
                    stacklevel=2,
                )
                train_records, val_records = _split_records_for_debug(
                    all_records,
                    seed=seed,
                    val_fraction=val_fraction,
                )
            else:
                raise ValueError("Train or val dataset is empty. Provide matching patch cache or enable fallback.")
        else:
            return DatasetBundle(train_dataset=train_dataset, val_dataset=val_dataset)

    train_dataset = TeacherPatchTrainDataset(
        patch_index_path=patch_index_path,
        records=train_records,
        split_config=None,
        corruption_config=corruption_config,
        seed=seed,
        dynamic_corruption=True,
    )
    val_dataset = TeacherPatchTrainDataset(
        patch_index_path=patch_index_path,
        records=val_records,
        split_config=None,
        corruption_config=corruption_config,
        seed=seed + 1000,
        dynamic_corruption=False,
    )
    return DatasetBundle(train_dataset=train_dataset, val_dataset=val_dataset)


class SS3DMPriorTrainer:
    def __init__(
        self,
        *,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        patch_index_path: str | Path,
        split_config: str | Path | dict[str, Any],
        output_dir: str | Path,
        run_name: str,
        run_metadata: dict[str, Any],
        resume_path: str | Path | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.run_metadata = run_metadata
        self.seed = int(train_config.get("seed", 0))
        set_global_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = bool(train_config.get("amp", False)) and self.device.type == "cuda"

        self.datasets = build_datasets(
            patch_index_path=patch_index_path,
            split_config=split_config,
            corruption_config=model_config["corruptions"],
            train_config=train_config,
            seed=self.seed,
        )
        self.train_loader = DataLoader(
            self.datasets.train_dataset,
            batch_size=int(train_config.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(train_config.get("num_workers", 0)),
            collate_fn=_collate_samples,
        )
        self.val_loader = DataLoader(
            self.datasets.val_dataset,
            batch_size=int(train_config.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(train_config.get("num_workers", 0)),
            collate_fn=_collate_samples,
        )

        self.model = LocalPatchDenoiser(**model_config["model"]).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_config.get("lr", 1e-3)),
            weight_decay=float(train_config.get("weight_decay", 1e-4)),
        )
        scheduler_name = str(train_config.get("lr_scheduler", "none")).strip().lower()
        if scheduler_name in {"", "none"}:
            self.scheduler = None
        elif scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(int(train_config.get("epochs", 1)), 1),
                eta_min=float(train_config.get("min_lr", 1e-5)),
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)

        self.best_metrics = {
            "best_recon": float("inf"),
            "best_gain": float("-inf"),
        }
        self.start_epoch = 0
        self.global_step = 0
        if resume_path is not None:
            payload = load_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            self.start_epoch = int(payload.get("epoch", 0)) + 1
            self.global_step = int(payload.get("global_step", 0))
            self.best_metrics.update(payload.get("best_metrics", {}))

        self.wandb_module, self.wandb_run = _init_wandb(train_config, self.output_dir, run_name)
        self.visual_dir = self.output_dir / "visualizations"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs" / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.step_visual_examples = self._build_step_visual_examples()

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _forward_batch(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        outputs = self.model(
            corrupted_points=batch["corrupted_points"].float(),
            corrupted_normals=batch["corrupted_normals"].float(),
            observed_points=batch["observed_points"].float(),
            clean_points=batch["clean_points"].float(),
            clean_normals=batch["clean_normals"].float(),
        )
        losses = compute_patch_losses(outputs, batch, self.model_config.get("loss_weights", {}))
        return outputs, losses

    def _log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=step)

    def _build_step_visual_examples(self) -> list[dict[str, Any]]:
        interval = int(self.train_config.get("step_visualization_interval_steps", 0) or 0)
        if interval <= 0:
            return []
        source_dataset = self.datasets.val_dataset if len(self.datasets.val_dataset) else self.datasets.train_dataset
        desired_ids = set(self.train_config.get("step_visualization_patch_ids", []) or [])
        max_examples = int(self.train_config.get("step_visualization_num_examples", 1))
        examples: list[dict[str, Any]] = []
        for idx in range(len(source_dataset)):
            sample = source_dataset[idx]
            if desired_ids and sample["patch_id"] not in desired_ids:
                continue
            examples.append(sample)
            if len(examples) >= max_examples:
                break
        return examples

    def _maybe_log_step_visualization(self) -> None:
        interval = int(self.train_config.get("step_visualization_interval_steps", 0) or 0)
        if interval <= 0 or not self.step_visual_examples or self.global_step % interval != 0:
            return

        step_dir = self.visual_dir / f"step_{self.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        logged_images = {}
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for sample in self.step_visual_examples:
                batch = _collate_samples([sample])
                batch = self._move_batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs, _ = self._forward_batch(batch)
                point_defect_pred_raw = torch.expm1(torch.clamp(outputs["point_defect_pred"], min=0.0))
                patch_score_pred_raw = torch.expm1(torch.clamp(outputs["patch_score_pred"], min=0.0))
                chamfer_before = float(
                    recon_chamfer_l1(batch["corrupted_points"], batch["clean_points"]).detach().cpu()
                )
                chamfer_after = float(
                    recon_chamfer_l1(outputs["recon_points"], batch["clean_points"]).detach().cpu()
                )
                info_lines = [
                    f"step: {self.global_step}",
                    f"town: {batch['town_id'][0]}",
                    f"sequence: {batch['sequence_id'][0]}",
                    f"patch: {batch['patch_id'][0]}",
                    f"score_target: {float(batch['corruption_score_target'][0].detach().cpu()):.4f}",
                    f"score_pred: {float(patch_score_pred_raw[0].detach().cpu()):.4f}",
                    f"chamfer_before: {chamfer_before:.4f}",
                    f"chamfer_after: {chamfer_after:.4f}",
                    f"denoise_gain: {chamfer_before - chamfer_after:.4f}",
                ]
                triptych_path = render_patch_triptych(
                    corrupted_points=batch["corrupted_points"][0].detach().cpu().numpy(),
                    recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
                    clean_points=batch["clean_points"][0].detach().cpu().numpy(),
                    info_lines=info_lines,
                    output_path=step_dir / f"{batch['patch_id'][0]}_triptych.png",
                )
                panel_path = render_patch_denoise_panel(
                    observed_points=batch["observed_points"][0].detach().cpu().numpy(),
                    corrupted_points=batch["corrupted_points"][0].detach().cpu().numpy(),
                    recon_points=outputs["recon_points"][0].detach().cpu().numpy(),
                    clean_points=batch["clean_points"][0].detach().cpu().numpy(),
                    defect_scores=point_defect_pred_raw[0].detach().cpu().numpy(),
                    info_lines=info_lines,
                    output_path=step_dir / f"{batch['patch_id'][0]}_panel.png",
                )
                if self.wandb_module is not None:
                    patch_id = batch["patch_id"][0]
                    logged_images[f"viz_step/comparison/{patch_id}"] = self.wandb_module.Image(str(triptych_path))
                    logged_images[f"viz_step/detail/{patch_id}"] = self.wandb_module.Image(str(panel_path))
        if was_training:
            self.model.train()
        if logged_images:
            self._log(logged_images, step=self.global_step)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        interval = int(self.train_config.get("log_interval", 10))
        stats = defaultdict(list)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            iter_start = time.time()
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs, losses = self._forward_batch(batch)
            self.scaler.scale(losses["total_loss"]).backward()
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            grad_clip_norm = float(self.train_config.get("grad_clip_norm", 0.0) or 0.0)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
            grad_norm = _safe_grad_norm(self.model.parameters())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
            self._maybe_log_step_visualization()

            batch_time = time.time() - iter_start
            chamfer_before = float(
                recon_chamfer_l1(
                    batch["corrupted_points"],
                    batch["clean_points"],
                )
                .detach()
                .cpu()
            )
            batch_gain = chamfer_before - float(losses["recon_chamfer_loss"].detach().cpu())
            stats["total_loss"].append(float(losses["total_loss"].detach().cpu()))
            for key, value in losses.items():
                if key == "total_loss":
                    continue
                stats[key].append(float(value.detach().cpu()))
            stats["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
            stats["grad_norm"].append(float(grad_norm))
            stats["batch_time"].append(float(batch_time))
            stats["denoise_gain_chamfer"].append(batch_gain)

            if (batch_idx + 1) % interval == 0:
                self._log(
                    {
                        "train/total_loss": _safe_mean(stats["total_loss"][-interval:]),
                        "train/recon_chamfer_loss": _safe_mean(stats["recon_chamfer_loss"][-interval:]),
                        "train/recon_normal_loss": _safe_mean(stats["recon_normal_loss"][-interval:]),
                        "train/point_defect_loss": _safe_mean(stats["point_defect_loss"][-interval:]),
                        "train/patch_score_loss": _safe_mean(stats["patch_score_loss"][-interval:]),
                        "train/latent_align_loss": _safe_mean(stats["latent_align_loss"][-interval:]),
                        "train/retrieval_align_loss": _safe_mean(stats["retrieval_align_loss"][-interval:]),
                        "train/denoise_gain_chamfer": _safe_mean(stats["denoise_gain_chamfer"][-interval:]),
                        "train/lr": stats["lr"][-1],
                        "train/grad_norm": stats["grad_norm"][-1],
                        "train/batch_time": stats["batch_time"][-1],
                    },
                    step=self.global_step,
                )

        train_metrics = {f"train_{key}": _safe_mean(values) for key, values in stats.items()}
        train_metrics["train_epoch_time"] = float(time.time() - epoch_start)
        return train_metrics

    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        losses_acc = defaultdict(list)
        metrics_acc = defaultdict(list)
        val_start = time.time()
        sequence_map_bank: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        patch_examples: list[dict[str, Any]] = []
        clean_retrieval_embeddings: list[torch.Tensor] = []
        retrieval_embeddings: list[torch.Tensor] = []
        retrieval_bank_points: list[np.ndarray] = []
        retrieval_bank_ids: list[str] = []
        score_preds = []
        score_targets = []
        gain_targets = []

        fixed_patch_ids = set(self.train_config.get("fixed_visualization_patch_ids", []) or [])
        max_examples = int(self.train_config.get("max_visualization_examples", 3))

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs, losses = self._forward_batch(batch)

                for key, value in losses.items():
                    losses_acc[key].append(float(value.detach().cpu()))

                point_defect_pred_raw = torch.expm1(torch.clamp(outputs["point_defect_pred"], min=0.0))
                patch_score_pred_raw = torch.expm1(torch.clamp(outputs["patch_score_pred"], min=0.0))
                metrics_acc["score_mae"].append(
                    float(score_mae(patch_score_pred_raw, batch["corruption_score_target"]).detach().cpu())
                )
                metrics_acc["point_defect_mae"].append(
                    float(point_defect_mae(point_defect_pred_raw, batch["point_defect_target"]).detach().cpu())
                )

                retrieval_embeddings.append(outputs["retrieval_embedding"].detach().cpu())
                if outputs["clean_retrieval_embedding"] is not None:
                    clean_retrieval_embeddings.append(outputs["clean_retrieval_embedding"].detach().cpu())

                batch_size = batch["clean_points"].shape[0]
                for sample_idx in range(batch_size):
                    patch_id = batch["patch_id"][sample_idx]
                    sequence_id = batch["sequence_id"][sample_idx]
                    chamfer_before = float(
                        recon_chamfer_l1(
                            batch["corrupted_points"][sample_idx : sample_idx + 1],
                            batch["clean_points"][sample_idx : sample_idx + 1],
                        )
                        .detach()
                        .cpu()
                    )
                    chamfer_after = float(
                        recon_chamfer_l1(
                            outputs["recon_points"][sample_idx : sample_idx + 1],
                            batch["clean_points"][sample_idx : sample_idx + 1],
                        )
                        .detach()
                        .cpu()
                    )
                    normal_cos = float(
                        recon_normal_cosine(
                            outputs["recon_points"][sample_idx : sample_idx + 1],
                            outputs["recon_normals"][sample_idx : sample_idx + 1],
                            batch["clean_points"][sample_idx : sample_idx + 1],
                            batch["clean_normals"][sample_idx : sample_idx + 1],
                        )
                        .detach()
                        .cpu()
                    )
                    gain = chamfer_before - chamfer_after
                    metrics_acc["recon_chamfer_l1"].append(chamfer_after)
                    metrics_acc["recon_normal_cosine"].append(normal_cos)
                    metrics_acc["denoise_gain_chamfer"].append(gain)
                    patch_center = batch["patch_center_world"][sample_idx].detach().cpu().numpy()
                    sequence_map_bank[sequence_id]["patch_centers"].append(patch_center)
                    sequence_map_bank[sequence_id]["pred_scores"].append(float(patch_score_pred_raw[sample_idx].detach().cpu()))
                    sequence_map_bank[sequence_id]["actual_gains"].append(float(gain))
                    retrieval_bank_points.append(batch["clean_points"][sample_idx].detach().cpu().numpy())
                    retrieval_bank_ids.append(patch_id)
                    score_preds.append(float(patch_score_pred_raw[sample_idx].detach().cpu()))
                    score_targets.append(float(batch["corruption_score_target"][sample_idx].detach().cpu()))
                    gain_targets.append(float(gain))

                    should_capture = len(patch_examples) < max_examples and (
                        not fixed_patch_ids or patch_id in fixed_patch_ids
                    )
                    if should_capture:
                        patch_examples.append(
                            {
                                "town_id": batch["town_id"][sample_idx],
                                "sequence_id": sequence_id,
                                "patch_id": patch_id,
                                "observed_points": batch["observed_points"][sample_idx].detach().cpu().numpy(),
                                "corrupted_points": batch["corrupted_points"][sample_idx].detach().cpu().numpy(),
                                "clean_points": batch["clean_points"][sample_idx].detach().cpu().numpy(),
                                "recon_points": outputs["recon_points"][sample_idx].detach().cpu().numpy(),
                                "defect_scores": point_defect_pred_raw[sample_idx].detach().cpu().numpy(),
                                "corruption_score_target": float(batch["corruption_score_target"][sample_idx].detach().cpu()),
                                "pred_score": float(patch_score_pred_raw[sample_idx].detach().cpu()),
                                "gain": float(gain),
                                "chamfer_before": chamfer_before,
                                "chamfer_after": chamfer_after,
                            }
                        )

        if clean_retrieval_embeddings and retrieval_embeddings:
            all_queries = torch.cat(retrieval_embeddings, dim=0)
            all_targets = torch.cat(clean_retrieval_embeddings, dim=0)
            metrics_acc["retrieval_top1"].append(retrieval_top1(all_queries, all_targets))
            metrics_acc["retrieval_top5"].append(retrieval_top5(all_queries, all_targets))
        else:
            warnings.warn("Retrieval metrics unavailable due to empty embedding bank.", stacklevel=2)
            metrics_acc["retrieval_top1"].append(float("nan"))
            metrics_acc["retrieval_top5"].append(float("nan"))

        if len(score_preds) >= 2:
            metrics_acc["score_spearman"].append(
                float(
                    score_spearman(
                        torch.tensor(score_preds, dtype=torch.float32),
                        torch.tensor(score_targets, dtype=torch.float32),
                    )
                )
            )
            metrics_acc["predicted_gain_spearman"].append(
                float(
                    score_spearman(
                        torch.tensor(score_preds, dtype=torch.float32),
                        torch.tensor(gain_targets, dtype=torch.float32),
                    )
                )
            )
        else:
            warnings.warn("score_spearman unavailable; too few validation samples.", stacklevel=2)
            metrics_acc["score_spearman"].append(float("nan"))
            metrics_acc["predicted_gain_spearman"].append(float("nan"))

        val_metrics = {f"val_{key}": _safe_mean(values) for key, values in losses_acc.items()}
        for key, values in metrics_acc.items():
            val_metrics[f"val_{key}"] = _safe_mean(values)
        val_metrics["val_epoch_time"] = float(time.time() - val_start)

        self._render_validation_artifacts(
            epoch=epoch,
            patch_examples=patch_examples,
            sequence_map_bank=sequence_map_bank,
            clean_latents=torch.cat(clean_retrieval_embeddings, dim=0) if clean_retrieval_embeddings else None,
            corrupted_latents=torch.cat(retrieval_embeddings, dim=0) if retrieval_embeddings else None,
            retrieval_bank_points=retrieval_bank_points,
            retrieval_bank_ids=retrieval_bank_ids,
        )
        return val_metrics

    def _render_validation_artifacts(
        self,
        *,
        epoch: int,
        patch_examples: list[dict[str, Any]],
        sequence_map_bank: dict[str, dict[str, list[Any]]],
        clean_latents: torch.Tensor | None,
        corrupted_latents: torch.Tensor | None,
        retrieval_bank_points: list[np.ndarray],
        retrieval_bank_ids: list[str],
    ) -> None:
        if not patch_examples:
            return
        epoch_dir = self.visual_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        logged_images = {}

        for example in patch_examples:
            info_lines = [
                f"town: {example['town_id']}",
                f"sequence: {example['sequence_id']}",
                f"patch: {example['patch_id']}",
                f"score_target: {example['corruption_score_target']:.4f}",
                f"score_pred: {example['pred_score']:.4f}",
                f"chamfer_before: {example['chamfer_before']:.4f}",
                f"chamfer_after: {example['chamfer_after']:.4f}",
                f"denoise_gain: {example['gain']:.4f}",
            ]
            triptych_path = render_patch_triptych(
                corrupted_points=example["corrupted_points"],
                recon_points=example["recon_points"],
                clean_points=example["clean_points"],
                info_lines=info_lines,
                output_path=epoch_dir / f"{example['patch_id']}_triptych.png",
            )
            panel_path = render_patch_denoise_panel(
                observed_points=example["observed_points"],
                corrupted_points=example["corrupted_points"],
                recon_points=example["recon_points"],
                clean_points=example["clean_points"],
                defect_scores=example["defect_scores"],
                info_lines=info_lines,
                output_path=epoch_dir / f"{example['patch_id']}_panel.png",
            )
            if self.wandb_module is not None:
                logged_images[f"viz/comparison/{example['patch_id']}"] = self.wandb_module.Image(str(triptych_path))
                logged_images[f"viz/patch_denoise_panel/{example['patch_id']}"] = self.wandb_module.Image(str(panel_path))

        first_sequence = next(iter(sequence_map_bank.keys()), None)
        if first_sequence is not None:
            seq_bank = sequence_map_bank[first_sequence]
            seq_map_path = render_sequence_improvement_map(
                patch_centers_world=np.asarray(seq_bank["patch_centers"], dtype=np.float32),
                predicted_scores=np.asarray(seq_bank["pred_scores"], dtype=np.float32),
                actual_gains=np.asarray(seq_bank["actual_gains"], dtype=np.float32),
                sequence_id=first_sequence,
                output_path=epoch_dir / f"{first_sequence}_sequence_map.png",
            )
            if self.wandb_module is not None:
                logged_images["viz/sequence_improvement_map"] = self.wandb_module.Image(str(seq_map_path))

        if clean_latents is not None and corrupted_latents is not None and len(patch_examples) >= 1:
            query_idx = 0
            similarity = torch.matmul(
                torch.nn.functional.normalize(corrupted_latents, dim=-1),
                torch.nn.functional.normalize(clean_latents, dim=-1).transpose(0, 1),
            )
            nearest_idx = int(similarity[query_idx].argmax().item())
            info_lines = [
                f"query_patch: {patch_examples[query_idx]['patch_id']}",
                f"nearest_clean_patch: {retrieval_bank_ids[nearest_idx] if nearest_idx < len(retrieval_bank_ids) else 'n/a'}",
                f"self_match: {retrieval_bank_ids[nearest_idx] == patch_examples[query_idx]['patch_id'] if nearest_idx < len(retrieval_bank_ids) else False}",
            ]
            gallery_path = render_retrieval_gallery(
                query_corrupted_points=patch_examples[query_idx]["corrupted_points"],
                target_clean_points=patch_examples[query_idx]["clean_points"],
                nearest_clean_points=retrieval_bank_points[nearest_idx] if nearest_idx < len(retrieval_bank_points) else patch_examples[query_idx]["clean_points"],
                info_lines=info_lines,
                output_path=epoch_dir / f"{patch_examples[query_idx]['patch_id']}_retrieval.png",
            )
            if self.wandb_module is not None:
                logged_images["viz/retrieval_gallery"] = self.wandb_module.Image(str(gallery_path))

        if logged_images:
            self._log(logged_images, step=self.global_step)

    def maybe_save_checkpoints(self, epoch: int, val_metrics: dict[str, float]) -> None:
        save_checkpoint(
            self.ckpt_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_metrics=self.best_metrics,
            run_config=self.run_metadata,
        )
        save_interval = int(self.train_config.get("save_interval", 1))
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_checkpoint(
                self.ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
            )
        recon_metric = val_metrics.get("val_recon_chamfer_l1", float("inf"))
        gain_metric = val_metrics.get("val_denoise_gain_chamfer", float("-inf"))
        if recon_metric < self.best_metrics["best_recon"]:
            self.best_metrics["best_recon"] = recon_metric
            save_checkpoint(
                self.ckpt_dir / "best_recon.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
            )
        if gain_metric > self.best_metrics["best_gain"]:
            self.best_metrics["best_gain"] = gain_metric
            save_checkpoint(
                self.ckpt_dir / "best_gain.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
            )

    def fit(self) -> dict[str, Any]:
        epochs = int(self.train_config.get("epochs", 1))
        val_interval = int(self.train_config.get("val_interval", 1))
        history = []
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_one_epoch(epoch)
            if val_interval > 0 and ((epoch + 1) % val_interval == 0 or epoch == epochs - 1):
                val_metrics = self.validate(epoch)
                self.maybe_save_checkpoints(epoch, val_metrics)
            else:
                val_metrics = {}
                save_checkpoint(
                    self.ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metrics=self.best_metrics,
                    run_config=self.run_metadata,
                )
            merged = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(merged)
            log_payload = {f"epoch/{key}": value for key, value in merged.items() if key != "epoch"}
            log_payload["epoch/index"] = epoch
            self._log(log_payload, step=self.global_step)
            if self.scheduler is not None:
                self.scheduler.step()

        history_path = self.output_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.wandb_run.finish()
        return {
            "history": history,
            "best_metrics": self.best_metrics,
            "history_path": str(history_path),
        }
