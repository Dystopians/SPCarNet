"""Training loop for SS3DM prior local patch learning."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
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
from torch.utils.data import WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.engine.checkpoint import load_checkpoint, save_checkpoint
from ss3dm_prior.losses import compute_patch_losses
from ss3dm_prior.metrics import (
    denoise_gain_chamfer,
    free_space_violation_rate,
    free_space_fp_rate,
    hidden_completion_gain_or_nan,
    intrinsic_difficulty_mae,
    intrinsic_difficulty_calibration_mae,
    intrinsic_difficulty_spearman,
    occupancy_iou_visible,
    point_defect_mae,
    prototype_usage_entropy,
    recon_chamfer_l1,
    recon_chamfer_l1_or_nan,
    recon_normal_cosine,
    recon_normal_cosine_or_nan,
    retrieval_top1_cross_sequence,
    retrieval_top1_nonself,
    retrieval_top1_self_aligned,
    retrieval_top5_nonself,
    retrieval_top5_self_aligned,
    score_mae,
    score_spearman,
)
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser
from ss3dm_prior.utils.io import load_yaml
from ss3dm_prior.viz.render_patch_panels import (
    render_difficulty_calibration_panel,
    render_free_space_error_panel,
    render_hybrid_reconstruction_panel,
    render_patch_denoise_panel,
    render_patch_triptych,
    render_prototype_usage_gallery,
    render_retrieval_gallery,
    render_visible_vs_hidden_panel,
    render_visibility_panel,
)
from ss3dm_prior.viz.render_sequence_maps import (
    render_sequence_improvement_map,
    render_sequence_visibility_map,
)


def set_global_seed(seed: int, *, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_float(value: float | int | np.floating | np.integer | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = float(value)
    return default if math.isnan(value) or math.isinf(value) else value


def _score_from_weights(metrics: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += float(weight) * _safe_float(metrics.get(key), default=0.0)
    return float(total)


def _effective_loss_weights(weights: dict[str, float]) -> dict[str, float]:
    corruption_score_weight = float(weights.get("corruption_score_loss", weights.get("patch_score_loss", 0.0)))
    return {
        "recon_chamfer_loss": float(weights.get("recon_chamfer_loss", 1.0)),
        "recon_normal_loss": float(weights.get("recon_normal_loss", 0.5)),
        "nearest_neighbor_l1_loss": float(weights.get("nearest_neighbor_l1_loss", 0.0)),
        "reverse_nearest_neighbor_l1_loss": float(weights.get("reverse_nearest_neighbor_l1_loss", 0.0)),
        "point_defect_loss": float(weights.get("point_defect_loss", 1.0)),
        "corruption_score_loss": corruption_score_weight,
        "intrinsic_difficulty_loss": float(weights.get("intrinsic_difficulty_loss", 0.0)),
        "occupancy_bce_loss": float(weights.get("occupancy_bce_loss", 0.0)),
        "free_space_violation_loss": float(weights.get("free_space_violation_loss", 0.0)),
        "hidden_completion_chamfer_loss": float(weights.get("hidden_completion_chamfer_loss", 0.0)),
        "visible_recon_chamfer_loss": float(weights.get("visible_recon_chamfer_loss", 0.0)),
        "vq_commitment_loss": float(weights.get("vq_commitment_loss", 0.0)),
        "prototype_diversity_loss": float(weights.get("prototype_diversity_loss", 0.0)),
        "latent_align_loss": float(weights.get("latent_align_loss", 0.25)),
        "retrieval_align_loss": float(weights.get("retrieval_align_loss", 0.0)),
        "latent_flow_matching_loss": float(weights.get("latent_flow_matching_loss", 0.0)),
        "point_flow_matching_loss": float(weights.get("point_flow_matching_loss", 0.0)),
        "symmetry_consistency_loss": float(weights.get("symmetry_consistency_loss", 0.0)),
    }


def _loss_contribution_means(
    metric_values: dict[str, float],
    effective_weights: dict[str, float],
) -> dict[str, float]:
    contributions: dict[str, float] = {}
    for key, weight in effective_weights.items():
        if key not in metric_values:
            continue
        contributions[key] = float(weight) * _safe_float(metric_values[key], default=0.0)
    return contributions


def _is_effectively_whole_car_run(run_metadata: dict[str, Any]) -> bool:
    dataset_cfg = (run_metadata.get("data_config") or {}).get("dataset", {}) or {}
    dataset_name = str(dataset_cfg.get("name", "")).strip().lower()
    dataset_source = str(dataset_cfg.get("source", "")).strip().lower()
    return dataset_name == "meshfleet_car_whole_mesh" or dataset_source == "meshfleet_trellis"


def _build_hard_example_sampler(
    records: list[dict[str, Any]],
    *,
    enable: bool,
    alpha: float,
    floor: float,
    power: float,
) -> WeightedRandomSampler | None:
    if not enable or not records:
        return None
    sample_weights = []
    for record in records:
        difficulty = _safe_float(record.get("intrinsic_patch_difficulty_target"), default=0.0)
        sample_weight = float(floor) + float(alpha) * float(max(difficulty, 0.0) ** power)
        sample_weights.append(max(sample_weight, 1e-6))
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(records), replacement=True)


def _merge_loss_weights_for_epoch(
    base_weights: dict[str, float],
    train_config: dict[str, Any],
    epoch: int,
) -> tuple[dict[str, float], str]:
    merged = dict(base_weights)
    curriculum = train_config.get("curriculum", {}) or {}
    recon_warmup_epochs = int(curriculum.get("recon_warmup_epochs", curriculum.get("warmup_epochs", 0)) or 0)
    main_start_epoch = int(curriculum.get("main_start_epoch", recon_warmup_epochs) or 0)
    occupancy_start_epoch = int(curriculum.get("occupancy_start_epoch", main_start_epoch) or 0)
    intrinsic_start_epoch = int(curriculum.get("intrinsic_start_epoch", main_start_epoch) or 0)
    vq_start_epoch = int(curriculum.get("vq_start_epoch", main_start_epoch) or 0)
    prototype_start_epoch = int(curriculum.get("prototype_start_epoch", main_start_epoch) or 0)
    flow_matching_start_epoch = int(curriculum.get("flow_matching_start_epoch", main_start_epoch) or 0)
    symmetry_start_epoch = int(curriculum.get("symmetry_start_epoch", main_start_epoch) or 0)

    if epoch < recon_warmup_epochs:
        stage_name = "recon_warmup"
    elif epoch < max(
        occupancy_start_epoch,
        intrinsic_start_epoch,
        vq_start_epoch,
        prototype_start_epoch,
        flow_matching_start_epoch,
        symmetry_start_epoch,
    ):
        stage_name = "transition"
    else:
        stage_name = "main"

    if epoch < occupancy_start_epoch:
        merged["occupancy_bce_loss"] = 0.0
        merged["free_space_violation_loss"] = 0.0
    if epoch < intrinsic_start_epoch:
        merged["intrinsic_difficulty_loss"] = 0.0
    if epoch < vq_start_epoch:
        merged["vq_commitment_loss"] = 0.0
    if epoch < prototype_start_epoch:
        merged["prototype_diversity_loss"] = 0.0
    if epoch < flow_matching_start_epoch:
        merged["latent_flow_matching_loss"] = 0.0
    if epoch < symmetry_start_epoch:
        merged["symmetry_consistency_loss"] = 0.0
    if epoch < recon_warmup_epochs:
        merged["retrieval_align_loss"] = 0.0
        merged["latent_align_loss"] = 0.0
        # During warmup the main recon losses are boosted so the model is
        # pushed hard toward an honest reconstruction before auxiliary heads
        # can distract. Multipliers default to 1.0 (no-op) and are read from
        # the curriculum block so individual configs can tune them.
        recon_boost = float(curriculum.get("recon_chamfer_warmup_scale", 1.0))
        normal_boost = float(curriculum.get("recon_normal_warmup_scale", 1.0))
        nn_boost = float(curriculum.get("nearest_neighbor_warmup_scale", 1.0))
        merged["recon_chamfer_loss"] = float(merged.get("recon_chamfer_loss", 0.0)) * recon_boost
        merged["recon_normal_loss"] = float(merged.get("recon_normal_loss", 0.0)) * normal_boost
        merged["nearest_neighbor_l1_loss"] = float(merged.get("nearest_neighbor_l1_loss", 0.0)) * nn_boost
        merged["reverse_nearest_neighbor_l1_loss"] = float(merged.get("reverse_nearest_neighbor_l1_loss", 0.0)) * nn_boost
    return merged, stage_name


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


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.backup:
                continue
            parameter.data.copy_(self.backup[name])
        self.backup = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        shadow = state_dict.get("shadow", {}) or {}
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in shadow.items()
            if isinstance(tensor, torch.Tensor)
        }


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
        "surface_support_mask",
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
        "visible_support_fraction",
        "hidden_surface_fraction",
        "free_space_fraction",
        "unknown_fraction",
        "free_space_hard_negative_count",
        "intrinsic_patch_difficulty_target",
        "symmetry_plane_normal",
        "symmetry_plane_offset",
        "symmetry_target_confidence",
        "symmetry_chamfer_residual",
        "patch_center_world",
        "patch_radius_m",
        "scale_id",
    }
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in tensor_keys:
            collated[key] = default_collate(values)
        else:
            collated[key] = values
    return collated


def _sample_optional_tensor(
    value: torch.Tensor | list[torch.Tensor] | None,
    index: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, list):
        if index >= len(value) or not isinstance(value[index], torch.Tensor):
            return None
        tensor = value[index].to(device)
        return tensor.unsqueeze(0)
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return value.unsqueeze(0).to(device)
        return value[index : index + 1].to(device)
    return None


def _safe_tensor_float(value: torch.Tensor | None) -> float:
    if value is None:
        return float("nan")
    return _safe_float(value.detach().cpu().item(), default=float("nan"))


def _curriculum_stage_index(stage_name: str) -> float:
    return float({"recon_warmup": 0, "transition": 1, "main": 2}.get(stage_name, -1))


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
        device: torch.device | None = None,
        resume_path: str | Path | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.run_metadata = run_metadata
        self.seed = int(train_config.get("seed", 0))
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(self.seed, device=self.device)
        self.amp_enabled = bool(train_config.get("amp", False)) and self.device.type == "cuda"
        self.grad_accum_steps = max(int(train_config.get("grad_accum_steps", 1) or 1), 1)
        ema_cfg = train_config.get("ema", {}) or {}
        self.ema_enabled = bool(ema_cfg.get("enable", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.999))
        self.ema_use_for_eval = bool(ema_cfg.get("use_for_eval", self.ema_enabled))

        self.datasets = build_datasets(
            patch_index_path=patch_index_path,
            split_config=split_config,
            corruption_config=model_config["corruptions"],
            train_config=train_config,
            seed=self.seed,
        )
        self.model_type = str(model_config.get("model", {}).get("model_type", "legacy_v1")).strip().lower()
        self.is_hybrid_v2_family = self.model_type in {
            "hybrid_v2",
            "hybrid_v2_wide",
            "cross_attention_hybrid_v10",
            "v10_cross_attention_hybrid",
            "latent_flow_hybrid_v11",
            "v11_latent_flow_hybrid",
        }
        hard_sampling_cfg = train_config.get("hard_example_sampling", {}) or {}
        train_sampler = _build_hard_example_sampler(
            self.datasets.train_dataset.records,
            enable=bool(hard_sampling_cfg.get("enable", False)),
            alpha=float(hard_sampling_cfg.get("alpha", 1.0)),
            floor=float(hard_sampling_cfg.get("floor", 1.0)),
            power=float(hard_sampling_cfg.get("power", 1.0)),
        )
        self.train_loader = DataLoader(
            self.datasets.train_dataset,
            batch_size=int(train_config.get("batch_size", 4)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
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
        self.model_total_params = int(sum(parameter.numel() for parameter in self.model.parameters()))
        self.model_trainable_params = int(
            sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        )
        print(
            f"[trainer] model_type={self.model_type} "
            f"model_total_params={self.model_total_params} "
            f"model_trainable_params={self.model_trainable_params}",
            flush=True,
        )
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
        self.ema = ExponentialMovingAverage(self.model, decay=self.ema_decay) if self.ema_enabled else None

        self.best_metrics = {
            "best_recon": float("inf"),
            "best_gain": float("-inf"),
            "best_composite": float("-inf"),
            "best_visibility": float("-inf"),
            "best_paper": float("-inf"),
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
            if self.ema is not None and isinstance(payload.get("ema_state_dict"), dict):
                self.ema.load_state_dict(payload["ema_state_dict"])
            # optimizer.load_state_dict overwrites param_groups[*]['lr'] with
            # the saved (usually cosine-decayed) value, and CosineAnnealingLR's
            # first step() snapshots the current lr rather than base_lrs when
            # last_epoch hits 0 — the combined effect pins lr at the saved
            # floor forever. Reset lr to the new train_config value so the
            # fresh scheduler actually controls the resumed schedule.
            configured_lr = float(train_config.get("lr", 1e-3))
            for group in self.optimizer.param_groups:
                group["lr"] = configured_lr
                group["initial_lr"] = configured_lr

        self.wandb_module, self.wandb_run = _init_wandb(train_config, self.output_dir, run_name)
        self.visual_dir = self.output_dir / "visualizations"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs" / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.is_whole_car_run = _is_effectively_whole_car_run(run_metadata)
        self.total_epochs = int(self.train_config.get("epochs", 1))
        self.epoch_visualization_interval = max(
            int(self.train_config.get("epoch_visualization_interval_epochs", 5)),
            1,
        )
        self.step_visual_examples = self._build_step_visual_examples()

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ema_enabled": self.ema_enabled,
            "ema_use_for_eval": self.ema_use_for_eval,
        }
        if self.ema is not None:
            payload["ema_state_dict"] = self.ema.state_dict()
        return payload

    @contextmanager
    def _evaluation_model_scope(self):
        if self.ema is not None and self.ema_use_for_eval:
            self.ema.apply_to(self.model)
            try:
                yield
            finally:
                self.ema.restore(self.model)
        else:
            yield

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
                moved[key] = [item.to(self.device) for item in value]
            else:
                moved[key] = value
        return moved

    def _forward_batch(
        self,
        batch: dict[str, Any],
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        outputs = self.model(
            corrupted_points=batch["corrupted_points"].float(),
            corrupted_normals=batch["corrupted_normals"].float(),
            observed_points=batch["observed_points"].float(),
            clean_points=batch["clean_points"].float(),
            clean_normals=batch["clean_normals"].float(),
            query_points_all=batch.get("query_points_all").float() if batch.get("query_points_all") is not None else None,
            visible_clean_points=batch.get("visible_clean_points"),
            visible_clean_normals=batch.get("visible_clean_normals"),
            hidden_clean_points=batch.get("hidden_clean_points"),
            hidden_clean_normals=batch.get("hidden_clean_normals"),
        )
        losses = compute_patch_losses(outputs, batch, loss_weights or self.model_config.get("loss_weights", {}))
        return outputs, losses

    def _log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=step)

    def _metric_should_be_logged(
        self,
        metric_name: str,
        value: float,
        *,
        effective_weights: dict[str, float],
    ) -> bool:
        if not np.isfinite(value):
            return False
        if metric_name in {"patch_score_loss", "val_patch_score_loss", "train_patch_score_loss"}:
            return False
        # The chamfer "loss" is exactly the chamfer L1 distance (weight=1 in
        # compute_patch_losses). Emitting both produces twin panels with
        # identical traces — keep only the *_chamfer_l1 family.
        if metric_name in {
            "recon_chamfer_loss",
            "visible_recon_chamfer_loss",
            "hidden_completion_chamfer_loss",
        }:
            return False
        # Constants that produce flat panels with no signal.
        if metric_name in {"corruption_severity_scale", "ema_decay", "grad_accum_steps"}:
            return False
        if metric_name.endswith("prototype_usage_entropy") and effective_weights.get("vq_commitment_loss", 0.0) <= 0.0:
            return False
        if "intrinsic_difficulty" in metric_name and effective_weights.get("intrinsic_difficulty_loss", 0.0) <= 0.0:
            return False
        if ("occupancy" in metric_name or "free_space" in metric_name) and max(
            effective_weights.get("occupancy_bce_loss", 0.0),
            effective_weights.get("free_space_violation_loss", 0.0),
        ) <= 0.0:
            return False
        if ("vq_commitment" in metric_name or "prototype_diversity" in metric_name) and max(
            effective_weights.get("vq_commitment_loss", 0.0),
            effective_weights.get("prototype_diversity_loss", 0.0),
        ) <= 0.0:
            return False
        if "latent_flow_matching" in metric_name and effective_weights.get("latent_flow_matching_loss", 0.0) <= 0.0:
            return False
        if metric_name.endswith("retrieval_top1_cross_sequence") and self.is_whole_car_run:
            return False
        # In whole-car runs each sample is its own sequence, so top-5 retrieval
        # saturates to 1.0 within a handful of epochs and stops carrying signal.
        # The self-aligned variants also plateau almost immediately.
        if self.is_whole_car_run and (
            metric_name.endswith("retrieval_top5_nonself")
            or metric_name.endswith("retrieval_top5_self_aligned")
            or metric_name.endswith("retrieval_top1_self_aligned")
        ):
            return False
        return True

    def _active_raw_loss_keys(self, effective_weights: dict[str, float]) -> list[str]:
        ordered = [
            "recon_chamfer_loss",
            "recon_normal_loss",
            "nearest_neighbor_l1_loss",
            "reverse_nearest_neighbor_l1_loss",
            "point_defect_loss",
            "corruption_score_loss",
            "intrinsic_difficulty_loss",
            "occupancy_bce_loss",
            "free_space_violation_loss",
            "hidden_completion_chamfer_loss",
            "visible_recon_chamfer_loss",
            "vq_commitment_loss",
            "prototype_diversity_loss",
            "latent_align_loss",
            "retrieval_align_loss",
            "latent_flow_matching_loss",
            "point_flow_matching_loss",
            "symmetry_consistency_loss",
        ]
        return [key for key in ordered if effective_weights.get(key, 0.0) > 0.0]

    def _compact_history_metrics(
        self,
        metrics: dict[str, float],
        *,
        prefix: str,
        effective_weights: dict[str, float],
    ) -> dict[str, float]:
        compact: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
            metric_name = key[len(prefix):] if key.startswith(prefix) else key
            if metric_name == "patch_score_loss":
                continue
            if self._metric_should_be_logged(metric_name, value, effective_weights=effective_weights):
                compact[key] = value
        # Note: weighted_X (= weight * raw) is emitted to the wandb dashboard
        # under epoch/{train,val}_loss_weighted/X for at-a-glance contribution
        # comparison. We do NOT replicate it in history.json — that file is
        # the canonical record and weight × raw is trivially recoverable from
        # the run config, so duplicating it here just doubles the column
        # count.
        return compact

    def _format_interval_train_payload(
        self,
        *,
        stats: dict[str, list[float]],
        interval: int,
        curriculum_stage: str,
        effective_weights: dict[str, float],
    ) -> dict[str, float]:
        payload: dict[str, float] = {}
        payload["train_step/total_loss"] = _safe_mean(stats["total_loss"][-interval:])
        payload["train_step/denoise_gain_chamfer"] = _safe_mean(stats["denoise_gain_chamfer"][-interval:])
        payload["train_step/curriculum_stage_index"] = _curriculum_stage_index(curriculum_stage)
        payload["train_step/lr"] = stats["lr"][-1]
        payload["train_step/grad_norm"] = _safe_mean(stats["grad_norm"][-interval:])
        payload["train_step/batch_time"] = stats["batch_time"][-1]
        payload["train_step/grad_accum_steps"] = float(self.grad_accum_steps)
        if stats.get("ema_decay"):
            payload["train_step/ema_decay"] = stats["ema_decay"][-1]
        for metric_key in [
            "visible_recon_chamfer_l1",
            "hidden_completion_chamfer_l1",
            "visible_recon_normal_cosine",
            "hidden_completion_gain",
            "intrinsic_difficulty_calibration_mae",
            "free_space_fp_rate",
        ]:
            if stats.get(metric_key):
                value = _safe_mean(stats[metric_key][-interval:])
                if self._metric_should_be_logged(metric_key, value, effective_weights=effective_weights):
                    payload[f"train_step/{metric_key}"] = value

        raw_means = {
            key: _safe_mean(stats[key][-interval:])
            for key in self._active_raw_loss_keys(effective_weights)
            if stats.get(key)
        }
        # Per-step we only emit the weighted (contribution to total_loss) view.
        # Raw per-loss means are available at epoch level; carrying both at step
        # cadence produces redundant panels with identical trajectories scaled
        # by a constant factor.
        for key, value in _loss_contribution_means(raw_means, effective_weights).items():
            short_key = key.removesuffix("_loss")
            payload[f"train_step/loss_weighted/{short_key}"] = value
        return payload

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

    def _free_space_violation_scores(
        self,
        query_occupancy_logits: np.ndarray | None,
        query_labels_all: np.ndarray | None,
        query_ignore_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_occupancy_logits is None or query_labels_all is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        labels = np.asarray(query_labels_all, dtype=np.float32)
        logits = np.asarray(query_occupancy_logits, dtype=np.float32)
        ignore_mask = (
            np.asarray(query_ignore_mask, dtype=bool)
            if query_ignore_mask is not None
            else np.zeros(labels.shape, dtype=bool)
        )
        free_mask = (labels <= 0.5) & (~ignore_mask)
        if not np.any(free_mask):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return free_mask, 1.0 / (1.0 + np.exp(-logits[free_mask]))

    def _sample_visibility_metrics(
        self,
        *,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor | None],
        sample_idx: int,
    ) -> dict[str, float]:
        clean_points = batch["clean_points"][sample_idx : sample_idx + 1]
        clean_normals = batch["clean_normals"][sample_idx : sample_idx + 1]
        corrupted_points = batch["corrupted_points"][sample_idx : sample_idx + 1]
        recon_points = outputs["recon_points"][sample_idx : sample_idx + 1]
        recon_normals = outputs["recon_normals"][sample_idx : sample_idx + 1]
        visible_clean_points = _sample_optional_tensor(
            batch.get("visible_clean_points"),
            sample_idx,
            device=self.device,
        )
        visible_clean_normals = _sample_optional_tensor(
            batch.get("visible_clean_normals"),
            sample_idx,
            device=self.device,
        )
        hidden_clean_points = _sample_optional_tensor(
            batch.get("hidden_clean_points"),
            sample_idx,
            device=self.device,
        )
        query_logits = (
            outputs["query_occupancy_logits"][sample_idx : sample_idx + 1]
            if outputs.get("query_occupancy_logits") is not None
            else None
        )
        query_labels_all = batch["query_labels_all"][sample_idx : sample_idx + 1]
        query_ignore_mask = batch["query_ignore_mask"][sample_idx : sample_idx + 1]
        intrinsic_pred = (
            outputs["intrinsic_difficulty_pred"][sample_idx : sample_idx + 1]
            if outputs.get("intrinsic_difficulty_pred") is not None
            else None
        )
        intrinsic_target = batch["intrinsic_patch_difficulty_target"][sample_idx : sample_idx + 1]
        visible_recon_chamfer = (
            recon_chamfer_l1_or_nan(recon_points, visible_clean_points)
            if visible_clean_points is not None
            else float("nan")
        )
        visible_recon_normal = (
            recon_normal_cosine_or_nan(recon_points, recon_normals, visible_clean_points, visible_clean_normals)
            if visible_clean_points is not None and visible_clean_normals is not None
            else float("nan")
        )
        hidden_completion_chamfer = (
            recon_chamfer_l1_or_nan(recon_points, hidden_clean_points)
            if hidden_clean_points is not None
            else float("nan")
        )
        hidden_completion_gain = (
            hidden_completion_gain_or_nan(corrupted_points, recon_points, hidden_clean_points)
            if hidden_clean_points is not None
            else float("nan")
        )
        occupancy_value = (
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
        free_space_fp_value = (
            free_space_fp_rate(
                query_logits.detach().cpu(),
                query_labels_all.detach().cpu(),
                query_ignore_mask.detach().cpu(),
            )
            if query_logits is not None and int(query_labels_all.numel()) > 0
            else float("nan")
        )
        intrinsic_calibration = (
            intrinsic_difficulty_calibration_mae(
                intrinsic_pred.detach().cpu(),
                intrinsic_target.detach().cpu(),
            )
            if intrinsic_pred is not None
            else float("nan")
        )
        return {
            "visible_recon_chamfer_l1": visible_recon_chamfer,
            "hidden_completion_chamfer_l1": hidden_completion_chamfer,
            "visible_recon_normal_cosine": visible_recon_normal,
            "hidden_completion_gain": hidden_completion_gain,
            "intrinsic_difficulty_calibration_mae": intrinsic_calibration,
            "free_space_fp_rate": free_space_fp_value,
            "occupancy_iou_visible": occupancy_value,
            "free_space_violation_rate": free_space_violation_value,
        }

    def _render_v2_patch_artifacts(
        self,
        *,
        example: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Path]:
        if example.get("query_points_all") is None:
            return {}
        query_points_all = np.asarray(example.get("query_points_all"), dtype=np.float32)
        visible_clean_points = np.asarray(
            example.get("visible_clean_points", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        hidden_clean_points = np.asarray(
            example.get("hidden_clean_points", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        free_query_points = np.asarray(
            example.get("free_query_points", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        unknown_query_points = np.asarray(
            example.get("unknown_query_points", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        surface_query_points = np.asarray(
            example.get("surface_query_points", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        has_semantic_queries = bool(
            len(surface_query_points) > 0 or len(free_query_points) > 0 or len(unknown_query_points) > 0
        )
        has_visibility_semantics = bool(len(visible_clean_points) > 0 or len(hidden_clean_points) > 0)
        info_lines = [
            f"town: {example['town_id']}",
            f"sequence: {example['sequence_id']}",
            f"patch: {example['patch_id']}",
            f"visible_surface_fraction: {example.get('visible_surface_fraction', float('nan')):.4f}",
            f"free_space_fraction: {example.get('free_space_fraction', float('nan')):.4f}",
            f"unknown_fraction: {example.get('unknown_fraction', float('nan')):.4f}",
            f"intrinsic_target: {example.get('intrinsic_target', float('nan')):.4f}",
            f"intrinsic_pred: {example.get('intrinsic_pred', float('nan')):.4f}",
        ]
        free_mask, free_scores = self._free_space_violation_scores(
            example.get("query_occupancy_logits"),
            example.get("query_labels_all"),
            example.get("query_ignore_mask"),
        )
        free_points = (
            query_points_all[free_mask]
            if isinstance(free_mask, np.ndarray) and free_mask.size
            else np.zeros((0, 3), dtype=np.float32)
        )
        prototype_summary_lines = [
            f"prototype_id: {example.get('code_index', 'n/a')}",
            f"prototype_entropy: {example.get('prototype_usage_entropy', float('nan')):.4f}",
        ]
        hybrid_path = render_hybrid_reconstruction_panel(
            corrupted_points=example["corrupted_points"],
            recon_points=example["recon_points"],
            clean_points=example["clean_points"],
            free_query_points=free_points,
            free_query_violation_scores=free_scores,
            intrinsic_pred=float(example.get("intrinsic_pred", float("nan"))),
            intrinsic_target=float(example.get("intrinsic_target", float("nan"))),
            prototype_summary_lines=prototype_summary_lines,
            info_lines=[
                f"score_target: {example['corruption_score_target']:.4f}",
                f"score_pred: {example['pred_score']:.4f}",
                f"gain: {example['gain']:.4f}",
                f"chamfer_before: {example['chamfer_before']:.4f}",
                f"chamfer_after: {example['chamfer_after']:.4f}",
            ],
            output_path=output_dir / f"{example['patch_id']}_hybrid_reconstruction_panel.png",
        )
        outputs: dict[str, Path] = {
            "hybrid_reconstruction_panel": hybrid_path,
        }
        # Panels are rendered when the underlying semantics are actually populated;
        # empty caches fall through via the data-presence guards below.
        if has_semantic_queries:
            outputs["visibility_panel"] = render_visibility_panel(
                clean_points=example["clean_points"],
                observed_points=example["observed_points"],
                surface_query_points=surface_query_points,
                free_query_points=free_query_points,
                unknown_query_points=unknown_query_points,
                info_lines=info_lines,
                output_path=output_dir / f"{example['patch_id']}_visibility_panel.png",
            )
        if has_visibility_semantics:
            outputs["visible_vs_hidden_panel"] = render_visible_vs_hidden_panel(
                observed_points=example["observed_points"],
                clean_points=example["clean_points"],
                visible_clean_points=visible_clean_points,
                hidden_clean_points=hidden_clean_points,
                recon_points=example["recon_points"],
                info_lines=[
                    f"patch: {example['patch_id']}",
                    f"visible_recon_chamfer: {example.get('visible_recon_chamfer_l1', float('nan')):.4f}",
                    f"hidden_completion_chamfer: {example.get('hidden_completion_chamfer_l1', float('nan')):.4f}",
                    f"hidden_completion_gain: {example.get('hidden_completion_gain', float('nan')):.4f}",
                ],
                output_path=output_dir / f"{example['patch_id']}_visible_vs_hidden_panel.png",
            )
        if len(free_points) > 0 or example.get("free_space_hard_negatives") is not None:
            outputs["free_space_error_panel"] = render_free_space_error_panel(
                corrupted_points=example["corrupted_points"],
                recon_points=example["recon_points"],
                clean_points=example["clean_points"],
                free_query_points=free_points,
                free_query_violation_scores=free_scores,
                free_space_hard_negatives=example.get("free_space_hard_negatives"),
                info_lines=[
                    f"patch: {example['patch_id']}",
                    f"free_space_violation_rate: {example.get('free_space_violation_rate', float('nan')):.4f}",
                    f"free_space_fp_rate: {example.get('free_space_fp_rate', float('nan')):.4f}",
                    f"hard_negative_count: {example.get('free_space_hard_negative_count', float('nan')):.0f}",
                ],
                output_path=output_dir / f"{example['patch_id']}_free_space_error_panel.png",
            )
        if np.isfinite(float(example.get("intrinsic_pred", float("nan")))) or np.isfinite(
            float(example.get("intrinsic_target", float("nan")))
        ):
            outputs["difficulty_calibration_panel"] = render_difficulty_calibration_panel(
                predicted=np.asarray([example.get("intrinsic_pred", float("nan"))], dtype=np.float32),
                target=np.asarray([example.get("intrinsic_target", float("nan"))], dtype=np.float32),
                info_lines=[
                    f"patch: {example['patch_id']}",
                    f"intrinsic_pred: {example.get('intrinsic_pred', float('nan')):.4f}",
                    f"intrinsic_target: {example.get('intrinsic_target', float('nan')):.4f}",
                    f"calibration_mae: {example.get('intrinsic_difficulty_calibration_mae', float('nan')):.4f}",
                ],
                output_path=output_dir / f"{example['patch_id']}_difficulty_calibration_panel.png",
            )
        return outputs

    def _select_sequence_ids_for_visualization(
        self,
        sequence_map_bank: dict[str, dict[str, list[Any]]],
    ) -> list[tuple[str, str]]:
        if not sequence_map_bank:
            return []
        sequence_rows = []
        for sequence_id, seq_bank in sequence_map_bank.items():
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "mean_gain": _safe_mean([float(x) for x in seq_bank.get("actual_gains", [])]),
                    "mean_intrinsic": _safe_mean([float(x) for x in seq_bank.get("intrinsic_targets", [])]),
                }
            )
        hardest = max(sequence_rows, key=lambda row: _safe_float(row["mean_intrinsic"], default=-1.0))
        best_gain = max(sequence_rows, key=lambda row: _safe_float(row["mean_gain"], default=-1e9))
        worst_gain = min(sequence_rows, key=lambda row: _safe_float(row["mean_gain"], default=1e9))
        ordered = [
            ("hardest_sequence", hardest["sequence_id"]),
            ("best_gain_sequence", best_gain["sequence_id"]),
            ("worst_gain_sequence", worst_gain["sequence_id"]),
        ]
        seen = set()
        unique = []
        for label, sequence_id in ordered:
            key = (label, sequence_id)
            if sequence_id in seen:
                continue
            seen.add(sequence_id)
            unique.append(key)
        return unique

    def _maybe_log_step_visualization(self, *, epoch: int) -> None:
        interval = int(self.train_config.get("step_visualization_interval_steps", 0) or 0)
        if interval <= 0 or not self.step_visual_examples or self.global_step % interval != 0:
            return
        current_loss_weights, _ = _merge_loss_weights_for_epoch(
            self.model_config.get("loss_weights", {}),
            self.train_config,
            epoch,
        )
        effective_weights = _effective_loss_weights(current_loss_weights)

        step_dir = self.visual_dir / f"step_{self.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        logged_images = {}
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for sample in self.step_visual_examples:
                batch = _collate_samples([sample])
                batch = self._move_batch_to_device(batch)
                loss_weights, _ = _merge_loss_weights_for_epoch(
                    self.model_config.get("loss_weights", {}),
                    self.train_config,
                    epoch,
                )
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs, _ = self._forward_batch(batch, loss_weights=loss_weights)
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
                render_patch_denoise_panel(
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
                    logged_images[f"viz_main_step/comparison/{patch_id}"] = self.wandb_module.Image(str(triptych_path))
                if self.is_hybrid_v2_family and batch["query_points_all"].shape[1] > 0:
                    intrinsic_pred = outputs["intrinsic_difficulty_pred"]
                    example = {
                        "town_id": batch["town_id"][0],
                        "sequence_id": batch["sequence_id"][0],
                        "patch_id": batch["patch_id"][0],
                        "observed_points": batch["observed_points"][0].detach().cpu().numpy(),
                        "corrupted_points": batch["corrupted_points"][0].detach().cpu().numpy(),
                        "clean_points": batch["clean_points"][0].detach().cpu().numpy(),
                        "recon_points": outputs["recon_points"][0].detach().cpu().numpy(),
                        "surface_query_points": batch["surface_query_points"][0].detach().cpu().numpy(),
                        "free_query_points": batch["free_query_points"][0].detach().cpu().numpy(),
                        "unknown_query_points": batch["unknown_query_points"][0].detach().cpu().numpy(),
                        "query_points_all": batch["query_points_all"][0].detach().cpu().numpy(),
                        "query_labels_all": batch["query_labels_all"][0].detach().cpu().numpy(),
                        "query_ignore_mask": batch["query_ignore_mask"][0].detach().cpu().numpy(),
                        "query_occupancy_logits": outputs["query_occupancy_logits"][0].detach().cpu().numpy()
                        if outputs.get("query_occupancy_logits") is not None
                        else None,
                        "corruption_score_target": float(batch["corruption_score_target"][0].detach().cpu()),
                        "pred_score": float(patch_score_pred_raw[0].detach().cpu()),
                        "gain": float(chamfer_before - chamfer_after),
                        "chamfer_before": chamfer_before,
                        "chamfer_after": chamfer_after,
                        "visible_surface_fraction": float(batch["visible_surface_fraction"][0].detach().cpu()),
                        "free_space_fraction": float(batch["free_space_fraction"][0].detach().cpu()),
                        "unknown_fraction": float(batch["unknown_fraction"][0].detach().cpu()),
                        "intrinsic_target": float(batch["intrinsic_patch_difficulty_target"][0].detach().cpu()),
                        "intrinsic_pred": float(intrinsic_pred[0].detach().cpu()) if intrinsic_pred is not None else float("nan"),
                        "intrinsic_difficulty_calibration_mae": float(
                            intrinsic_difficulty_calibration_mae(
                                intrinsic_pred[0:1].detach().cpu(),
                                batch["intrinsic_patch_difficulty_target"][0:1].detach().cpu(),
                            )
                        )
                        if intrinsic_pred is not None
                        else float("nan"),
                        "code_index": int(outputs["code_indices"][0].detach().cpu()) if outputs.get("code_indices") is not None else -1,
                        "prototype_usage_entropy": _safe_float(
                            outputs.get("codebook_stats", {}).get("usage_entropy")
                            if isinstance(outputs.get("codebook_stats"), dict)
                            else None,
                            default=float("nan"),
                        ),
                        "visible_clean_points": batch["visible_clean_points"][0].detach().cpu().numpy()
                        if isinstance(batch.get("visible_clean_points"), list)
                        else batch["visible_clean_points"][0].detach().cpu().numpy(),
                        "hidden_clean_points": batch["hidden_clean_points"][0].detach().cpu().numpy()
                        if isinstance(batch.get("hidden_clean_points"), list)
                        else batch["hidden_clean_points"][0].detach().cpu().numpy(),
                        "free_space_hard_negatives": batch["free_space_query_hard_negatives"][0].detach().cpu().numpy()
                        if isinstance(batch.get("free_space_query_hard_negatives"), list)
                        else (
                            batch["free_space_query_hard_negatives"][0].detach().cpu().numpy()
                            if batch.get("free_space_query_hard_negatives") is not None
                            else np.zeros((0, 3), dtype=np.float32)
                        ),
                        **self._sample_visibility_metrics(batch=batch, outputs=outputs, sample_idx=0),
                    }
                    v2_paths = self._render_v2_patch_artifacts(example=example, output_dir=step_dir)
                    if self.wandb_module is not None:
                        for image_name, image_path in v2_paths.items():
                            if image_name == "visibility_panel" and max(
                                effective_weights.get("occupancy_bce_loss", 0.0),
                                effective_weights.get("free_space_violation_loss", 0.0),
                            ) <= 0.0:
                                continue
                            logged_images[f"viz_main_step/{image_name}/{patch_id}"] = self.wandb_module.Image(str(image_path))
        if was_training:
            self.model.train()
        if logged_images:
            self._log(logged_images, step=self.global_step)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if hasattr(self.datasets.train_dataset, "set_epoch"):
            self.datasets.train_dataset.set_epoch(epoch)
        interval = int(self.train_config.get("log_interval", 10))
        stats = defaultdict(list)
        epoch_start = time.time()
        current_loss_weights, curriculum_stage = _merge_loss_weights_for_epoch(
            self.model_config.get("loss_weights", {}),
            self.train_config,
            epoch,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            iter_start = time.time()
            batch = self._move_batch_to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs, losses = self._forward_batch(batch, loss_weights=current_loss_weights)
            self.scaler.scale(losses["total_loss"] / float(self.grad_accum_steps)).backward()
            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))
            grad_norm = float("nan")
            if should_step:
                if self.amp_enabled:
                    self.scaler.unscale_(self.optimizer)
                grad_clip_norm = float(self.train_config.get("grad_clip_norm", 0.0) or 0.0)
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                grad_norm = _safe_grad_norm(self.model.parameters())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._maybe_log_step_visualization(epoch=epoch)

            batch_time = time.time() - iter_start
            chamfer_before = float(
                recon_chamfer_l1(
                    batch["corrupted_points"],
                    batch["clean_points"],
                )
                .detach()
                .cpu()
            )
            recon_to_corrupted = float(
                recon_chamfer_l1(
                    outputs["recon_points"],
                    batch["corrupted_points"],
                )
                .detach()
                .cpu()
            )
            batch_gain = chamfer_before - float(losses["recon_chamfer_loss"].detach().cpu())
            stats["corrupted_chamfer_l1"].append(chamfer_before)
            stats["recon_to_corrupted_chamfer_l1"].append(recon_to_corrupted)
            stats["total_loss"].append(float(losses["total_loss"].detach().cpu()))
            for key, value in losses.items():
                if key == "total_loss":
                    continue
                stats[key].append(float(value.detach().cpu()))
            stats["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
            if np.isfinite(grad_norm):
                stats["grad_norm"].append(float(grad_norm))
            stats["batch_time"].append(float(batch_time))
            stats["denoise_gain_chamfer"].append(batch_gain)
            batch_visibility_metrics = defaultdict(list)
            for sample_idx in range(batch["clean_points"].shape[0]):
                sample_metrics = self._sample_visibility_metrics(batch=batch, outputs=outputs, sample_idx=sample_idx)
                for key, value in sample_metrics.items():
                    batch_visibility_metrics[key].append(value)
            for key, values in batch_visibility_metrics.items():
                stats[key].append(_safe_mean([float(value) for value in values]))
            if batch_idx == 0 and hasattr(self.datasets.train_dataset, "current_corruption_severity_scale"):
                stats["corruption_severity_scale"].append(
                    float(self.datasets.train_dataset.current_corruption_severity_scale())
                )
            if self.ema_enabled:
                stats["ema_decay"].append(self.ema_decay)

            if (batch_idx + 1) % interval == 0:
                self._log(
                    self._format_interval_train_payload(
                        stats=stats,
                        interval=interval,
                        curriculum_stage=curriculum_stage,
                        effective_weights=_effective_loss_weights(current_loss_weights),
                    ),
                    step=self.global_step,
                )

        train_metrics = {f"train_{key}": _safe_mean(values) for key, values in stats.items()}
        if hasattr(self.datasets.train_dataset, "current_corruption_severity_scale"):
            train_metrics["train_corruption_severity_scale"] = float(
                self.datasets.train_dataset.current_corruption_severity_scale()
            )
        train_metrics["train_epoch_time"] = float(time.time() - epoch_start)
        train_metrics["train_curriculum_stage_index"] = _curriculum_stage_index(curriculum_stage)
        return train_metrics

    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        losses_acc = defaultdict(list)
        metrics_acc = defaultdict(list)
        val_start = time.time()
        sequence_map_bank: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        patch_examples: list[dict[str, Any]] = []
        prototype_examples_by_code: dict[int, dict[str, Any]] = {}
        clean_retrieval_embeddings: list[torch.Tensor] = []
        retrieval_embeddings: list[torch.Tensor] = []
        retrieval_bank_points: list[np.ndarray] = []
        retrieval_bank_ids: list[str] = []
        retrieval_bank_sequence_ids: list[str] = []
        score_preds = []
        score_targets = []
        gain_targets = []
        intrinsic_preds = []
        intrinsic_targets = []
        prototype_code_indices: list[torch.Tensor] = []
        codebook_entropy_values: list[float] = []

        fixed_patch_ids = set(self.train_config.get("fixed_visualization_patch_ids", []) or [])
        max_examples = int(self.train_config.get("max_visualization_examples", 3))
        current_loss_weights, _ = _merge_loss_weights_for_epoch(
            self.model_config.get("loss_weights", {}),
            self.train_config,
            epoch,
        )

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs, losses = self._forward_batch(batch, loss_weights=current_loss_weights)

                for key, value in losses.items():
                    losses_acc[key].append(float(value.detach().cpu()))

                point_defect_pred_raw = torch.expm1(torch.clamp(outputs["point_defect_pred"], min=0.0))
                patch_score_pred_raw = torch.expm1(torch.clamp(outputs["patch_score_pred"], min=0.0))
                intrinsic_pred_raw = outputs.get("intrinsic_difficulty_pred")
                if intrinsic_pred_raw is not None:
                    intrinsic_pred_raw = torch.clamp(intrinsic_pred_raw, min=0.0, max=1.0)
                metrics_acc["score_mae"].append(
                    float(score_mae(patch_score_pred_raw, batch["corruption_score_target"]).detach().cpu())
                )
                metrics_acc["point_defect_mae"].append(
                    float(point_defect_mae(point_defect_pred_raw, batch["point_defect_target"]).detach().cpu())
                )
                if intrinsic_pred_raw is not None:
                    metrics_acc["intrinsic_difficulty_mae"].append(
                        intrinsic_difficulty_mae(
                            intrinsic_pred_raw.detach().cpu(),
                            batch["intrinsic_patch_difficulty_target"].detach().cpu(),
                        )
                    )
                    metrics_acc["intrinsic_difficulty_calibration_mae"].append(
                        intrinsic_difficulty_calibration_mae(
                            intrinsic_pred_raw.detach().cpu(),
                            batch["intrinsic_patch_difficulty_target"].detach().cpu(),
                        )
                    )
                    intrinsic_preds.extend(intrinsic_pred_raw.detach().cpu().reshape(-1).tolist())
                    intrinsic_targets.extend(batch["intrinsic_patch_difficulty_target"].detach().cpu().reshape(-1).tolist())
                if outputs.get("query_occupancy_logits") is not None and batch["query_points_all"].shape[1] > 0:
                    metrics_acc["occupancy_iou_visible"].append(
                        occupancy_iou_visible(
                            outputs["query_occupancy_logits"].detach().cpu(),
                            batch["query_labels_all"].detach().cpu(),
                            batch["query_ignore_mask"].detach().cpu(),
                        )
                    )
                    metrics_acc["free_space_violation_rate"].append(
                        free_space_violation_rate(
                            outputs["query_occupancy_logits"].detach().cpu(),
                            batch["query_labels_all"].detach().cpu(),
                            batch["query_ignore_mask"].detach().cpu(),
                        )
                    )
                    metrics_acc["free_space_fp_rate"].append(
                        free_space_fp_rate(
                            outputs["query_occupancy_logits"].detach().cpu(),
                            batch["query_labels_all"].detach().cpu(),
                            batch["query_ignore_mask"].detach().cpu(),
                        )
                    )
                if outputs.get("code_indices") is not None:
                    prototype_code_indices.append(outputs["code_indices"].detach().cpu())
                if isinstance(outputs.get("codebook_stats"), dict):
                    codebook_entropy_values.append(
                        _safe_float(outputs["codebook_stats"].get("usage_entropy"), default=float("nan"))
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
                    recon_to_corrupted = float(
                        recon_chamfer_l1(
                            outputs["recon_points"][sample_idx : sample_idx + 1],
                            batch["corrupted_points"][sample_idx : sample_idx + 1],
                        )
                        .detach()
                        .cpu()
                    )
                    gain = chamfer_before - chamfer_after
                    metrics_acc["recon_chamfer_l1"].append(chamfer_after)
                    metrics_acc["corrupted_chamfer_l1"].append(chamfer_before)
                    metrics_acc["recon_to_corrupted_chamfer_l1"].append(recon_to_corrupted)
                    metrics_acc["recon_normal_cosine"].append(normal_cos)
                    metrics_acc["denoise_gain_chamfer"].append(gain)
                    sample_visibility_metrics = self._sample_visibility_metrics(
                        batch=batch,
                        outputs=outputs,
                        sample_idx=sample_idx,
                    )
                    for key in [
                        "visible_recon_chamfer_l1",
                        "hidden_completion_chamfer_l1",
                        "visible_recon_normal_cosine",
                        "hidden_completion_gain",
                    ]:
                        value = sample_visibility_metrics[key]
                        metrics_acc[key].append(value)
                    patch_center = batch["patch_center_world"][sample_idx].detach().cpu().numpy()
                    sequence_map_bank[sequence_id]["patch_centers"].append(patch_center)
                    sequence_map_bank[sequence_id]["pred_scores"].append(float(patch_score_pred_raw[sample_idx].detach().cpu()))
                    sequence_map_bank[sequence_id]["actual_gains"].append(float(gain))
                    sequence_map_bank[sequence_id]["visible_surface_fraction"].append(
                        float(batch["visible_surface_fraction"][sample_idx].detach().cpu())
                    )
                    sequence_map_bank[sequence_id]["free_space_fraction"].append(
                        float(batch["free_space_fraction"][sample_idx].detach().cpu())
                    )
                    sequence_map_bank[sequence_id]["intrinsic_targets"].append(
                        float(batch["intrinsic_patch_difficulty_target"][sample_idx].detach().cpu())
                    )
                    retrieval_bank_points.append(batch["clean_points"][sample_idx].detach().cpu().numpy())
                    retrieval_bank_ids.append(patch_id)
                    retrieval_bank_sequence_ids.append(sequence_id)
                    score_preds.append(float(patch_score_pred_raw[sample_idx].detach().cpu()))
                    score_targets.append(float(batch["corruption_score_target"][sample_idx].detach().cpu()))
                    gain_targets.append(float(gain))
                    if intrinsic_pred_raw is not None:
                        predicted_intrinsic = float(intrinsic_pred_raw[sample_idx].detach().cpu())
                    else:
                        predicted_intrinsic = float("nan")
                    code_index = (
                        int(outputs["code_indices"][sample_idx].detach().cpu())
                        if outputs.get("code_indices") is not None
                        else -1
                    )
                    if code_index >= 0 and code_index not in prototype_examples_by_code:
                        prototype_examples_by_code[code_index] = {
                            "patch_id": patch_id,
                            "code_index": code_index,
                            "clean_points": batch["clean_points"][sample_idx].detach().cpu().numpy(),
                            "intrinsic_pred": f"{predicted_intrinsic:.4f}",
                            "intrinsic_target": f"{float(batch['intrinsic_patch_difficulty_target'][sample_idx].detach().cpu()):.4f}",
                        }

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
                                "surface_query_points": batch["surface_query_points"][sample_idx].detach().cpu().numpy(),
                                "free_query_points": batch["free_query_points"][sample_idx].detach().cpu().numpy(),
                                "unknown_query_points": batch["unknown_query_points"][sample_idx].detach().cpu().numpy(),
                                "query_points_all": batch["query_points_all"][sample_idx].detach().cpu().numpy(),
                                "query_labels_all": batch["query_labels_all"][sample_idx].detach().cpu().numpy(),
                                "query_ignore_mask": batch["query_ignore_mask"][sample_idx].detach().cpu().numpy(),
                                "query_occupancy_logits": outputs["query_occupancy_logits"][sample_idx].detach().cpu().numpy()
                                if outputs.get("query_occupancy_logits") is not None
                                else None,
                                "corruption_score_target": float(batch["corruption_score_target"][sample_idx].detach().cpu()),
                                "pred_score": float(patch_score_pred_raw[sample_idx].detach().cpu()),
                                "intrinsic_target": float(batch["intrinsic_patch_difficulty_target"][sample_idx].detach().cpu()),
                                "intrinsic_pred": predicted_intrinsic,
                                "code_index": code_index,
                                "prototype_usage_entropy": _safe_float(
                                    outputs.get("codebook_stats", {}).get("usage_entropy")
                                    if isinstance(outputs.get("codebook_stats"), dict)
                                    else None,
                                    default=float("nan"),
                                ),
                                "visible_clean_points": batch["visible_clean_points"][sample_idx].detach().cpu().numpy()
                                if isinstance(batch.get("visible_clean_points"), list)
                                else batch["visible_clean_points"][sample_idx].detach().cpu().numpy(),
                                "hidden_clean_points": batch["hidden_clean_points"][sample_idx].detach().cpu().numpy()
                                if isinstance(batch.get("hidden_clean_points"), list)
                                else batch["hidden_clean_points"][sample_idx].detach().cpu().numpy(),
                                "free_space_hard_negatives": batch["free_space_query_hard_negatives"][sample_idx].detach().cpu().numpy()
                                if isinstance(batch.get("free_space_query_hard_negatives"), list)
                                else (
                                    batch["free_space_query_hard_negatives"][sample_idx].detach().cpu().numpy()
                                    if batch.get("free_space_query_hard_negatives") is not None
                                    else np.zeros((0, 3), dtype=np.float32)
                                ),
                                "visible_surface_fraction": float(batch["visible_surface_fraction"][sample_idx].detach().cpu()),
                                "free_space_fraction": float(batch["free_space_fraction"][sample_idx].detach().cpu()),
                                "unknown_fraction": float(batch["unknown_fraction"][sample_idx].detach().cpu()),
                                "gain": float(gain),
                                "chamfer_before": chamfer_before,
                                "chamfer_after": chamfer_after,
                                **sample_visibility_metrics,
                            }
                        )

        if clean_retrieval_embeddings and retrieval_embeddings:
            all_queries = torch.cat(retrieval_embeddings, dim=0)
            all_targets = torch.cat(clean_retrieval_embeddings, dim=0)
            metrics_acc["retrieval_top1_self_aligned"].append(retrieval_top1_self_aligned(all_queries, all_targets))
            metrics_acc["retrieval_top5_self_aligned"].append(retrieval_top5_self_aligned(all_queries, all_targets))
            metrics_acc["retrieval_top1_nonself"].append(
                retrieval_top1_nonself(
                    all_queries,
                    all_targets,
                    query_patch_ids=retrieval_bank_ids,
                    target_patch_ids=retrieval_bank_ids,
                    query_sequence_ids=retrieval_bank_sequence_ids,
                    target_sequence_ids=retrieval_bank_sequence_ids,
                )
            )
            metrics_acc["retrieval_top5_nonself"].append(
                retrieval_top5_nonself(
                    all_queries,
                    all_targets,
                    query_patch_ids=retrieval_bank_ids,
                    target_patch_ids=retrieval_bank_ids,
                    query_sequence_ids=retrieval_bank_sequence_ids,
                    target_sequence_ids=retrieval_bank_sequence_ids,
                )
            )
            metrics_acc["retrieval_top1_cross_sequence"].append(
                retrieval_top1_cross_sequence(
                    all_queries,
                    all_targets,
                    query_patch_ids=retrieval_bank_ids,
                    target_patch_ids=retrieval_bank_ids,
                    query_sequence_ids=retrieval_bank_sequence_ids,
                    target_sequence_ids=retrieval_bank_sequence_ids,
                )
            )
        else:
            warnings.warn("Retrieval metrics unavailable due to empty embedding bank.", stacklevel=2)
            metrics_acc["retrieval_top1_self_aligned"].append(float("nan"))
            metrics_acc["retrieval_top5_self_aligned"].append(float("nan"))
            metrics_acc["retrieval_top1_nonself"].append(float("nan"))
            metrics_acc["retrieval_top5_nonself"].append(float("nan"))
            metrics_acc["retrieval_top1_cross_sequence"].append(float("nan"))

        if intrinsic_preds and len(intrinsic_preds) >= 2:
            metrics_acc["intrinsic_difficulty_spearman"].append(
                intrinsic_difficulty_spearman(
                    torch.tensor(intrinsic_preds, dtype=torch.float32),
                    torch.tensor(intrinsic_targets, dtype=torch.float32),
                )
            )
        else:
            metrics_acc["intrinsic_difficulty_spearman"].append(float("nan"))
        if prototype_code_indices:
            all_code_indices = torch.cat(prototype_code_indices, dim=0)
            codebook_size = int(self.model_config.get("model", {}).get("codebook_size", 0) or 0)
            metrics_acc["prototype_usage_entropy"].append(
                prototype_usage_entropy(all_code_indices, codebook_size=codebook_size if codebook_size > 0 else None)
            )
        elif codebook_entropy_values:
            metrics_acc["prototype_usage_entropy"].append(_safe_mean(codebook_entropy_values))
        else:
            metrics_acc["prototype_usage_entropy"].append(float("nan"))

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
            prototype_examples=list(prototype_examples_by_code.values())[:6],
            intrinsic_preds=intrinsic_preds,
            intrinsic_targets=intrinsic_targets,
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
        prototype_examples: list[dict[str, Any]],
        intrinsic_preds: list[float],
        intrinsic_targets: list[float],
    ) -> None:
        if not patch_examples:
            return
        is_final_epoch = (epoch + 1) >= self.total_epochs
        is_first_epoch = epoch == self.start_epoch
        interval = self.epoch_visualization_interval
        if (
            not is_final_epoch
            and not is_first_epoch
            and interval > 1
            and (epoch % interval) != 0
        ):
            return
        current_loss_weights, _ = _merge_loss_weights_for_epoch(
            self.model_config.get("loss_weights", {}),
            self.train_config,
            epoch,
        )
        effective_weights = _effective_loss_weights(current_loss_weights)
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
            render_patch_denoise_panel(
                observed_points=example["observed_points"],
                corrupted_points=example["corrupted_points"],
                recon_points=example["recon_points"],
                clean_points=example["clean_points"],
                defect_scores=example["defect_scores"],
                info_lines=info_lines,
                output_path=epoch_dir / f"{example['patch_id']}_panel.png",
            )
            if self.wandb_module is not None:
                logged_images[f"viz_main/comparison/{example['patch_id']}"] = self.wandb_module.Image(str(triptych_path))
            if self.is_hybrid_v2_family and example.get("query_points_all") is not None:
                v2_paths = self._render_v2_patch_artifacts(example=example, output_dir=epoch_dir)
                if self.wandb_module is not None:
                    for image_name, image_path in v2_paths.items():
                        if image_name == "visibility_panel" and max(
                            effective_weights.get("occupancy_bce_loss", 0.0),
                            effective_weights.get("free_space_violation_loss", 0.0),
                        ) <= 0.0:
                            continue
                        logged_images[f"viz_main/{image_name}/{example['patch_id']}"] = self.wandb_module.Image(str(image_path))

        for map_label, sequence_id in self._select_sequence_ids_for_visualization(sequence_map_bank):
            seq_bank = sequence_map_bank[sequence_id]
            render_sequence_improvement_map(
                patch_centers_world=np.asarray(seq_bank["patch_centers"], dtype=np.float32),
                predicted_scores=np.asarray(seq_bank["pred_scores"], dtype=np.float32),
                actual_gains=np.asarray(seq_bank["actual_gains"], dtype=np.float32),
                sequence_id=sequence_id,
                output_path=epoch_dir / f"{sequence_id}_{map_label}_gain_map.png",
            )
            visibility_map_path = render_sequence_visibility_map(
                patch_centers_world=np.asarray(seq_bank["patch_centers"], dtype=np.float32),
                visible_surface_fraction=np.asarray(seq_bank["visible_surface_fraction"], dtype=np.float32),
                free_space_fraction=np.asarray(seq_bank["free_space_fraction"], dtype=np.float32),
                intrinsic_targets=np.asarray(seq_bank["intrinsic_targets"], dtype=np.float32),
                actual_gains=np.asarray(seq_bank["actual_gains"], dtype=np.float32),
                sequence_id=sequence_id,
                map_title=map_label.replace("_", " "),
                output_path=epoch_dir / f"{sequence_id}_{map_label}_visibility_map.png",
            )
            if self.wandb_module is not None:
                if max(
                    effective_weights.get("occupancy_bce_loss", 0.0),
                    effective_weights.get("free_space_violation_loss", 0.0),
                ) > 0.0:
                    logged_images[f"viz_main/sequence_visibility_map/{map_label}"] = self.wandb_module.Image(
                        str(visibility_map_path)
                    )

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
            render_retrieval_gallery(
                query_corrupted_points=patch_examples[query_idx]["corrupted_points"],
                target_clean_points=patch_examples[query_idx]["clean_points"],
                nearest_clean_points=retrieval_bank_points[nearest_idx] if nearest_idx < len(retrieval_bank_points) else patch_examples[query_idx]["clean_points"],
                info_lines=info_lines,
                output_path=epoch_dir / f"{patch_examples[query_idx]['patch_id']}_retrieval.png",
            )

        if prototype_examples:
            prototype_gallery_path = render_prototype_usage_gallery(
                prototype_examples=prototype_examples,
                output_path=epoch_dir / "prototype_usage_gallery.png",
            )
            if self.wandb_module is not None:
                logged_images["viz_main/prototype_usage_gallery"] = self.wandb_module.Image(str(prototype_gallery_path))

        if intrinsic_preds and intrinsic_targets:
            calibration_path = render_difficulty_calibration_panel(
                predicted=np.asarray(intrinsic_preds, dtype=np.float32),
                target=np.asarray(intrinsic_targets, dtype=np.float32),
                info_lines=[
                    f"epoch: {epoch}",
                    f"samples: {len(intrinsic_preds)}",
                    f"calibration_mae: {intrinsic_difficulty_calibration_mae(torch.tensor(intrinsic_preds), torch.tensor(intrinsic_targets)):.4f}",
                ],
                output_path=epoch_dir / "difficulty_calibration_panel.png",
            )
            if self.wandb_module is not None:
                logged_images["viz_main/difficulty_calibration_panel"] = self.wandb_module.Image(str(calibration_path))

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
            extra_payload=self._checkpoint_extra_payload(),
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
                extra_payload=self._checkpoint_extra_payload(),
            )
        recon_metric = val_metrics.get("val_recon_chamfer_l1", float("inf"))
        gain_metric = val_metrics.get("val_denoise_gain_chamfer", float("-inf"))
        composite_cfg = self.train_config.get("checkpoint_selection", {}) or {}
        composite_score = _score_from_weights(
            val_metrics,
            composite_cfg.get(
                "best_composite_weights",
                {
                    "val_denoise_gain_chamfer": 1.0,
                    "val_recon_chamfer_l1": -1.0,
                    "val_occupancy_iou_visible": 0.5,
                    "val_free_space_violation_rate": -0.5,
                    "val_intrinsic_difficulty_spearman": 0.2,
                },
            ),
        )
        visibility_score = _score_from_weights(
            val_metrics,
            composite_cfg.get(
                "best_visibility_weights",
                {
                    "val_occupancy_iou_visible": 1.0,
                    "val_free_space_violation_rate": -1.0,
                },
            ),
        )
        paper_score = _score_from_weights(
            val_metrics,
            composite_cfg.get(
                "best_paper_weights",
                {
                    "val_denoise_gain_chamfer": 1.0,
                    "val_recon_chamfer_l1": -1.0,
                    "val_intrinsic_difficulty_spearman": 0.25,
                    "val_occupancy_iou_visible": 0.5,
                    "val_free_space_violation_rate": -0.5,
                    "val_retrieval_top1_nonself": 0.25,
                },
            ),
        )
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
                extra_payload=self._checkpoint_extra_payload(),
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
                extra_payload=self._checkpoint_extra_payload(),
            )
        if composite_score > self.best_metrics["best_composite"]:
            self.best_metrics["best_composite"] = composite_score
            save_checkpoint(
                self.ckpt_dir / "best_composite.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
                extra_payload=self._checkpoint_extra_payload(),
            )
        if visibility_score > self.best_metrics["best_visibility"]:
            self.best_metrics["best_visibility"] = visibility_score
            save_checkpoint(
                self.ckpt_dir / "best_visibility.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
                extra_payload=self._checkpoint_extra_payload(),
            )
        if paper_score > self.best_metrics["best_paper"]:
            self.best_metrics["best_paper"] = paper_score
            save_checkpoint(
                self.ckpt_dir / "best_paper.pt",
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metrics=self.best_metrics,
                run_config=self.run_metadata,
                extra_payload=self._checkpoint_extra_payload(),
            )

    def fit(self) -> dict[str, Any]:
        epochs = int(self.train_config.get("epochs", 1))
        val_interval = int(self.train_config.get("val_interval", 1))
        history = []
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_one_epoch(epoch)
            current_loss_weights, curriculum_stage = _merge_loss_weights_for_epoch(
                self.model_config.get("loss_weights", {}),
                self.train_config,
                epoch,
            )
            effective_weights = _effective_loss_weights(current_loss_weights)
            if val_interval > 0 and ((epoch + 1) % val_interval == 0 or epoch == epochs - 1):
                with self._evaluation_model_scope():
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
                    extra_payload=self._checkpoint_extra_payload(),
                )
            history_entry = {
                "epoch": epoch,
                **self._compact_history_metrics(train_metrics, prefix="train_", effective_weights=effective_weights),
                **self._compact_history_metrics(val_metrics, prefix="val_", effective_weights=effective_weights),
            }
            history.append(history_entry)
            log_payload = {"epoch/index": epoch, "epoch/curriculum_stage_index": _curriculum_stage_index(curriculum_stage)}
            for key, value in train_metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                value = float(value)
                metric_name = key[len("train_"):] if key.startswith("train_") else key
                if metric_name in {"patch_score_loss", "batch_time", "curriculum_stage_index"}:
                    continue
                if self._metric_should_be_logged(metric_name, value, effective_weights=effective_weights):
                    log_payload[f"epoch/train/{metric_name}"] = value
            for key, value in val_metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                value = float(value)
                metric_name = key[len("val_"):] if key.startswith("val_") else key
                if metric_name == "patch_score_loss":
                    continue
                if self._metric_should_be_logged(metric_name, value, effective_weights=effective_weights):
                    log_payload[f"epoch/val/{metric_name}"] = value
            # Raw per-epoch loss means are already emitted above as
            # epoch/{split}/<loss_name>. Keep only the "contribution to total_loss"
            # view here (weight * mean) so the wandb run carries one canonical
            # representation per loss plus a single weighted-contribution group.
            train_raw = {
                key: float(train_metrics[f"train_{key}"])
                for key in self._active_raw_loss_keys(effective_weights)
                if f"train_{key}" in train_metrics
            }
            val_raw = {
                key: float(val_metrics[f"val_{key}"])
                for key in self._active_raw_loss_keys(effective_weights)
                if f"val_{key}" in val_metrics
            }
            for key, value in _loss_contribution_means(train_raw, effective_weights).items():
                short_key = key.removesuffix("_loss")
                log_payload[f"epoch/train_loss_weighted/{short_key}"] = value
            for key, value in _loss_contribution_means(val_raw, effective_weights).items():
                short_key = key.removesuffix("_loss")
                log_payload[f"epoch/val_loss_weighted/{short_key}"] = value
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
