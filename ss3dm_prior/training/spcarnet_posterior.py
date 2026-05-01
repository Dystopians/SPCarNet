"""SP-CarNet Stage 3 — amortised posterior encoder training loop.

Trains ``q(z | partial observation)`` against per-object Stage-2 latent codes
and the same occupancy / free-space query supervision used in Stage 2. The
Stage-2 ``SPCarShapeFieldDecoder`` is loaded from a checkpoint and frozen by
default.

Design: ``docs/car_model/spcarnet_stage3_posterior_encoder_design.md``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset
from ss3dm_prior.models.spcarnet_posterior import (
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PosteriorEncoderModelConfig:
    latent_dim: int = 256
    encoder_feature_dim: int = 256
    num_xattn_layers: int = 4
    num_self_attn_layers: int = 2
    num_latent_queries: int = 32
    attention_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    posterior_kind: str = "variational"  # or "deterministic"
    use_normals: bool = False
    use_conditioning_adapter: bool = True

    # Stage-2 decoder (frozen by default)
    decoder_checkpoint: str = ""
    decoder_finetune_enabled: bool = False
    decoder_finetune_tail_blocks: int = 2

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PosteriorEncoderModelConfig":
        kwargs = {k: v for k, v in d.items() if k in PosteriorEncoderModelConfig.__dataclass_fields__}
        return PosteriorEncoderModelConfig(**kwargs)


@dataclass
class PosteriorEncoderLossConfig:
    w_z: float = 10.0
    w_z_warmup: float = 2.0           # ramp w_z from this to w_z over warmup_epochs
    w_kl: float = 1.0e-3
    w_surf: float = 1.0
    w_free: float = 1.0
    w_hard: float = 0.5
    w_mixed: float = 0.5
    w_visible_chamfer: float = 0.0
    w_hidden_chamfer: float = 0.0
    free_bits_per_dim: float = 0.1     # nats; clamps KL contribution per dim from below
    kl_warmup_epochs: int = 10
    z_warmup_epochs: int = 10
    multi_sample_train_enabled: bool = False
    multi_sample_K: int = 4
    w_diversity: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PosteriorEncoderLossConfig":
        kwargs = {k: v for k, v in d.items() if k in PosteriorEncoderLossConfig.__dataclass_fields__}
        return PosteriorEncoderLossConfig(**kwargs)


@dataclass
class PosteriorEncoderTrainConfig:
    object_index_path: str = ""
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    val_splits: list[str] = field(default_factory=lambda: ["val"])
    output_dir: str = ""
    run_name: str = "spcarnet_posterior_encoder_v1"

    seed: int = 0
    epochs: int = 150
    batch_size: int = 16
    queries_surface: int = 384
    queries_free: int = 384
    queries_hard: int = 128
    queries_mixed: int = 128
    queries_partial: int = 768       # subset of partial_observed_points fed to encoder

    lr_encoder: float = 3e-4
    lr_decoder_finetune: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    log_every: int = 50
    eval_every_epochs: int = 10
    save_every_epochs: int = 10

    device: str = "cuda"
    num_workers: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PosteriorEncoderTrainConfig":
        kwargs = {k: v for k, v in d.items() if k in PosteriorEncoderTrainConfig.__dataclass_fields__}
        return PosteriorEncoderTrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _take_random_subset(
    arr: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if mask is not None:
        arr = arr[mask]
    n = arr.shape[0]
    if n == 0:
        return np.zeros((count, arr.shape[-1] if arr.ndim > 1 else 3), dtype=np.float32)
    if n >= count:
        idx = rng.choice(n, size=count, replace=False)
    else:
        idx = rng.choice(n, size=count, replace=True)
    return arr[idx]


def assemble_stage3_batch(
    item: dict[str, Any],
    *,
    cfg: PosteriorEncoderTrainConfig,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Assemble per-object tensors needed by the Stage-3 trainer."""
    surf_pool = item["clean_points_object"]
    if "surface_query_points" in item and item.get("surface_query_points") is not None:
        surf_pool = np.concatenate([surf_pool, item["surface_query_points"]], axis=0)
    free_pool = item.get("free_space_query_points")
    hard_pool = item.get("free_space_query_hard_negatives")
    qall_pts = item.get("occupancy_query_points")
    qall_lab = item.get("occupancy_query_labels")
    qall_ign = item.get("occupancy_query_ignore")

    surf = _take_random_subset(surf_pool, cfg.queries_surface, rng)
    free = _take_random_subset(free_pool if free_pool is not None else surf_pool, cfg.queries_free, rng)
    hard = _take_random_subset(hard_pool if hard_pool is not None else surf_pool, cfg.queries_hard, rng)

    if qall_pts is not None and qall_lab is not None:
        keep_mask = ~qall_ign if qall_ign is not None else np.ones(qall_pts.shape[0], dtype=bool)
        kept_pts = qall_pts[keep_mask]
        kept_lab = qall_lab[keep_mask]
        if kept_pts.shape[0] >= cfg.queries_mixed:
            idx = rng.choice(kept_pts.shape[0], size=cfg.queries_mixed, replace=False)
        else:
            idx = rng.choice(kept_pts.shape[0], size=cfg.queries_mixed, replace=True)
        mixed_pts = kept_pts[idx]
        mixed_lab = kept_lab[idx].astype(np.float32)
    else:
        mixed_pts = surf[: cfg.queries_mixed]
        mixed_lab = np.ones(cfg.queries_mixed, dtype=np.float32)

    partial = item.get("partial_observed_points")
    if partial is None or partial.shape[0] == 0:
        # Fallback: use clean_points (only happens when the cache lacks observations).
        partial = item["clean_points_object"]
    partial = _take_random_subset(partial, cfg.queries_partial, rng)

    return {
        "partial_points": torch.from_numpy(np.asarray(partial, dtype=np.float32)),
        "surface": torch.from_numpy(np.asarray(surf, dtype=np.float32)),
        "free": torch.from_numpy(np.asarray(free, dtype=np.float32)),
        "hard": torch.from_numpy(np.asarray(hard, dtype=np.float32)),
        "mixed_pts": torch.from_numpy(np.asarray(mixed_pts, dtype=np.float32)),
        "mixed_lab": torch.from_numpy(mixed_lab),
    }


# ---------------------------------------------------------------------------
# Loss assembly
# ---------------------------------------------------------------------------


def _bce(x: torch.Tensor, target_value: float) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros(())
    target = torch.full_like(x, target_value)
    return F.binary_cross_entropy_with_logits(x, target)


def _kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def compute_posterior_losses(
    *,
    completion_model: SPCarPosteriorCompletionModel,
    observation: dict[str, torch.Tensor],
    queries: dict[str, torch.Tensor],
    z_targets: torch.Tensor | None,
    z_target_mask: torch.Tensor | None,
    loss_cfg: PosteriorEncoderLossConfig,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Forward + assemble Stage-3 loss.

    Parameters
    ----------
    z_targets, z_target_mask:
        Per-batch Stage-2 latents and a mask indicating which entries are
        valid (train-split only). Non-train objects are masked out of L_z.

    Returns
    -------
    (total_loss, scalar_metrics, model_outputs)
    """
    out = completion_model(
        observation=observation,
        query_points={
            "surf": queries["surface"],
            "free": queries["free"],
            "hard": queries["hard"],
            "mixed": queries["mixed_pts"],
        },
        sample=None,
    )

    # ---------------- weight schedules ----------------
    kl_warmup = max(loss_cfg.kl_warmup_epochs, 1)
    z_warmup = max(loss_cfg.z_warmup_epochs, 1)
    kl_alpha = min(1.0, max(0.0, epoch / kl_warmup))
    z_alpha = min(1.0, max(0.0, epoch / z_warmup))
    w_kl_eff = loss_cfg.w_kl * kl_alpha
    w_z_eff = loss_cfg.w_z_warmup + z_alpha * (loss_cfg.w_z - loss_cfg.w_z_warmup)

    metrics: dict[str, float] = {
        "schedule/kl_alpha": kl_alpha,
        "schedule/w_z_eff": w_z_eff,
    }

    z_pred_mean = out["z_mean"]
    z_pred_logvar = out["z_logvar"]

    # ---------------- L_z (latent regression) ----------------
    if z_targets is not None and z_target_mask is not None and z_target_mask.any():
        diff = z_pred_mean - z_targets.detach()
        per_obj = (diff.pow(2).sum(dim=-1) / max(diff.shape[-1], 1))
        per_obj = per_obj * z_target_mask.to(per_obj.dtype)
        denom = z_target_mask.to(per_obj.dtype).sum().clamp_min(1.0)
        l_z = per_obj.sum() / denom
    else:
        l_z = z_pred_mean.new_zeros(())
    metrics["loss_z"] = float(l_z.detach().item())

    # ---------------- KL ----------------
    if z_pred_logvar is not None:
        kl_per_dim = _kl_to_standard_normal(z_pred_mean, z_pred_logvar)  # (B, d_z)
        if loss_cfg.free_bits_per_dim > 0.0:
            floor = loss_cfg.free_bits_per_dim
            kl_per_dim = torch.clamp(kl_per_dim, min=floor)
        l_kl = kl_per_dim.sum(dim=-1).mean()
        metrics["loss_kl"] = float(l_kl.detach().item())
        metrics["posterior/logvar_mean"] = float(z_pred_logvar.detach().mean().item())
        metrics["posterior/mu_norm"] = float(z_pred_mean.detach().norm(dim=-1).mean().item())
    else:
        l_kl = z_pred_mean.new_zeros(())
        metrics["loss_kl"] = 0.0

    # ---------------- BCE reconstruction terms ----------------
    l_surf = _bce(out["surf_logits"], 1.0)
    l_free = _bce(out["free_logits"], 0.0)
    l_hard = _bce(out["hard_logits"], 0.0)
    if out.get("mixed_logits") is not None and queries["mixed_lab"].numel() > 0:
        l_mixed = F.binary_cross_entropy_with_logits(out["mixed_logits"], queries["mixed_lab"])
    else:
        l_mixed = z_pred_mean.new_zeros(())

    metrics["loss_surf"] = float(l_surf.detach().item())
    metrics["loss_free"] = float(l_free.detach().item())
    metrics["loss_hard"] = float(l_hard.detach().item())
    metrics["loss_mixed"] = float(l_mixed.detach().item())

    total = (
        w_z_eff * l_z
        + w_kl_eff * l_kl
        + loss_cfg.w_surf * l_surf
        + loss_cfg.w_free * l_free
        + loss_cfg.w_hard * l_hard
        + loss_cfg.w_mixed * l_mixed
    )

    metrics["loss_total"] = float(total.detach().item())
    return total, metrics, out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class StageThreeState:
    epoch: int = 0
    step: int = 0
    best_loss: float = math.inf


class PosteriorEncoderTrainer:
    """Training loop for the Stage-3 amortised posterior encoder."""

    def __init__(
        self,
        *,
        model_cfg: PosteriorEncoderModelConfig,
        loss_cfg: PosteriorEncoderLossConfig,
        train_cfg: PosteriorEncoderTrainConfig,
    ) -> None:
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg

        torch.manual_seed(train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_cfg.seed)
        np.random.seed(train_cfg.seed)

        self.device = torch.device(train_cfg.device)
        self.dataset = SPCarObjectDataset(
            object_index_path=train_cfg.object_index_path,
            splits=tuple(train_cfg.train_splits),
            estimate_symmetry=False,
        )

        # ---------------- decoder + Stage-2 latent table ----------------
        if not model_cfg.decoder_checkpoint:
            raise ValueError("model_cfg.decoder_checkpoint must point to the Stage-2 checkpoint")
        ckpt = torch.load(model_cfg.decoder_checkpoint, map_location=self.device)
        decoder_model_cfg = ckpt.get("model_cfg", {})
        self.decoder = SPCarShapeFieldDecoder(
            latent_dim=int(decoder_model_cfg.get("latent_dim", model_cfg.latent_dim)),
            hidden_dim=int(decoder_model_cfg.get("hidden_dim", 384)),
            depth=int(decoder_model_cfg.get("depth", 6)),
            num_fourier_freqs=int(decoder_model_cfg.get("num_fourier_freqs", 32)),
            field_kind=str(decoder_model_cfg.get("field_kind", "occupancy")),
            feature_dim=int(decoder_model_cfg.get("feature_dim", 0)),
        ).to(self.device)
        self.decoder.load_state_dict(ckpt["decoder_state_dict"])

        # Stage-2 latent table — registered as a buffer; never updated.
        latent_table = ckpt["latent_table"].to(self.device)
        self._latent_table = latent_table  # (N_train_stage2, d_z)
        self._stage2_object_id_to_row: dict[str, int] = ckpt["object_id_to_row"]

        if int(decoder_model_cfg.get("latent_dim", model_cfg.latent_dim)) != model_cfg.latent_dim:
            raise ValueError(
                "model_cfg.latent_dim mismatches the Stage-2 decoder's latent_dim"
            )

        # ---------------- encoder + completion wrapper ----------------
        self.encoder = SPCarPosteriorEncoder(
            latent_dim=model_cfg.latent_dim,
            feature_dim=model_cfg.encoder_feature_dim,
            num_xattn_layers=model_cfg.num_xattn_layers,
            num_self_attn_layers=model_cfg.num_self_attn_layers,
            num_latent_queries=model_cfg.num_latent_queries,
            attention_heads=model_cfg.attention_heads,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            posterior_kind=model_cfg.posterior_kind,
            use_normals=model_cfg.use_normals,
            use_conditioning_adapter=model_cfg.use_conditioning_adapter,
        ).to(self.device)
        self.completion = SPCarPosteriorCompletionModel(
            encoder=self.encoder,
            decoder=self.decoder,
            decoder_finetune_enabled=model_cfg.decoder_finetune_enabled,
            decoder_finetune_tail_blocks=model_cfg.decoder_finetune_tail_blocks,
        ).to(self.device)

        encoder_params = list(self.encoder.parameters())
        decoder_trainable = [p for p in self.decoder.parameters() if p.requires_grad]
        if decoder_trainable:
            self.optimiser = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": train_cfg.lr_encoder},
                    {"params": decoder_trainable, "lr": train_cfg.lr_decoder_finetune},
                ],
                weight_decay=train_cfg.weight_decay,
            )
        else:
            self.optimiser = torch.optim.AdamW(
                encoder_params,
                lr=train_cfg.lr_encoder,
                weight_decay=train_cfg.weight_decay,
            )

        self.state = StageThreeState()
        self.rng = np.random.default_rng(train_cfg.seed)

        self.output_dir = Path(train_cfg.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if os.environ.get("WANDB_MODE", "online") != "disabled":
            try:
                import wandb  # type: ignore

                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "spcarnet"),
                    name=train_cfg.run_name,
                    dir=str(self.output_dir),
                    config={
                        "stage": 3,
                        "model": asdict(model_cfg),
                        "loss": asdict(loss_cfg),
                        "train": asdict(train_cfg),
                    },
                    reinit=True,
                )
                self._wandb = wandb
            except Exception as exc:  # pragma: no cover
                print(f"[stage3] wandb init failed: {exc}", flush=True)
                self._wandb = None

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str) -> Path:
        path = self.output_dir / f"{name}.pt"
        payload = {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "stage2_object_id_to_row": self._stage2_object_id_to_row,
            "stage2_latent_table": self._latent_table.detach().cpu(),
            "state": asdict(self.state),
            "model_cfg": asdict(self.model_cfg),
            "decoder_finetune_enabled": self.model_cfg.decoder_finetune_enabled,
        }
        torch.save(payload, path)
        return path

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _draw_batch(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        order = self.rng.permutation(len(self.dataset))[: self.train_cfg.batch_size]
        items = [self.dataset[int(i)] for i in order]
        per_obj = [
            assemble_stage3_batch(it, cfg=self.train_cfg, rng=self.rng) for it in items
        ]
        observation = {
            "partial_points": torch.stack([p["partial_points"] for p in per_obj], dim=0).to(
                self.device, non_blocking=True
            )
        }
        queries = {
            key: torch.stack([p[key] for p in per_obj], dim=0).to(self.device, non_blocking=True)
            for key in ("surface", "free", "hard", "mixed_pts", "mixed_lab")
        }
        # Stage-2 latent supervision: only for train-split objects with a row.
        z_targets_list: list[torch.Tensor] = []
        mask_list: list[float] = []
        for it in items:
            row = self._stage2_object_id_to_row.get(it["object_id"])
            if row is not None and 0 <= row < self._latent_table.shape[0]:
                z_targets_list.append(self._latent_table[row])
                mask_list.append(1.0)
            else:
                z_targets_list.append(torch.zeros(self.model_cfg.latent_dim, device=self.device))
                mask_list.append(0.0)
        z_targets = torch.stack(z_targets_list, dim=0)
        z_mask = torch.tensor(mask_list, dtype=torch.float32, device=self.device)
        return observation, queries, z_targets, z_mask

    # ------------------------------------------------------------------
    # Step / fit
    # ------------------------------------------------------------------

    def step(self) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train() if any(p.requires_grad for p in self.decoder.parameters()) else self.decoder.eval()

        observation, queries, z_targets, z_mask = self._draw_batch()
        total, metrics, _ = compute_posterior_losses(
            completion_model=self.completion,
            observation=observation,
            queries=queries,
            z_targets=z_targets,
            z_target_mask=z_mask,
            loss_cfg=self.loss_cfg,
            epoch=self.state.epoch,
        )
        if not torch.isfinite(total):
            metrics["nonfinite"] = 1.0
            return metrics

        self.optimiser.zero_grad(set_to_none=True)
        total.backward()
        if self.train_cfg.grad_clip > 0:
            trainable = [p for p in self.encoder.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.train_cfg.grad_clip)
            decoder_trainable = [p for p in self.decoder.parameters() if p.requires_grad]
            if decoder_trainable:
                torch.nn.utils.clip_grad_norm_(decoder_trainable, self.train_cfg.grad_clip)
        self.optimiser.step()
        self.state.step += 1
        return metrics

    def fit(self, *, max_steps: int | None = None) -> dict[str, Any]:
        steps_per_epoch = max(1, len(self.dataset) // self.train_cfg.batch_size)
        history: list[dict[str, float]] = []
        total_steps_target = (
            max_steps if max_steps is not None else self.train_cfg.epochs * steps_per_epoch
        )
        started = time.time()
        last_ckpt_path: Path | None = None
        for step_i in range(total_steps_target):
            metrics = self.step()
            history.append(metrics)

            cur_epoch = (step_i + 1) // steps_per_epoch
            self.state.epoch = cur_epoch

            if (step_i + 1) % self.train_cfg.log_every == 0:
                last = metrics
                print(
                    f"[stage3] step={step_i+1}/{total_steps_target} epoch={cur_epoch} "
                    f"loss={last.get('loss_total', float('nan')):.4f} "
                    f"z={last.get('loss_z', float('nan')):.4f} "
                    f"surf={last.get('loss_surf', float('nan')):.4f} "
                    f"free={last.get('loss_free', float('nan')):.4f} "
                    f"kl={last.get('loss_kl', float('nan')):.4f}",
                    flush=True,
                )
                if self._wandb is not None:
                    self._wandb.log(
                        {f"train/{k}": v for k, v in last.items() if isinstance(v, (int, float))},
                        step=step_i + 1,
                    )

            if (
                steps_per_epoch > 0
                and (step_i + 1) % steps_per_epoch == 0
                and self.train_cfg.save_every_epochs > 0
                and cur_epoch % self.train_cfg.save_every_epochs == 0
            ):
                last_ckpt_path = self.save_checkpoint("checkpoint_last")
                if self._wandb is not None:
                    self._wandb.log(
                        {"train/epoch": cur_epoch, "train/checkpoint_step": step_i + 1},
                        step=step_i + 1,
                    )

        last_ckpt_path = self.save_checkpoint("checkpoint_last")
        if self._wandb is not None:
            self._wandb.finish()

        return {
            "elapsed": time.time() - started,
            "n_steps": len(history),
            "checkpoint_path": str(last_ckpt_path) if last_ckpt_path else None,
        }


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------


def load_configs(model_yaml: str | Path, train_yaml: str | Path) -> tuple[
    PosteriorEncoderModelConfig, PosteriorEncoderLossConfig, PosteriorEncoderTrainConfig
]:
    with open(model_yaml) as f:
        model_doc = yaml.safe_load(f) or {}
    with open(train_yaml) as f:
        train_doc = yaml.safe_load(f) or {}
    model_cfg = PosteriorEncoderModelConfig.from_dict(model_doc.get("model", model_doc))
    loss_cfg = PosteriorEncoderLossConfig.from_dict(model_doc.get("losses", {}))
    train_cfg = PosteriorEncoderTrainConfig.from_dict(train_doc.get("train", train_doc))
    return model_cfg, loss_cfg, train_cfg


__all__ = [
    "PosteriorEncoderModelConfig",
    "PosteriorEncoderLossConfig",
    "PosteriorEncoderTrainConfig",
    "PosteriorEncoderTrainer",
    "assemble_stage3_batch",
    "compute_posterior_losses",
    "load_configs",
]
