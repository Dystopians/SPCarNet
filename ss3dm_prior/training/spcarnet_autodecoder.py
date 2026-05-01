"""SP-CarNet Stage 2 — auto-decoder training loop.

This is a standalone trainer for the implicit shape-field auto-decoder. It does
NOT reuse ``ss3dm_prior.engine.trainer`` because that trainer is patch-centric,
expects a single point-cloud output head, and aggregates batch losses in a way
that does not fit the per-object latent-code paradigm.

Design: ``docs/car_model/spcarnet_stage2_shape_field_design.md``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ss3dm_prior.data.spcarnet_object_dataset import (
    SPCarObjectDataset,
    collate_object_batch,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ShapeFieldModelConfig:
    field_kind: str = "occupancy"  # or "sdf"
    latent_dim: int = 256
    hidden_dim: int = 384
    depth: int = 6
    num_fourier_freqs: int = 32
    feature_dim: int = 0
    latent_init_std: float = 0.01

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ShapeFieldModelConfig":
        return ShapeFieldModelConfig(**{k: v for k, v in d.items() if k in ShapeFieldModelConfig.__dataclass_fields__})


@dataclass
class ShapeFieldLossConfig:
    w_surf: float = 1.0
    w_free: float = 1.0
    w_hard: float = 0.5
    w_mixed: float = 0.5
    w_zL2: float = 1e-4
    w_eik: float = 0.1
    w_normal: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ShapeFieldLossConfig":
        return ShapeFieldLossConfig(**{k: v for k, v in d.items() if k in ShapeFieldLossConfig.__dataclass_fields__})


@dataclass
class ShapeFieldTrainConfig:
    object_index_path: str = ""
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    val_splits: list[str] = field(default_factory=lambda: ["val"])
    output_dir: str = ""
    run_name: str = "spcarnet_autodecoder"

    seed: int = 0
    epochs: int = 200
    batch_size: int = 8
    queries_surface: int = 384
    queries_free: int = 384
    queries_hard: int = 128
    queries_mixed: int = 128
    queries_eikonal: int = 256

    lr_decoder: float = 5e-4
    lr_latent: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    log_every: int = 50
    eval_every_epochs: int = 10
    mc_resolution_smoke: int = 32
    mc_resolution_full: int = 64

    device: str = "cuda"
    num_workers: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ShapeFieldTrainConfig":
        kwargs = {k: v for k, v in d.items() if k in ShapeFieldTrainConfig.__dataclass_fields__}
        return ShapeFieldTrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# Per-object latent table
# ---------------------------------------------------------------------------


class LatentTable(torch.nn.Module):
    """Free-trained per-object latent codes.

    Parameters indexed by an object_id-> row mapping persisted alongside the
    checkpoint so eval scripts can recover the trained latent for a given car.
    """

    def __init__(self, *, num_objects: int, latent_dim: int, init_std: float = 0.01) -> None:
        super().__init__()
        codes = torch.randn(num_objects, latent_dim) * float(init_std)
        self.codes = torch.nn.Parameter(codes)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codes[indices]


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
    """Random subset (with replacement when undersized) of an (N, 3) tensor."""
    if mask is not None:
        arr = arr[mask]
    n = arr.shape[0]
    if n == 0:
        return np.zeros((count, arr.shape[-1]), dtype=arr.dtype)
    if n >= count:
        idx = rng.choice(n, size=count, replace=False)
    else:
        idx = rng.choice(n, size=count, replace=True)
    return arr[idx]


def assemble_query_batch(
    item: dict[str, Any],
    *,
    cfg: ShapeFieldTrainConfig,
    rng: np.random.Generator,
    field_kind: str,
) -> dict[str, torch.Tensor]:
    """Per-object query assembly: returns surface/free/hard/mixed point arrays.

    Output tensors are CPU float/int tensors; trainer moves to GPU.
    """
    surf_pool = item["clean_points_object"]  # (2048, 3)
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

    out = {
        "surface": torch.from_numpy(np.asarray(surf, dtype=np.float32)),
        "free": torch.from_numpy(np.asarray(free, dtype=np.float32)),
        "hard": torch.from_numpy(np.asarray(hard, dtype=np.float32)),
        "mixed_pts": torch.from_numpy(np.asarray(mixed_pts, dtype=np.float32)),
        "mixed_lab": torch.from_numpy(mixed_lab),
    }
    if field_kind == "sdf" and cfg.queries_eikonal > 0:
        eik = (rng.random((cfg.queries_eikonal, 3)) * 2.0 - 1.0).astype(np.float32)
        out["eikonal"] = torch.from_numpy(eik)
    return out


# ---------------------------------------------------------------------------
# Loss assembly
# ---------------------------------------------------------------------------


def compute_losses(
    decoder: SPCarShapeFieldDecoder,
    z_batch: torch.Tensor,
    queries: dict[str, torch.Tensor],
    *,
    loss_cfg: ShapeFieldLossConfig,
    field_kind: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the training loss for one batch.

    Parameters
    ----------
    queries:
        Dict with stacked tensors of shape ``(B, Q_*, 3)`` for the surface / free /
        hard / mixed sets, plus ``mixed_lab`` of shape ``(B, Q_mixed)`` and
        optional ``eikonal`` of shape ``(B, Q_eik, 3)``.
    """
    metrics: dict[str, float] = {}
    total = z_batch.new_zeros(())

    def _bce(x: torch.Tensor, target_value: float) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros(())
        target = torch.full_like(x, target_value)
        return F.binary_cross_entropy_with_logits(x, target)

    if field_kind == "occupancy":
        surf_logits = decoder(queries["surface"], z_batch)
        free_logits = decoder(queries["free"], z_batch)
        hard_logits = decoder(queries["hard"], z_batch)
        mixed_logits = decoder(queries["mixed_pts"], z_batch)
        mixed_target = queries["mixed_lab"]

        l_surf = _bce(surf_logits, 1.0)
        l_free = _bce(free_logits, 0.0)
        l_hard = _bce(hard_logits, 0.0)
        l_mixed = (
            F.binary_cross_entropy_with_logits(mixed_logits, mixed_target)
            if mixed_logits.numel() > 0
            else mixed_logits.new_zeros(())
        )
        total = (
            loss_cfg.w_surf * l_surf
            + loss_cfg.w_free * l_free
            + loss_cfg.w_hard * l_hard
            + loss_cfg.w_mixed * l_mixed
        )
        metrics["loss_surf"] = float(l_surf.detach().item())
        metrics["loss_free"] = float(l_free.detach().item())
        metrics["loss_hard"] = float(l_hard.detach().item())
        metrics["loss_mixed"] = float(l_mixed.detach().item())
    elif field_kind == "sdf":
        surf_sdf = decoder(queries["surface"], z_batch)
        free_sdf = decoder(queries["free"], z_batch)
        hard_sdf = decoder(queries["hard"], z_batch)
        l_surf = (surf_sdf**2).mean()
        l_free = F.relu(0.05 - free_sdf).mean()
        l_hard = F.relu(0.05 - hard_sdf).mean()
        total = loss_cfg.w_surf * l_surf + loss_cfg.w_free * l_free + loss_cfg.w_hard * l_hard
        metrics["loss_surf"] = float(l_surf.detach().item())
        metrics["loss_free"] = float(l_free.detach().item())
        metrics["loss_hard"] = float(l_hard.detach().item())

        if "eikonal" in queries and loss_cfg.w_eik > 0:
            eik_pts = queries["eikonal"].requires_grad_(True)
            eik_sdf = decoder(eik_pts, z_batch)
            grad = torch.autograd.grad(
                outputs=eik_sdf.sum(),
                inputs=eik_pts,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = grad.norm(dim=-1)
            l_eik = ((grad_norm - 1.0) ** 2).mean()
            total = total + loss_cfg.w_eik * l_eik
            metrics["loss_eikonal"] = float(l_eik.detach().item())
    else:
        raise ValueError(f"Unknown field_kind: {field_kind}")

    if loss_cfg.w_zL2 > 0 and z_batch.numel() > 0:
        l_zL2 = (z_batch.pow(2).sum(dim=-1) / max(z_batch.shape[-1], 1)).mean()
        total = total + loss_cfg.w_zL2 * l_zL2
        metrics["loss_zL2"] = float(l_zL2.detach().item())

    metrics["loss_total"] = float(total.detach().item())
    return total, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class StageTwoState:
    epoch: int = 0
    step: int = 0
    best_loss: float = math.inf


class ShapeFieldAutoDecoderTrainer:
    """Encapsulates the auto-decoder training loop."""

    def __init__(
        self,
        *,
        model_cfg: ShapeFieldModelConfig,
        loss_cfg: ShapeFieldLossConfig,
        train_cfg: ShapeFieldTrainConfig,
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
        self.object_id_to_row: dict[str, int] = {
            self.dataset.get_object_record(i)["object_id"]: i for i in range(len(self.dataset))
        }
        self.row_to_object_id: list[str] = [
            self.dataset.get_object_record(i)["object_id"] for i in range(len(self.dataset))
        ]

        self.decoder = SPCarShapeFieldDecoder(
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.hidden_dim,
            depth=model_cfg.depth,
            num_fourier_freqs=model_cfg.num_fourier_freqs,
            field_kind=model_cfg.field_kind,
            feature_dim=model_cfg.feature_dim,
        ).to(self.device)
        self.latents = LatentTable(
            num_objects=len(self.dataset),
            latent_dim=model_cfg.latent_dim,
            init_std=model_cfg.latent_init_std,
        ).to(self.device)

        self.opt_decoder = torch.optim.Adam(
            self.decoder.parameters(),
            lr=train_cfg.lr_decoder,
            weight_decay=train_cfg.weight_decay,
        )
        self.opt_latent = torch.optim.Adam(
            self.latents.parameters(),
            lr=train_cfg.lr_latent,
        )
        self.state = StageTwoState()
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
                        "model": asdict(model_cfg),
                        "loss": asdict(loss_cfg),
                        "train": asdict(train_cfg),
                    },
                    reinit=True,
                )
                self._wandb = wandb
            except Exception as exc:  # pragma: no cover
                print(f"[stage2] wandb init failed: {exc}", flush=True)
                self._wandb = None

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str) -> Path:
        path = self.output_dir / f"{name}.pt"
        payload = {
            "decoder_state_dict": self.decoder.state_dict(),
            "latent_table": self.latents.codes.detach().cpu(),
            "object_id_to_row": self.object_id_to_row,
            "row_to_object_id": self.row_to_object_id,
            "state": asdict(self.state),
            "model_cfg": asdict(self.model_cfg),
        }
        torch.save(payload, path)
        return path

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _draw_batch(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        order = self.rng.permutation(len(self.dataset))[: self.train_cfg.batch_size]
        items = [self.dataset[int(i)] for i in order]
        per_object_queries = [
            assemble_query_batch(it, cfg=self.train_cfg, rng=self.rng, field_kind=self.model_cfg.field_kind)
            for it in items
        ]
        stacked: dict[str, torch.Tensor] = {}
        for key in ("surface", "free", "hard", "mixed_pts", "mixed_lab", "eikonal"):
            if key not in per_object_queries[0]:
                continue
            stacked[key] = torch.stack([q[key] for q in per_object_queries], dim=0).to(self.device, non_blocking=True)
        indices = torch.tensor(order, dtype=torch.long, device=self.device)
        return indices, stacked

    # ------------------------------------------------------------------
    # Step / epoch
    # ------------------------------------------------------------------

    def step(self) -> dict[str, float]:
        self.decoder.train()
        self.latents.train()
        indices, queries = self._draw_batch()
        z_batch = self.latents(indices)
        total, metrics = compute_losses(
            self.decoder,
            z_batch,
            queries,
            loss_cfg=self.loss_cfg,
            field_kind=self.model_cfg.field_kind,
        )
        if not torch.isfinite(total):
            metrics["nonfinite"] = 1.0
            return metrics

        self.opt_decoder.zero_grad(set_to_none=True)
        self.opt_latent.zero_grad(set_to_none=True)
        total.backward()
        if self.train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.train_cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.latents.parameters(), self.train_cfg.grad_clip)
        self.opt_decoder.step()
        self.opt_latent.step()
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
                    f"[stage2] step={step_i+1}/{total_steps_target} "
                    f"epoch={cur_epoch} "
                    f"loss={last.get('loss_total', float('nan')):.4f} "
                    f"surf={last.get('loss_surf', float('nan')):.4f} "
                    f"free={last.get('loss_free', float('nan')):.4f}",
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
                and self.train_cfg.eval_every_epochs > 0
                and cur_epoch % self.train_cfg.eval_every_epochs == 0
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
            "history": history,
            "checkpoint_path": str(last_ckpt_path) if last_ckpt_path else None,
        }


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------


def load_configs(model_yaml: str | Path, train_yaml: str | Path) -> tuple[
    ShapeFieldModelConfig, ShapeFieldLossConfig, ShapeFieldTrainConfig
]:
    with open(model_yaml) as f:
        model_doc = yaml.safe_load(f) or {}
    with open(train_yaml) as f:
        train_doc = yaml.safe_load(f) or {}
    model_cfg = ShapeFieldModelConfig.from_dict(model_doc.get("model", model_doc))
    loss_cfg = ShapeFieldLossConfig.from_dict(model_doc.get("losses", {}))
    train_cfg = ShapeFieldTrainConfig.from_dict(train_doc.get("train", train_doc))
    return model_cfg, loss_cfg, train_cfg


__all__ = [
    "ShapeFieldModelConfig",
    "ShapeFieldLossConfig",
    "ShapeFieldTrainConfig",
    "LatentTable",
    "ShapeFieldAutoDecoderTrainer",
    "assemble_query_batch",
    "compute_losses",
    "load_configs",
]
