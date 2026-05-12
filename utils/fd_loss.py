"""
Frechet distance judge for ELA portfolio / alpha calibration.

Re-implementation of the core math from Jiawei Yang et al., "Representation
Frechet Loss for Visual Generation" (FD-Loss,
https://github.com/Jiawei-Yang/FD-Loss). Single-GPU, no distributed all-gather,
no streaming queue: used only as an additional non-regression selector signal
over small per-scene train batches, never as the main training loss.

API:
    FrozenReprConfig(model_name, pool_type, image_size, ...)
    FrozenReprModel(config, device)
        .encode(images_in_0_1)            -> [N, D] features (no grad)
        .feature_dim, .image_size, .pool_type
    frechet_distance(feats_a, feats_b)    -> dict with scalar fd / mean_term / trace_term
    frechet_distance_loss(feats, mu_ref, sigma_ref, sigma_ref_sqrt=None) -> scalar torch.Tensor
    empirical_gaussian(feats)             -> (mu, sigma)
    sqrtm_psd(matrix)                     -> PSD square root via eigh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FrozenReprConfig:
    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    pool_type: Literal["cls", "mean"] = "cls"
    image_size: int = 224


class FDBackboneUnavailable(RuntimeError):
    """Raised when the FD frozen backbone cannot be constructed (e.g. timm weights cannot be downloaded)."""


class FrozenReprModel:
    def __init__(
        self,
        config: FrozenReprConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        try:
            import timm
            from timm.data import resolve_data_config
        except Exception as exc:
            raise FDBackboneUnavailable(
                f"FD backbone requires `timm` but the import failed: {exc}. "
                f"Install `timm` or run without --fd_weight/--fd_strict."
            ) from exc

        cfg = config or FrozenReprConfig()
        self.cfg = cfg
        self.device = torch.device(device)

        try:
            model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0)
        except Exception as exc:
            raise FDBackboneUnavailable(
                f"Failed to load timm model '{cfg.model_name}' (likely a download / cache issue: {exc}). "
                f"Either pre-cache the weights under $HF_HOME or $TORCH_HOME, "
                f"or rerun without --fd_weight/--fd_strict."
            ) from exc
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        try:
            data_cfg = resolve_data_config({}, model=model)
            mean = tuple(data_cfg.get("mean", _IMAGENET_MEAN))
            std = tuple(data_cfg.get("std", _IMAGENET_STD))
            in_size = data_cfg.get("input_size", (3, cfg.image_size, cfg.image_size))
            image_size = int(in_size[-1])
        except Exception:
            mean = _IMAGENET_MEAN
            std = _IMAGENET_STD
            image_size = cfg.image_size

        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self._image_size = image_size

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def feature_dim(self) -> int:
        return int(self.model.num_features)

    @property
    def pool_type(self) -> str:
        return self.cfg.pool_type

    @torch.no_grad()
    def prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Pre-resize a single image to the backbone's input size.

        Use this when stacking many large frames before encode to keep the
        peak stack memory bounded by backbone resolution, not original size.
        Returns a CPU/GPU tensor of shape [3, image_size, image_size].
        """
        if image.ndim == 3:
            x = image.unsqueeze(0)
        elif image.ndim == 4 and image.shape[0] == 1:
            x = image
        else:
            raise ValueError(
                f"prepare expects [3,H,W] or [1,3,H,W], got {tuple(image.shape)}"
            )
        x = F.interpolate(
            x.float(),
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return x.squeeze(0)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: float in [0,1], shape [N,3,H,W] or [3,H,W]. Returns [N,D].

        If `images` are already at the backbone's input size the internal
        F.interpolate is a no-op; combine with `prepare` upstream to bound
        peak memory on large frames.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"encode expects [N,3,H,W] float[0,1] tensor, got {tuple(images.shape)}"
            )
        x = images.to(self.device, non_blocking=True).float()
        if x.shape[-2] != self._image_size or x.shape[-1] != self._image_size:
            x = F.interpolate(
                x,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        x = (x.clamp(0.0, 1.0) - self._mean) / self._std
        feats = self.model.forward_features(x)

        if isinstance(feats, dict):
            cls_tok = feats.get("x_norm_clstoken")
            patch_tok = feats.get("x_norm_patchtokens")
            if self.cfg.pool_type == "cls" and cls_tok is not None:
                return cls_tok
            if patch_tok is not None:
                return patch_tok.mean(dim=1)
            feats = next(iter(feats.values()))

        if feats.ndim == 3:
            return feats[:, 0, :] if self.cfg.pool_type == "cls" else feats.mean(dim=1)
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3))
        if feats.ndim == 2:
            return feats
        raise RuntimeError(
            f"Unexpected feature shape from {self.cfg.model_name}: {tuple(feats.shape)}"
        )


def empirical_gaussian(
    feats: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    if feats.ndim != 2:
        raise ValueError(f"empirical_gaussian expects [N,D], got {tuple(feats.shape)}")
    n = feats.shape[0]
    feats = feats.double()
    mu = feats.mean(dim=0)
    if n <= 1:
        sigma = torch.zeros(
            (feats.shape[1], feats.shape[1]), dtype=feats.dtype, device=feats.device
        )
    else:
        centered = feats - mu
        sigma = centered.t() @ centered / (n - 1)
    sigma = sigma + eps * torch.eye(
        sigma.shape[0], dtype=sigma.dtype, device=sigma.device
    )
    return mu, sigma


def sqrtm_psd(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric eigendecomposition-based PSD square root."""
    sym = 0.5 * (matrix + matrix.t())
    eigvals, eigvecs = torch.linalg.eigh(sym)
    if torch.is_complex(eigvals):
        eigvals = eigvals.real
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    sqrt_eig = torch.sqrt(eigvals_clamped + eps)
    return (eigvecs * sqrt_eig.unsqueeze(0)) @ eigvecs.t()


def frechet_distance(
    feats_a: torch.Tensor, feats_b: torch.Tensor, eps: float = 1e-6
) -> dict[str, float]:
    """Closed-form Frechet distance between two empirical Gaussians."""
    mu_a, sigma_a = empirical_gaussian(feats_a, eps=eps)
    mu_b, sigma_b = empirical_gaussian(feats_b, eps=eps)
    sqrt_sa = sqrtm_psd(sigma_a)
    middle = sqrt_sa @ sigma_b @ sqrt_sa
    sqrt_middle = sqrtm_psd(middle)
    mean_term = ((mu_a - mu_b) ** 2).sum()
    trace_term = (
        torch.trace(sigma_a) + torch.trace(sigma_b) - 2.0 * torch.trace(sqrt_middle)
    )
    fd = mean_term + trace_term
    return {
        "fd": float(fd.detach().cpu().item()),
        "mean_term": float(mean_term.detach().cpu().item()),
        "trace_term": float(trace_term.detach().cpu().item()),
        "n_a": int(feats_a.shape[0]),
        "n_b": int(feats_b.shape[0]),
    }


def frechet_distance_loss(
    feats: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-direction Frechet loss vs. a precomputed reference Gaussian.

    The math is differentiable (torch.linalg.eigh supports autograd), but
    `FrozenReprModel.encode` runs under @torch.no_grad and will block
    gradients from reaching `feats`. To use this as a training loss, run
    the backbone forward yourself without the no-grad wrapper.
    """
    if feats.ndim != 2:
        raise ValueError(f"frechet_distance_loss expects [N,D], got {tuple(feats.shape)}")
    feats = feats.double()
    mu_ref = mu_ref.double()
    sigma_ref = sigma_ref.double()
    mu, sigma = empirical_gaussian(feats, eps=eps)
    if sigma_ref_sqrt is None:
        sigma_ref_sqrt = sqrtm_psd(sigma_ref)
    middle = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
    sqrt_middle = sqrtm_psd(middle)
    mean_term = ((mu - mu_ref) ** 2).sum()
    trace_term = (
        torch.trace(sigma) + torch.trace(sigma_ref) - 2.0 * torch.trace(sqrt_middle)
    )
    return mean_term + trace_term
