import json
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class TriangleState(IntEnum):
    ACTIVE = 0
    PROTECTED = 1
    DEAD = 2
    CANDIDATE = 3
    SUSPICIOUS = 4
    PRUNED = 5


@dataclass
class TriangleStats:
    """
    Per-triangle statistics for PRISM-Prune collection.

    Notes:
    - Most fields are EMA statistics so they remain stable under noisy per-view updates.
    - `view_direction_histogram` is an EMA histogram over azimuth bins.
    - `triangle_state` stores TriangleState values as int64 tensor.
    """

    vis_count_ema: torch.Tensor
    projected_area_ema: torch.Tensor
    grad_pos_norm_ema: torch.Tensor
    grad_app_norm_ema: torch.Tensor
    grad_norm_var_ema: torch.Tensor
    view_direction_histogram: torch.Tensor
    birth_iter: torch.Tensor
    last_topology_change_iter: torch.Tensor
    active_mask: torch.Tensor
    triangle_state: torch.Tensor


class TriangleStatsManager:
    """
    Collects neutral per-triangle observables for PRISM.

    This manager does not perform any prune decision. It only tracks statistics.
    """

    def __init__(
        self,
        num_triangles: int,
        device: torch.device,
        init_iter: int = 0,
        ema_decay: float = 0.95,
        view_hist_bins: int = 8,
    ):
        self.device = device
        self.ema_decay = float(max(0.0, min(ema_decay, 0.9999)))
        self.view_hist_bins = int(max(1, view_hist_bins))
        self.num_triangles = int(max(0, num_triangles))
        self.stats = self._make_empty_stats(self.num_triangles, init_iter=int(init_iter))
        # Internal EMA mean for uncertainty (variance) tracking.
        self._grad_total_mean_ema = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)
        self.last_seen_iteration = int(init_iter)
        self.last_global_topology_change_iter = int(init_iter)

    def _make_empty_stats(self, num_triangles: int, init_iter: int) -> TriangleStats:
        n = int(max(0, num_triangles))
        return TriangleStats(
            vis_count_ema=torch.zeros((n,), dtype=torch.float32, device=self.device),
            projected_area_ema=torch.zeros((n,), dtype=torch.float32, device=self.device),
            grad_pos_norm_ema=torch.zeros((n,), dtype=torch.float32, device=self.device),
            grad_app_norm_ema=torch.zeros((n,), dtype=torch.float32, device=self.device),
            grad_norm_var_ema=torch.zeros((n,), dtype=torch.float32, device=self.device),
            view_direction_histogram=torch.zeros((n, self.view_hist_bins), dtype=torch.float32, device=self.device),
            birth_iter=torch.full((n,), int(init_iter), dtype=torch.int64, device=self.device),
            last_topology_change_iter=torch.full((n,), int(init_iter), dtype=torch.int64, device=self.device),
            active_mask=torch.ones((n,), dtype=torch.bool, device=self.device),
            triangle_state=torch.full((n,), int(TriangleState.ACTIVE), dtype=torch.int64, device=self.device),
        )

    def on_topology_change(self, new_num_triangles: int, iteration: int):
        """
        Reset stats when topology changes.

        Fallback policy for now: full reset to avoid stale triangle-ID associations.
        """
        iter_now = int(iteration)
        self.last_seen_iteration = iter_now
        self.last_global_topology_change_iter = iter_now
        self.num_triangles = int(max(0, new_num_triangles))
        self.stats = self._make_empty_stats(self.num_triangles, init_iter=iter_now)
        self._grad_total_mean_ema = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)

    def sync_iteration(self, iteration: int):
        self.last_seen_iteration = int(iteration)

    def _ema_update(self, old: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        d = self.ema_decay
        return d * old + (1.0 - d) * new_value

    def update_visibility_from_render(self, render_pkg: Dict, triangles, viewpoint_cam=None, iteration: Optional[int] = None) -> bool:
        """
        Update visibility-related stats.

        Real signal path:
        - uses `triangle_was_rendered` if available (per-triangle coverage count)
        - uses `scaling` as projected-area proxy if available
        """
        if iteration is not None:
            self.sync_iteration(int(iteration))
        tri_cov = render_pkg.get("triangle_was_rendered", None)
        if tri_cov is None:
            return False

        tri_cov = tri_cov.detach().to(self.device).float()
        if tri_cov.numel() != self.num_triangles:
            iter_now = int(self.last_seen_iteration if iteration is None else iteration)
            self.on_topology_change(new_num_triangles=int(tri_cov.numel()), iteration=iter_now)
            tri_cov = tri_cov[: self.num_triangles]

        vis_signal = (tri_cov > 0).float()
        self.stats.vis_count_ema = self._ema_update(self.stats.vis_count_ema, vis_signal)

        area_proxy = render_pkg.get("scaling", None)
        if area_proxy is None:
            area_signal = tri_cov
        else:
            area_signal = area_proxy.detach().to(self.device).float()
            if area_signal.numel() != self.num_triangles:
                # Fallback to coverage proxy if scaling shape mismatches.
                area_signal = tri_cov
        self.stats.projected_area_ema = self._ema_update(self.stats.projected_area_ema, area_signal)

        # Optional view-direction histogram update.
        if viewpoint_cam is not None and triangles is not None and self.num_triangles > 0:
            visible_ids = torch.nonzero(tri_cov > 0, as_tuple=True)[0]
            if visible_ids.numel() > 0:
                tri_idx = triangles._triangle_indices.long()
                tri_pts = triangles.vertices[tri_idx[visible_ids]]
                centroids = tri_pts.mean(dim=1)
                cam_center = viewpoint_cam.camera_center.detach()
                dirs = centroids - cam_center.unsqueeze(0)
                dirs = F.normalize(dirs, dim=1, eps=1e-8)
                azimuth = torch.atan2(dirs[:, 1], dirs[:, 0])  # [-pi, pi]
                bins = ((azimuth + torch.pi) / (2.0 * torch.pi) * float(self.view_hist_bins)).long()
                bins = bins.clamp(0, self.view_hist_bins - 1)
                one_hot = F.one_hot(bins, num_classes=self.view_hist_bins).to(torch.float32)

                self.stats.view_direction_histogram.mul_(self.ema_decay)
                self.stats.view_direction_histogram[visible_ids] += (1.0 - self.ema_decay) * one_hot

        return True

    def update_gradient_stats(self, triangles, iteration: Optional[int] = None) -> bool:
        """
        Update gradient-based stats from current backward pass.

        Fallback behavior:
        - if a gradient source is missing, corresponding update uses zeros.
        """
        if iteration is not None:
            self.sync_iteration(int(iteration))
        if self.num_triangles <= 0:
            return False

        tri_idx = triangles._triangle_indices.long()
        if tri_idx.shape[0] != self.num_triangles:
            iter_now = int(self.last_seen_iteration if iteration is None else iteration)
            self.on_topology_change(new_num_triangles=int(tri_idx.shape[0]), iteration=iter_now)
            tri_idx = triangles._triangle_indices.long()

        v_grad = triangles.vertices.grad
        if v_grad is None:
            pos_per_vertex = torch.zeros((triangles.vertices.shape[0],), dtype=torch.float32, device=self.device)
        else:
            pos_per_vertex = torch.linalg.norm(v_grad.detach(), dim=1).to(torch.float32)
        grad_pos_tri = pos_per_vertex[tri_idx].mean(dim=1)
        self.stats.grad_pos_norm_ema = self._ema_update(self.stats.grad_pos_norm_ema, grad_pos_tri)

        dc_grad = triangles._features_dc.grad
        rest_grad = triangles._features_rest.grad
        app_sq = torch.zeros((triangles.vertices.shape[0],), dtype=torch.float32, device=self.device)
        if dc_grad is not None:
            app_sq += dc_grad.detach().pow(2).sum(dim=(1, 2)).to(torch.float32)
        if rest_grad is not None:
            app_sq += rest_grad.detach().pow(2).sum(dim=(1, 2)).to(torch.float32)
        grad_app_vertex = torch.sqrt(torch.clamp(app_sq, min=0.0))
        grad_app_tri = grad_app_vertex[tri_idx].mean(dim=1)
        self.stats.grad_app_norm_ema = self._ema_update(self.stats.grad_app_norm_ema, grad_app_tri)

        grad_total = grad_pos_tri + grad_app_tri
        new_mean = self._ema_update(self._grad_total_mean_ema, grad_total)
        residual_sq = (grad_total - new_mean).pow(2)
        self.stats.grad_norm_var_ema = self._ema_update(self.stats.grad_norm_var_ema, residual_sq)
        self._grad_total_mean_ema = new_mean
        return True

    def get_age(self, current_iter: int) -> torch.Tensor:
        return torch.full_like(self.stats.birth_iter, int(current_iter)) - self.stats.birth_iter

    def get_debug_summary(self, iteration: int) -> Dict[str, float]:
        age = self.get_age(iteration).to(torch.float32)
        return {
            "num_triangles": float(self.num_triangles),
            "vis_count_ema_mean": float(self.stats.vis_count_ema.mean().item()) if self.num_triangles > 0 else 0.0,
            "projected_area_ema_mean": float(self.stats.projected_area_ema.mean().item()) if self.num_triangles > 0 else 0.0,
            "grad_pos_norm_ema_mean": float(self.stats.grad_pos_norm_ema.mean().item()) if self.num_triangles > 0 else 0.0,
            "grad_app_norm_ema_mean": float(self.stats.grad_app_norm_ema.mean().item()) if self.num_triangles > 0 else 0.0,
            "grad_norm_var_ema_mean": float(self.stats.grad_norm_var_ema.mean().item()) if self.num_triangles > 0 else 0.0,
            "age_mean": float(age.mean().item()) if self.num_triangles > 0 else 0.0,
        }

    def maybe_save_debug_json(self, output_dir: str, iteration: int):
        os.makedirs(output_dir, exist_ok=True)
        payload = self.get_debug_summary(iteration=iteration)
        payload["iteration"] = int(iteration)
        path = os.path.join(output_dir, f"triangle_stats_iter_{int(iteration):06d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
