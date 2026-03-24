import os
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class GroundAssociationConfig:
    min_observations: int
    min_ground_ratio: float
    min_view_consistency: float
    per_view_ground_ratio: float
    boundary_margin: float
    confidence_min: float
    use_cache: bool
    cache_file: str
    cache_every: int
    debug_every: int
    debug_dir: str
    hist_bins: int


def _resize_mask(mask_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if int(mask_hw.shape[0]) == out_h and int(mask_hw.shape[1]) == out_w:
        return mask_hw
    m = mask_hw.float().unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(out_h, out_w), mode="nearest")
    return (m.squeeze(0).squeeze(0) > 0.5)


class GroundAssociationTracker:
    """
    Multi-view robust aggregation of image-space ground supervision into mesh-space labels.
    """

    def __init__(self, num_triangles: int, device: torch.device, model_path: str, cfg: GroundAssociationConfig):
        self.num_triangles = int(num_triangles)
        self.device = device
        self.model_path = model_path
        self.cfg = cfg
        self.obs_pixels = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)
        self.ground_pixels = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)
        self.obs_views = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)
        self.ground_views = torch.zeros((self.num_triangles,), dtype=torch.float32, device=self.device)

    def _cache_path(self) -> str:
        if os.path.isabs(self.cfg.cache_file):
            return self.cfg.cache_file
        return os.path.join(self.model_path, self.cfg.cache_file)

    def _config_signature(self):
        return (
            int(self.cfg.min_observations),
            float(self.cfg.min_ground_ratio),
            float(self.cfg.min_view_consistency),
            float(self.cfg.per_view_ground_ratio),
            float(self.cfg.boundary_margin),
            float(self.cfg.confidence_min),
        )

    def ensure_num_triangles(self, new_num_triangles: int):
        new_num = int(new_num_triangles)
        if new_num == self.num_triangles:
            return
        if new_num <= 0:
            self.num_triangles = 0
            self.obs_pixels = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.ground_pixels = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.obs_views = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.ground_views = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return

        if new_num < self.num_triangles:
            self.obs_pixels = self.obs_pixels[:new_num]
            self.ground_pixels = self.ground_pixels[:new_num]
            self.obs_views = self.obs_views[:new_num]
            self.ground_views = self.ground_views[:new_num]
        else:
            grow = new_num - self.num_triangles
            self.obs_pixels = torch.cat([self.obs_pixels, torch.zeros((grow,), dtype=torch.float32, device=self.device)], dim=0)
            self.ground_pixels = torch.cat([self.ground_pixels, torch.zeros((grow,), dtype=torch.float32, device=self.device)], dim=0)
            self.obs_views = torch.cat([self.obs_views, torch.zeros((grow,), dtype=torch.float32, device=self.device)], dim=0)
            self.ground_views = torch.cat([self.ground_views, torch.zeros((grow,), dtype=torch.float32, device=self.device)], dim=0)
        self.num_triangles = new_num

    def load_cache(self):
        if not self.cfg.use_cache:
            return
        path = self._cache_path()
        if not os.path.exists(path):
            return
        try:
            payload = torch.load(path, map_location=self.device)
            if int(payload.get("num_triangles", -1)) != self.num_triangles:
                print("[GroundAssoc] Cache triangle count mismatch; ignoring cache.")
                return
            if tuple(payload.get("config_signature", ())) != self._config_signature():
                print("[GroundAssoc] Cache config mismatch; ignoring cache.")
                return
            self.obs_pixels = payload["obs_pixels"].to(self.device).to(torch.float32)
            self.ground_pixels = payload["ground_pixels"].to(self.device).to(torch.float32)
            self.obs_views = payload["obs_views"].to(self.device).to(torch.float32)
            self.ground_views = payload["ground_views"].to(self.device).to(torch.float32)
            print(f"[GroundAssoc] Loaded cache: {path}")
        except Exception as exc:
            print(f"[GroundAssoc] Failed to load cache '{path}': {exc}")

    def save_cache(self):
        if not self.cfg.use_cache:
            return
        path = self._cache_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "num_triangles": self.num_triangles,
            "config_signature": self._config_signature(),
            "obs_pixels": self.obs_pixels.detach().cpu(),
            "ground_pixels": self.ground_pixels.detach().cpu(),
            "obs_views": self.obs_views.detach().cpu(),
            "ground_views": self.ground_views.detach().cpu(),
        }
        torch.save(payload, path)

    def update_from_render(self, render_pkg: Dict, viewpoint_cam):
        if getattr(viewpoint_cam, "ground_mask", None) is None:
            return
        ids = render_pkg["rend_ids"].squeeze(0).long()
        h, w = int(ids.shape[0]), int(ids.shape[1])
        mask = _resize_mask(viewpoint_cam.ground_mask, out_h=h, out_w=w).to(self.device)
        valid = (ids >= 0) & (ids < self.num_triangles)
        if not torch.any(valid):
            return

        ids_valid = ids[valid]
        uniq_ids, inv = torch.unique(ids_valid, return_inverse=True)
        if uniq_ids.numel() == 0:
            return

        total_local = torch.bincount(inv, minlength=uniq_ids.numel()).to(torch.float32)
        ground_local = torch.zeros((uniq_ids.numel(),), dtype=torch.float32, device=self.device)

        ids_ground = ids[valid & mask]
        if ids_ground.numel() > 0:
            ground_ids, ground_counts = torch.unique(ids_ground, return_counts=True)
            pos = torch.searchsorted(uniq_ids, ground_ids)
            in_range = pos < uniq_ids.numel()
            if torch.any(in_range):
                pos = pos[in_range]
                g_ids = ground_ids[in_range]
                g_cnt = ground_counts[in_range].to(torch.float32)
                matched = uniq_ids[pos] == g_ids
                if torch.any(matched):
                    ground_local[pos[matched]] = g_cnt[matched]

        per_view_ratio = ground_local / torch.clamp(total_local, min=1.0)
        ground_in_view = per_view_ratio >= float(self.cfg.per_view_ground_ratio)

        self.obs_pixels.index_add_(0, uniq_ids, total_local)
        self.ground_pixels.index_add_(0, uniq_ids, ground_local)
        self.obs_views[uniq_ids] += 1.0
        if torch.any(ground_in_view):
            self.ground_views[uniq_ids[ground_in_view]] += 1.0

    def get_statistics(self) -> Dict:
        ratio = self.ground_pixels / torch.clamp(self.obs_pixels, min=1.0)
        consistency = self.ground_views / torch.clamp(self.obs_views, min=1.0)
        obs = self.obs_views
        obs_scale = 1.0 - torch.exp(-obs / max(float(self.cfg.min_observations), 1.0))
        confidence = 0.5 * (ratio + consistency) * obs_scale

        reliable_obs = obs >= float(self.cfg.min_observations)
        ratio_thr = float(max(0.0, min(1.0, self.cfg.min_ground_ratio)))
        margin = float(max(0.0, min(0.49, self.cfg.boundary_margin)))
        consistency_thr = float(max(0.0, min(1.0, self.cfg.min_view_consistency)))
        confidence_thr = float(max(0.0, min(1.0, self.cfg.confidence_min)))
        ratio_low = ratio >= (ratio_thr - margin)
        ratio_high = ratio >= (ratio_thr + margin)
        consistent = consistency >= consistency_thr
        confident = confidence >= confidence_thr

        boundary_uncertain = reliable_obs & ratio_low & (~ratio_high)
        is_ground = reliable_obs & ratio_high & consistent & confident
        candidate = reliable_obs & ratio_low & consistent

        return {
            "observations": obs.clone(),
            "ground_pixel_hits": self.ground_pixels.clone(),
            "ground_support_ratio": ratio.clone(),
            "view_consistency": consistency.clone(),
            "confidence": confidence.clone(),
            "reliable_observation_mask": reliable_obs,
            "candidate_mask": candidate,
            "boundary_uncertain_mask": boundary_uncertain,
            "is_ground_mask": is_ground,
        }

    def maybe_save_debug(self, iteration: int):
        if int(self.cfg.debug_every) <= 0:
            return
        if int(iteration) % int(self.cfg.debug_every) != 0:
            return
        out_dir = self.cfg.debug_dir if self.cfg.debug_dir else os.path.join(self.model_path, "ground_assoc_debug")
        os.makedirs(out_dir, exist_ok=True)
        stats = self.get_statistics()
        ratio = stats["ground_support_ratio"].detach().cpu().numpy()
        is_ground = stats["is_ground_mask"].detach().cpu()
        boundary_uncertain = stats["boundary_uncertain_mask"].detach().cpu()
        summary_path = os.path.join(out_dir, f"summary_iter_{int(iteration):06d}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"iteration={int(iteration)}\n")
            f.write(f"triangles_total={self.num_triangles}\n")
            f.write(f"triangles_ground={int(is_ground.sum().item())}\n")
            f.write(f"triangles_boundary_uncertain={int(boundary_uncertain.sum().item())}\n")
            reliable = stats["reliable_observation_mask"]
            f.write(f"triangles_reliable_obs={int(reliable.sum().item())}\n")
            f.write(f"triangles_filtered_unreliable={int((~reliable).sum().item())}\n")
            f.write(f"ratio_mean={float(ratio.mean()):.6f}\n")
            f.write(f"ratio_q50={float(float(torch.tensor(ratio).quantile(0.5))):.6f}\n")
            f.write(f"ratio_q90={float(float(torch.tensor(ratio).quantile(0.9))):.6f}\n")
        torch.save(
            {
                "is_ground_ids": torch.nonzero(is_ground, as_tuple=True)[0],
                "boundary_uncertain_ids": torch.nonzero(boundary_uncertain, as_tuple=True)[0],
            },
            os.path.join(out_dir, f"classified_ids_iter_{int(iteration):06d}.pt"),
        )
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7, 4))
            plt.hist(ratio, bins=max(int(self.cfg.hist_bins), 20))
            plt.xlabel("ground support ratio")
            plt.ylabel("triangle count")
            plt.title(f"Ground support ratio histogram @iter {int(iteration)}")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"ratio_hist_iter_{int(iteration):06d}.png"), dpi=180)
            plt.close(fig)
        except Exception as exc:
            print(f"[GroundAssoc] Failed to save histogram: {exc}")
