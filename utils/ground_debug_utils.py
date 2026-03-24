import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image


def _ids_to_color(ids_hw: np.ndarray) -> np.ndarray:
    h, w = ids_hw.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = ids_hw >= 0
    if not np.any(valid):
        return color
    ids = ids_hw[valid].astype(np.int64)
    r = ((ids * 37) % 255).astype(np.uint8)
    g = ((ids * 57) % 255).astype(np.uint8)
    b = ((ids * 97) % 255).astype(np.uint8)
    color_valid = np.stack([r, g, b], axis=1)
    color[valid] = color_valid
    return color


def save_ground_debug_view(
    out_dir: str,
    iteration: int,
    image_name: str,
    gt_image_chw: torch.Tensor,
    ground_mask_hw: Optional[torch.Tensor],
    rend_ids_hw: Optional[torch.Tensor],
    association_stats: Optional[Dict],
):
    os.makedirs(out_dir, exist_ok=True)
    rgb = gt_image_chw.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    h, w = rgb_u8.shape[:2]

    if ground_mask_hw is None:
        mask = np.zeros((h, w), dtype=bool)
    else:
        m = ground_mask_hw.detach().cpu().numpy().astype(np.float32)
        if m.shape != (h, w):
            # nearest resize via PIL
            m_img = Image.fromarray((m > 0.5).astype(np.uint8) * 255)
            m_img = m_img.resize((w, h), resample=Image.NEAREST)
            m = np.array(m_img).astype(np.float32) / 255.0
        mask = m > 0.5

    mask_vis = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)

    if rend_ids_hw is None:
        ids = -np.ones((h, w), dtype=np.int64)
    else:
        ids = rend_ids_hw.detach().cpu().numpy().astype(np.int64)
        if ids.shape != (h, w):
            ids = np.resize(ids, (h, w))
    ids_color = _ids_to_color(ids)

    overlay_ground = rgb_u8.copy()
    overlay_ground[mask] = (0.45 * overlay_ground[mask] + 0.55 * np.array([40, 220, 40], dtype=np.float32)).astype(np.uint8)

    overlay_reg = rgb_u8.copy()
    if association_stats is not None and rend_ids_hw is not None:
        is_ground = association_stats["is_ground_mask"].detach().cpu().numpy()
        boundary = association_stats["boundary_uncertain_mask"].detach().cpu().numpy()
        valid = ids >= 0
        ground_px = valid & (ids < is_ground.shape[0]) & is_ground[np.clip(ids, 0, is_ground.shape[0] - 1)]
        boundary_px = valid & (ids < boundary.shape[0]) & boundary[np.clip(ids, 0, boundary.shape[0] - 1)]
        overlay_reg[ground_px] = (0.4 * overlay_reg[ground_px] + 0.6 * np.array([30, 220, 30], dtype=np.float32)).astype(np.uint8)
        overlay_reg[boundary_px] = (0.4 * overlay_reg[boundary_px] + 0.6 * np.array([220, 180, 30], dtype=np.float32)).astype(np.uint8)

    sheet = np.concatenate([rgb_u8, mask_vis, ids_color, overlay_ground, overlay_reg], axis=1)
    path = os.path.join(out_dir, f"iter_{int(iteration):06d}_{image_name}_ground_debug.png")
    Image.fromarray(sheet).save(path)
