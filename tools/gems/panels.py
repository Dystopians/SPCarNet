"""GEMS Stage One qualitative panels (PROTOCOL.md 4.5).

For 6 evenly-spaced test views (or all if fewer):
RGB render | GT | x5 error heatmap | median-depth map, one PNG per view,
plus one floater overlay if floater triangle ids are provided by the
geometry module. Small PNGs only.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

_PANEL_DPI = 80


def _even_indices(n_total: int, n_wanted: int):
    if n_total <= n_wanted:
        return list(range(n_total))
    idxs = np.round(np.linspace(0, n_total - 1, n_wanted)).astype(int)
    return sorted(set(int(i) for i in idxs))


def _to_hwc(img_chw: torch.Tensor) -> np.ndarray:
    return img_chw.detach().float().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)


def _save_fig(fig, path: str):
    fig.savefig(path, dpi=_PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


def write_panels(ctx, out_dir: str, n_views: int = 6, floater_tri_ids=None) -> list:
    """Render panels for `n_views` evenly-spaced test views.

    floater_tri_ids: optional 1-D array/tensor of global triangle ids that
    belong to floater components (from geometry g3); enables the overlay.
    Returns the list of written PNG paths.
    """
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    cams = ctx.test_cams
    written = []

    for i in _even_indices(len(cams), n_views):
        cam = cams[i]
        pkg = ctx.render_view(cam)
        render = _to_hwc(pkg["render"])
        gt = _to_hwc(cam.original_image[:3].to(pkg["render"].device))
        err = np.clip(5.0 * np.abs(render - gt).mean(axis=-1), 0.0, 1.0)
        depth = pkg["surf_depth"][0].detach().float().cpu().numpy()
        valid = depth > 0
        vmax = float(np.quantile(depth[valid], 0.99)) if valid.any() else 1.0

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
        axes[0].imshow(render)
        axes[0].set_title("render")
        axes[1].imshow(gt)
        axes[1].set_title("GT")
        axes[2].imshow(err, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title(r"$\times$5 |error|")
        axes[3].imshow(np.where(valid, depth, np.nan), cmap="turbo", vmin=0.0, vmax=vmax)
        axes[3].set_title("median depth")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"{ctx.spec.name} test[{i}] {cam.image_name}", fontsize=10)
        path = os.path.join(panels_dir, f"panel_test{i:03d}_{cam.image_name}.png")
        _save_fig(fig, path)
        written.append(path)

    if floater_tri_ids is not None and len(floater_tri_ids) > 0:
        written.append(_write_floater_overlay(ctx, panels_dir, floater_tri_ids))
    return written


def _write_floater_overlay(ctx, panels_dir: str, floater_tri_ids) -> str:
    """One representative view with floater-component triangles tinted red."""
    cam = ctx.test_cams[len(ctx.test_cams) // 2]
    pkg = ctx.render_view(cam)
    render = _to_hwc(pkg["render"])

    ids_hr = pkg["rend_ids"][0].detach().long()  # supersampled [H*s, W*s]
    fl = torch.as_tensor(np.asarray(floater_tri_ids, dtype=np.int64),
                         device=ids_hr.device)
    mask_hr = torch.isin(ids_hr, fl) & (ids_hr >= 0)
    # rend_ids is uninitialized on pixels no triangle reached (background):
    # gate by the median depth at the same supersampled resolution.
    depth_full = pkg.get("depth_full")
    if depth_full is not None:
        mask_hr &= depth_full[0].detach() > 0
    h, w = render.shape[:2]
    mask = F.interpolate(mask_hr[None, None].float(), size=(h, w),
                         mode="area")[0, 0] > 0
    mask = mask.cpu().numpy()

    overlay = render.copy()
    overlay[mask] = 0.4 * overlay[mask] + 0.6 * np.array([1.0, 0.0, 0.0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title(
        f"{ctx.spec.name} floater overlay ({int(len(floater_tri_ids))} tris) "
        f"{cam.image_name}", fontsize=10,
    )
    path = os.path.join(panels_dir, f"floater_overlay_{cam.image_name}.png")
    _save_fig(fig, path)
    return path
