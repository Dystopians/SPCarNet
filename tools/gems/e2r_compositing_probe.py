"""GEMS Stage-1R R2 (E2R) — MANDATORY compositing-order sanity probe.

The tile compositor sorts fragments by TRIANGLE-CENTER depth and applies no
backface culling. Before any E2R run with genuinely semi-transparent
triangles, this probe must establish (on a small model, train views ONLY —
D4-pure) whether that compositing scheme is sane under transparency:

  P1  step-0 identity: releasing the opacity floor 0.999 -> o_min=0.01 with
      the update_min_weight re-expression leaves renders unchanged;
  P2  checkerboard 0.3/1.0 opacities vs all-(~)1.0: paired SSIM under a
      small camera perturbation (ordering instability under transparency
      shows up as extra popping vs the opaque control) + seam panels;
  P3  a triangle at o=0.01 is visually negligible (render with a subset at
      realized opacity 0.01 vs the SAME subset removed via the temporary
      active mask -> difference must be tiny; vs subset opaque -> large);
  P4  checkpoint round-trip: the persisted opacity_floor key reloads to
      identical realized opacities (eval-path correctness for E2R models);
  P5  lambda_opacity_decay scale sanity: photometric loss magnitude on train
      views vs lambda_o * L_o (pre-registered lambda_o = 1e-4, GOAL #R-00;
      exponent may be adjusted BEFORE the real runs iff >10x mismatch).

Usage:
    python -m tools.gems.e2r_compositing_probe \
        --checkpoint <point_cloud_state_dict.pt> --scene toy_parking \
        --out /data/peilincai/gems_stage1/analysis/e2r_probe [--gpu 4]

Read-only w.r.t. the source checkpoint; writes JSON + PNG panels to --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="E2R compositing-order probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--o-min", type=float, default=0.01)
    p.add_argument("--cb-opacity", type=float, default=0.3)
    p.add_argument("--cb-cell", type=float, default=1.0,
                   help="checkerboard cell size in scene units (toy: meters)")
    p.add_argument("--perturb-rot-deg", type=float, default=0.2)
    p.add_argument("--perturb-trans", type=float, default=0.01)
    p.add_argument("--lambda-opacity-decay", type=float, default=1e-4)
    p.add_argument("--n-photometric-views", type=int, default=10)
    return p.parse_args()


def _to_u8(img):
    import numpy as np
    return (img.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def _save_png(img_chw, path):
    from PIL import Image
    Image.fromarray(_to_u8(img_chw)).save(path)


def _save_diff_heat(a, b, path, gain=10.0):
    """|a-b| mean over channels, amplified, saved as grayscale heat."""
    import numpy as np
    from PIL import Image
    d = (a - b).abs().mean(dim=0)  # [H,W]
    arr = (d.detach().cpu().numpy() * 255.0 * gain).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _perturbed_cam(cam, rot_deg, trans_dist):
    """Small SE(3) perturbation of a camera; returns a MiniCam."""
    import torch
    from scene.cameras import MiniCam
    th = math.radians(rot_deg)
    # rotation about the camera-frame y axis + small camera-frame x translation
    R_delta = torch.tensor(
        [[math.cos(th), 0.0, math.sin(th), 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-math.sin(th), 0.0, math.cos(th), 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda")
    T_delta = torch.eye(4, dtype=torch.float32, device="cuda")
    T_delta[0, 3] = trans_dist
    # world_view_transform is stored TRANSPOSED (row-vector convention):
    # p_cam = p_world @ W. Compose the perturbation on the camera side.
    W = cam.world_view_transform  # [4,4], transposed W2C
    W2C = W.transpose(0, 1)
    W2C_new = T_delta @ R_delta @ W2C
    W_new = W2C_new.transpose(0, 1).contiguous()
    full_new = (W_new.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    return MiniCam(cam.image_width, cam.image_height, cam.FoVy, cam.FoVx,
                   cam.znear, cam.zfar, W_new, full_new)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from tools.gems.scenes import SCENES
    from tools.gems.triangle_evidence import _TrainOnlyContext
    from utils.loss_utils import l1_loss, ssim

    spec = SCENES[args.scene]
    ctx = _TrainOnlyContext(args.checkpoint, spec)
    tri = ctx.triangles
    report = {"checkpoint": os.path.abspath(args.checkpoint), "scene": args.scene,
              "o_min": args.o_min, "cb_opacity": args.cb_opacity,
              "cb_cell": args.cb_cell,
              "perturb": {"rot_deg": args.perturb_rot_deg, "trans": args.perturb_trans}}

    cams = ctx.train_cams
    view_ids = [0, len(cams) // 2, len(cams) - 1]
    views = [cams[i] for i in view_ids]
    report["views"] = [c.image_name for c in views]
    print(f"[probe] {len(cams)} train cams; probe views: {report['views']}")

    def render_views(cam_list):
        return [ctx.render_view(c)["render"].detach().clamp(0, 1) for c in cam_list]

    # ---------------- P1: floor release identity ----------------
    base = render_views(views)  # floor 0.999 as loaded
    pre_realized = tri.get_vertex_weight.detach().clone()
    logits_backup = tri.vertex_weight.detach().clone()
    tri.update_min_weight(args.o_min, preserve_outputs=True)
    post_realized = tri.get_vertex_weight.detach()
    released = render_views(views)
    p1 = {
        "opacity_floor_after": float(tri.opacity_floor),
        "max_abs_delta_realized_opacity": float((post_realized - pre_realized).abs().max().item()),
        "per_view_max_abs_pixel_delta": [float((r - b).abs().max().item())
                                         for r, b in zip(released, base)],
        "per_view_mean_abs_pixel_delta": [float((r - b).abs().mean().item())
                                          for r, b in zip(released, base)],
        "realized_opacity_min_after_release": float(post_realized.min().item()),
        "realized_opacity_mean_after_release": float(post_realized.mean().item()),
    }
    report["P1_floor_release_identity"] = p1
    print("[probe] P1 floor release:", json.dumps(p1, indent=1))

    # ---------------- checkerboard construction ----------------
    with torch.no_grad():
        V = tri.vertices.detach()
        F = tri._triangle_indices.long()
        cent = V[F].mean(dim=1)  # [T,3]
        cell = float(args.cb_cell)
        lab = ((torch.floor(cent[:, 0] / cell) + torch.floor(cent[:, 1] / cell)
                + torch.floor(cent[:, 2] / cell)).long() % 2) == 0  # [T] bool
        # vertices touched by any labeled triangle -> semi-transparent
        v_sel = torch.zeros(V.shape[0], dtype=torch.bool, device=V.device)
        v_sel[F[lab].reshape(-1)] = True
        tri_sel_count = torch.zeros(F.shape[0], dtype=torch.int32, device=V.device)
        tri_sel_count = v_sel[F].sum(dim=1).to(torch.int32)
    report["checkerboard"] = {
        "labeled_triangles": int(lab.sum().item()),
        "total_triangles": int(F.shape[0]),
        "triangles_fully_semitransparent": int((tri_sel_count == 3).sum().item()),
        "triangles_mixed_1or2_verts": int(((tri_sel_count > 0) & (tri_sel_count < 3)).sum().item()),
        "triangles_untouched": int((tri_sel_count == 0).sum().item()),
        "note": "vertices are SHARED between triangles; per-triangle opacity is "
                "approximated by setting all vertices of labeled triangles to "
                "cb_opacity (boundary triangles blend, honest construction)",
    }

    def set_realized(mask_v, value):
        """Set realized opacity of masked vertices to `value`, rest to ~as-loaded."""
        with torch.no_grad():
            tri.vertex_weight.data.copy_(logits_backup)  # floor-0.999 logits
            # re-express all logits under the released floor (identical realized)
            tri.opacity_floor = 0.999
            tri.update_min_weight(args.o_min, preserve_outputs=True)
            y = tri.get_vertex_weight.detach().clone()
            y[mask_v] = value
            y = y.clamp(args.o_min + tri.eps, 1.0 - tri.eps)
            tri.vertex_weight.data.copy_(tri.inverse_opacity_activation(y))

    # ---------------- P2a: renderer determinism + epsilon-order stability ----
    # A sort-order pathology signature that is independent of image parallax:
    # an epsilon camera translation (1e-4 units) moves image content by a
    # sub-pixel amount but perturbs every triangle-center sort key, flipping
    # near-ties. Order-sensitive semi-transparent compositing then produces
    # discrete pixel jumps (MAD_cb >> MAD_opaque).
    p2a = {"per_view": []}
    for vi, cam in zip(view_ids, views):
        ecam = _perturbed_cam(cam, 0.0, 1e-4)
        set_realized(torch.zeros_like(v_sel), 1.0)
        a0 = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        a0b = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        ae = ctx.render_view(ecam)["render"].detach().clamp(0, 1)
        set_realized(v_sel, args.cb_opacity)
        b0 = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        b0b = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        be = ctx.render_view(ecam)["render"].detach().clamp(0, 1)
        row = {
            "view": cam.image_name,
            "determinism_mad_opaque": float((a0 - a0b).abs().max().item()),
            "determinism_mad_cb": float((b0 - b0b).abs().max().item()),
            "eps_mad_opaque": float((a0 - ae).abs().mean().item()),
            "eps_max_opaque": float((a0 - ae).abs().max().item()),
            "eps_mad_cb": float((b0 - be).abs().mean().item()),
            "eps_max_cb": float((b0 - be).abs().max().item()),
        }
        p2a["per_view"].append(row)
        print(f"[probe] P2a {cam.image_name}:", json.dumps(row))
    report["P2a_determinism_and_eps_order_stability"] = p2a

    # ---------------- P2: perturbation SSIM, opaque vs checkerboard ----------------
    p2 = {"per_view": []}
    for vi, cam in zip(view_ids, views):
        pcam = _perturbed_cam(cam, args.perturb_rot_deg, args.perturb_trans)
        # (a) all ~1.0 (as-loaded, re-expressed)
        set_realized(torch.zeros_like(v_sel), 1.0)  # no-op mask -> as loaded
        a0 = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        a1 = ctx.render_view(pcam)["render"].detach().clamp(0, 1)
        # (b) checkerboard
        set_realized(v_sel, args.cb_opacity)
        b0 = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        b1 = ctx.render_view(pcam)["render"].detach().clamp(0, 1)
        ssim_a = float(ssim(a0, a1).item())
        ssim_b = float(ssim(b0, b1).item())
        row = {
            "view": cam.image_name,
            "ssim_opaque_perturb": ssim_a,
            "ssim_checkerboard_perturb": ssim_b,
            "delta_ssim_opaque_minus_cb": ssim_a - ssim_b,
            "mad_opaque_perturb": float((a0 - a1).abs().mean().item()),
            "mad_cb_perturb": float((b0 - b1).abs().mean().item()),
            "mad_cb_vs_opaque_same_pose": float((b0 - a0).abs().mean().item()),
        }
        p2["per_view"].append(row)
        tag = f"v{vi:03d}"
        _save_png(a0, os.path.join(args.out, f"{tag}_a_opaque.png"))
        _save_png(b0, os.path.join(args.out, f"{tag}_b_checkerboard.png"))
        _save_png(b1, os.path.join(args.out, f"{tag}_b_checkerboard_perturbed.png"))
        _save_diff_heat(b0, a0, os.path.join(args.out, f"{tag}_diff_cb_vs_opaque_x10.png"))
        _save_diff_heat(b0, b1, os.path.join(args.out, f"{tag}_diff_cb_perturb_x10.png"))
        _save_diff_heat(a0, a1, os.path.join(args.out, f"{tag}_diff_opaque_perturb_x10.png"))
        print(f"[probe] P2 {cam.image_name}:", json.dumps(row))
    report["P2_perturbation_ssim"] = p2

    # ---------------- P3: o=0.01 negligibility (paired construction) ----------
    # Vertices are shared between triangles, so "set a triangle to 0.01" leaks
    # into its neighbors. Paired isolation: inner_verts = vertices referenced
    # ONLY by labeled triangles; S_inner = labeled triangles whose 3 vertices
    # are all inner. Both renders set inner_verts to o_min; they differ ONLY
    # by S_inner being present (at fully-o_min alpha) vs masked out. The diff
    # therefore isolates exactly "a triangle at o=o_min vs absent". The
    # opaque-reference render keeps everything as loaded but masks S_inner
    # out, showing S_inner's actual visible footprint.
    with torch.no_grad():
        vert_touch_unlabeled = torch.zeros(V.shape[0], dtype=torch.bool, device=V.device)
        vert_touch_unlabeled[F[~lab].reshape(-1)] = True
        inner_verts = v_sel & (~vert_touch_unlabeled)
        s_inner = inner_verts[F].all(dim=1)  # [T] bool, subset of lab
    p3 = {
        "inner_vertices": int(inner_verts.sum().item()),
        "s_inner_triangles": int(s_inner.sum().item()),
        "per_view": [],
    }
    for vi, cam in zip(view_ids, views):
        # (A) inner verts at o_min, S_inner present
        set_realized(inner_verts, args.o_min)
        r_low = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        # (B) same opacities, S_inner masked out
        tri.set_temporary_active_mask(~s_inner)
        r_removed = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        tri.clear_temporary_active_mask()
        # (C) reference: as-loaded opacities, S_inner masked out (footprint)
        set_realized(torch.zeros_like(v_sel), 1.0)
        tri.set_temporary_active_mask(~s_inner)
        r_ref_removed = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        tri.clear_temporary_active_mask()
        r_ref_full = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        row = {
            "view": cam.image_name,
            "mad_o001_vs_removed_paired": float((r_low - r_removed).abs().mean().item()),
            "max_o001_vs_removed_paired": float((r_low - r_removed).abs().max().item()),
            "p99_o001_vs_removed_paired": float(
                torch.quantile((r_low - r_removed).abs().reshape(-1).float(), 0.99).item()),
            "mad_opaque_footprint": float((r_ref_full - r_ref_removed).abs().mean().item()),
            "max_opaque_footprint": float((r_ref_full - r_ref_removed).abs().max().item()),
        }
        p3["per_view"].append(row)
        tag = f"v{vi:03d}"
        _save_diff_heat(r_low, r_removed, os.path.join(args.out, f"{tag}_diff_o001_vs_removed_x10.png"))
        print(f"[probe] P3 {cam.image_name}:", json.dumps(row))
    report["P3_omin_negligibility"] = p3

    # ---------------- P3b: SINGLE triangle at o=0.01 (the literal check) -----
    # P3's aggregate can accumulate many stacked low-alpha triangles (correct
    # alpha math: N layers of 0.01 -> 1-0.99^N). The spec's unit check is one
    # triangle: take the largest-screen-area S_inner triangles individually,
    # set that triangle's vertices to o_min, and compare present vs masked.
    with torch.no_grad():
        areas = tri.triangle_areas().detach()
        areas_inner = torch.where(s_inner, areas, torch.zeros_like(areas))
        k = min(5, int(s_inner.sum().item()))
        top_ids = torch.topk(areas_inner, k).indices
    p3b = {"per_triangle": []}
    cam = views[0]
    for t_id in top_ids.tolist():
        t_verts = torch.zeros(V.shape[0], dtype=torch.bool, device=V.device)
        t_verts[F[t_id]] = True
        set_realized(t_verts, args.o_min)
        r_one = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        mask_one = torch.ones(F.shape[0], dtype=torch.bool, device=V.device)
        mask_one[t_id] = False
        tri.set_temporary_active_mask(mask_one)
        r_one_rm = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        tri.clear_temporary_active_mask()
        # opaque footprint of the same triangle
        set_realized(torch.zeros_like(v_sel), 1.0)
        tri.set_temporary_active_mask(mask_one)
        r_op_rm = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        tri.clear_temporary_active_mask()
        r_op = ctx.render_view(cam)["render"].detach().clamp(0, 1)
        p3b["per_triangle"].append({
            "triangle_id": int(t_id),
            "area": float(areas[t_id].item()),
            "max_o001_vs_removed": float((r_one - r_one_rm).abs().max().item()),
            "mad_o001_vs_removed": float((r_one - r_one_rm).abs().mean().item()),
            "max_opaque_footprint": float((r_op - r_op_rm).abs().max().item()),
        })
        print(f"[probe] P3b tri {t_id}:", json.dumps(p3b["per_triangle"][-1]))
    report["P3b_single_triangle_omin"] = p3b

    # ---------------- P4: checkpoint round-trip of opacity_floor ----------------
    set_realized(v_sel, args.cb_opacity)  # save a genuinely semi-transparent state
    rt_dir = os.path.join(args.out, "roundtrip_ckpt")
    tri.save_parameters(rt_dir)
    from scene.triangle_model import TriangleModel
    tri2 = TriangleModel(3)
    tri2.load_parameters(rt_dir, device="cuda")
    p4 = {
        "saved_floor": float(tri.opacity_floor),
        "loaded_floor": float(tri2.opacity_floor),
        "max_abs_delta_realized": float(
            (tri2.get_vertex_weight.detach() - tri.get_vertex_weight.detach()).abs().max().item()),
    }
    report["P4_checkpoint_roundtrip"] = p4
    print("[probe] P4 roundtrip:", json.dumps(p4))
    del tri2
    torch.cuda.empty_cache()
    # D6 hygiene: the round-trip ckpt is a GB-scale scratch artifact.
    rt_pt = os.path.join(rt_dir, "point_cloud_state_dict.pt")
    if os.path.isfile(rt_pt):
        os.remove(rt_pt)

    # ---------------- P5: lambda_o scale sanity ----------------
    set_realized(torch.zeros_like(v_sel), 1.0)  # back to as-loaded realized
    lam = float(args.lambda_opacity_decay)
    photo, n = [], min(args.n_photometric_views, len(cams))
    step = max(1, len(cams) // n)
    for c in cams[::step][:n]:
        pkg = ctx.render_view(c)
        img = pkg["render"].clamp(0, 1)
        gt = c.original_image.cuda()
        photo.append(float(((1.0 - 0.2) * l1_loss(img, gt)
                            + 0.2 * (1.0 - ssim(img, gt))).item()))
    L_o = float(tri.get_vertex_weight[tri._triangle_indices.long()].mean().item())
    photo_mean = sum(photo) / len(photo)
    p5 = {
        "photometric_loss_mean": photo_mean,
        "photometric_views_used": len(photo),
        "L_o_mean_realized_opacity": L_o,
        "lambda_o": lam,
        "weighted_L_o": lam * L_o,
        "ratio_weighted_Lo_over_photometric": (lam * L_o) / max(photo_mean, 1e-12),
    }
    report["P5_lambda_scale_sanity"] = p5
    print("[probe] P5 scale sanity:", json.dumps(p5, indent=1))

    out_json = os.path.join(args.out, "probe_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[probe] DONE -> {out_json}")


if __name__ == "__main__":
    main()
