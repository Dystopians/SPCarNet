import json
import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from triangle_renderer import render
from utils.loss_utils import l1_loss
from utils.prism_scoring import (
    PrismScoreConfig,
    PrismScoreInputs,
    compute_prism_scores,
    summarize_prism_scores,
)
from utils.triangle_sparse_support import SparseSupportConfig, TriangleSparseSupportEstimator
from utils.triangle_stats import TriangleStatsManager
from utils.triangle_structure_utils import compute_triangle_structure_metrics


def _build_score_cfg(opt) -> PrismScoreConfig:
    return PrismScoreConfig(
        utility_w_vis=float(opt.prism_utility_w_vis),
        utility_w_sens=float(opt.prism_utility_w_sens),
        utility_w_geo=float(opt.prism_utility_w_geo),
        utility_w_viewdiv=float(opt.prism_utility_w_viewdiv),
        utility_w_edge=float(opt.prism_utility_w_edge),
        redund_w_flat=float(opt.prism_redund_w_flat),
        redund_w_coplanar=float(opt.prism_redund_w_coplanar),
        norm_percentile_low=float(opt.prism_norm_percentile_low),
        norm_percentile_high=float(opt.prism_norm_percentile_high),
        norm_eps=float(opt.prism_norm_eps),
        thresh_protected_edge=float(opt.prism_thresh_protected_edge),
        thresh_protected_geo=float(opt.prism_thresh_protected_geo),
        thresh_protected_sens=float(opt.prism_thresh_protected_sens),
        thresh_protected_unc=float(opt.prism_thresh_protected_unc),
        thresh_dead_vis=float(opt.prism_thresh_dead_vis),
        thresh_dead_sens=float(opt.prism_thresh_dead_sens),
        thresh_dead_geo=float(opt.prism_thresh_dead_geo),
        thresh_dead_edge=float(opt.prism_thresh_dead_edge),
        thresh_suspicious_vis=float(opt.prism_thresh_suspicious_vis),
        thresh_suspicious_geo=float(opt.prism_thresh_suspicious_geo),
        thresh_suspicious_unc=float(opt.prism_thresh_suspicious_unc),
        boundary_risk_value=float(opt.prism_boundary_risk_value),
        nonmanifold_risk_value=float(opt.prism_nonmanifold_risk_value),
        recent_age_iters=int(opt.prism_recent_age_iters),
        ground_protect_bonus=float(opt.prism_ground_protect_bonus),
        roi_protect_bonus=float(opt.prism_roi_protect_bonus),
        use_ground_protect=bool(opt.prism_use_ground_protect),
        use_roi_protect=bool(opt.prism_use_roi_protect),
    )


def _topk_ids(mask: torch.Tensor, score: torch.Tensor, k: int) -> torch.Tensor:
    ids = torch.nonzero(mask, as_tuple=True)[0]
    if ids.numel() == 0:
        return ids
    s = score[ids]
    kk = min(int(k), int(ids.numel()))
    _, order = torch.topk(s, k=kk, largest=True, sorted=True)
    return ids[order]


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect PRISM scores and triangle classifier without pruning")
    mp = ModelParams(parser, sentinel=False)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Use -1 for latest")
    parser.add_argument("--collect_views", type=int, default=24, help="How many train views to collect stats from")
    parser.add_argument("--max_print", type=int, default=12, help="Top-K entries per class")
    parser.add_argument("--output_json", type=str, default="", help="Optional output json path")
    args = get_combined_args(parser)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=opt.set_weight,
        set_sigma=opt.set_sigma,
        load_iteration=args.iteration,
        shuffle=False,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    stats = TriangleStatsManager(
        num_triangles=int(triangles._triangle_indices.shape[0]),
        device=triangles.vertices.device,
        init_iter=int(scene.loaded_iter or 0),
        ema_decay=0.95,
        view_hist_bins=8,
    )

    views = scene.getTrainCameras()
    num_collect = min(int(args.collect_views), len(views))
    if num_collect <= 0:
        raise RuntimeError("No train cameras available for PRISM score inspection.")

    for i in range(num_collect):
        cam = views[i]
        pkg = render(cam, triangles, pipe, background)
        stats.update_visibility_from_render(render_pkg=pkg, triangles=triangles, viewpoint_cam=cam)
        gt = cam.original_image.cuda()
        loss = l1_loss(pkg["render"], gt)
        loss.backward()
        stats.update_gradient_stats(triangles=triangles)
        if triangles.vertices.grad is not None:
            triangles.vertices.grad.zero_()
        if triangles._features_dc.grad is not None:
            triangles._features_dc.grad.zero_()
        if triangles._features_rest.grad is not None:
            triangles._features_rest.grad.zero_()
        if triangles.vertex_weight.grad is not None:
            triangles.vertex_weight.grad.zero_()

    struct_metrics, _ = compute_triangle_structure_metrics(
        vertices=triangles.vertices,
        triangle_indices=triangles._triangle_indices,
        cache=None,
    )
    sparse_cfg = SparseSupportConfig(max_point_error=float(opt.ground_plane_colmap_error_max))
    sparse_est = TriangleSparseSupportEstimator.from_scene(scene=scene, cfg=sparse_cfg)
    sparse_metrics = sparse_est.compute(vertices=triangles.vertices, triangle_indices=triangles._triangle_indices)

    score_cfg = _build_score_cfg(opt)
    score_inputs = PrismScoreInputs(
        vis_count_ema=stats.stats.vis_count_ema,
        grad_pos_norm_ema=stats.stats.grad_pos_norm_ema,
        grad_app_norm_ema=stats.stats.grad_app_norm_ema,
        grad_norm_var_ema=stats.stats.grad_norm_var_ema,
        view_direction_histogram=stats.stats.view_direction_histogram,
        birth_iter=stats.stats.birth_iter,
        geometry_support_score_base=sparse_metrics.geometry_support_score_base,
        boundary_edge_count=struct_metrics.boundary_edge_count,
        nonmanifold_edge_count=struct_metrics.nonmanifold_edge_count,
        flatness_score=struct_metrics.flatness_score,
        coplanar_neighbor_fraction=struct_metrics.coplanar_neighbor_fraction,
        ground_protect_t=None,
        roi_protect_t=None,
    )
    scores = compute_prism_scores(
        inputs=score_inputs,
        current_iter=int(scene.loaded_iter or 0),
        cfg=score_cfg,
    )
    summary = summarize_prism_scores(scores)

    k = int(args.max_print)
    top_protected = _topk_ids(scores.protected_mask, scores.utility_t, k=k)
    top_dead = _topk_ids(scores.dead_mask, 1.0 - scores.utility_t, k=k)
    top_suspicious = _topk_ids(scores.suspicious_mask, scores.risk_t, k=k)
    top_candidates = _topk_ids(scores.candidate_mask, scores.prune_score_t, k=k)

    print("[PRISM] summary:", summary)
    print("[PRISM] top protected:", top_protected.tolist())
    print("[PRISM] top dead:", top_dead.tolist())
    print("[PRISM] top suspicious:", top_suspicious.tolist())
    print("[PRISM] top candidates:", top_candidates.tolist())

    for name, ids in [
        ("protected", top_protected),
        ("dead", top_dead),
        ("suspicious", top_suspicious),
        ("candidate", top_candidates),
    ]:
        print(f"[PRISM] {name} details:")
        for tid in ids.tolist():
            print(
                "  tri={} utility={:.4f} risk={:.4f} redund={:.4f} prune_score={:.4f}".format(
                    int(tid),
                    float(scores.utility_t[tid].item()),
                    float(scores.risk_t[tid].item()),
                    float(scores.redund_t[tid].item()),
                    float(scores.prune_score_t[tid].item()),
                )
            )

    hist = torch.histc(scores.prune_score_t, bins=20, min=0.0, max=1.0).detach().cpu().tolist()
    payload = {
        "summary": summary,
        "top_protected": [int(x) for x in top_protected.tolist()],
        "top_dead": [int(x) for x in top_dead.tolist()],
        "top_suspicious": [int(x) for x in top_suspicious.tolist()],
        "top_candidates": [int(x) for x in top_candidates.tolist()],
        "prune_score_hist_20bins": hist,
    }
    out_path = (
        args.output_json
        if args.output_json
        else os.path.join(dataset.model_path, "prism_debug", f"prism_score_summary_iter_{int(scene.loaded_iter or 0):06d}.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[PRISM] saved summary json: {out_path}")
