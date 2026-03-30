from argparse import ArgumentParser

import torch

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from utils.triangle_sparse_support import SparseSupportConfig, TriangleSparseSupportEstimator


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect triangle local sparse COLMAP support")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", type=int, default=-1, help="Use -1 to load latest model iteration.")
    parser.add_argument("--radius", type=float, default=-1.0, help="Explicit support radius; <=0 uses radius_factor * scene_scale.")
    parser.add_argument("--radius_factor", type=float, default=0.02, help="Radius as fraction of scene scale when radius<=0.")
    parser.add_argument("--knn", type=int, default=32, help="kNN fallback when local radius support is insufficient.")
    parser.add_argument("--min_support_points", type=int, default=6)
    parser.add_argument("--pca_min_points", type=int, default=10)
    parser.add_argument("--max_point_error", type=float, default=2.0, help="Filter COLMAP points by reprojection error.")
    parser.add_argument("--max_print", type=int, default=12, help="Print top-K triangles by support score.")
    args = get_combined_args(parser)

    dataset = model.extract(args)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=args.iteration,
        shuffle=False,
    )

    cfg = SparseSupportConfig(
        radius=float(args.radius),
        radius_factor=float(args.radius_factor),
        knn=int(args.knn),
        min_support_points=int(args.min_support_points),
        pca_min_points=int(args.pca_min_points),
        max_point_error=float(args.max_point_error),
    )

    estimator = TriangleSparseSupportEstimator.from_scene(scene=scene, cfg=cfg)
    with torch.no_grad():
        result = estimator.compute(vertices=triangles.vertices, triangle_indices=triangles._triangle_indices)

    t = int(result.support_count.numel())
    print(
        "[TriSparse] triangles={} query_radius={:.6f} scene_scale={:.6f}".format(
            t, float(result.query_radius), float(result.scene_scale)
        )
    )
    if t > 0:
        print(
            "[TriSparse] means: support_count={:.3f} plane_resid_med={:.6f} normal_angle_deg={:.3f} support_score={:.4f}".format(
                float(result.support_count.to(torch.float32).mean().item()),
                float(result.plane_residual_median.mean().item()),
                float(result.normal_angle_residual_deg[result.normal_angle_valid].mean().item())
                if torch.any(result.normal_angle_valid)
                else 0.0,
                float(result.geometry_support_score_base.mean().item()),
            )
        )

        k = min(int(args.max_print), t)
        _, ids = torch.topk(result.geometry_support_score_base, k=k, largest=True, sorted=True)
        print("[TriSparse] top support triangles")
        for tid in ids.tolist():
            print(
                "  tri={} count={} resid_mean={:.6f} resid_med={:.6f} normal_angle_deg={} score={:.4f}".format(
                    int(tid),
                    int(result.support_count[tid].item()),
                    float(result.plane_residual_mean[tid].item()),
                    float(result.plane_residual_median[tid].item()),
                    "{:.3f}".format(float(result.normal_angle_residual_deg[tid].item()))
                    if bool(result.normal_angle_valid[tid].item())
                    else "NA",
                    float(result.geometry_support_score_base[tid].item()),
                )
            )
