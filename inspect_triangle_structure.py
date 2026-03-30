from argparse import ArgumentParser

import torch

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from utils.triangle_structure_utils import (
    compute_triangle_structure_metrics,
    debug_print_triangle_structure,
)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect triangle structural geometry signals")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", type=int, default=-1, help="Use -1 to load latest iteration.")
    parser.add_argument("--max_print", type=int, default=12, help="How many triangles to print.")
    parser.add_argument(
        "--coplanar_angle_threshold_deg",
        type=float,
        default=8.0,
        help="Neighbor angle threshold for coplanar fraction.",
    )
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

    with torch.no_grad():
        metrics, _ = compute_triangle_structure_metrics(
            vertices=triangles.vertices,
            triangle_indices=triangles._triangle_indices,
            cache=None,
            coplanar_angle_threshold_deg=float(args.coplanar_angle_threshold_deg),
        )

    print(
        "[TriStruct] summary: triangles={} boundary_triangles={} nonmanifold_triangles={}".format(
            int(metrics.flatness_score.numel()),
            int(metrics.is_boundary_triangle.sum().item()),
            int(metrics.is_nonmanifold_triangle.sum().item()),
        )
    )
    print(
        "[TriStruct] means: dihedral_deg={:.4f} coplanar_frac={:.4f} qem_like={:.6e} flatness={:.4f}".format(
            float(metrics.mean_abs_dihedral_deg.mean().item()) if metrics.mean_abs_dihedral_deg.numel() > 0 else 0.0,
            float(metrics.coplanar_neighbor_fraction.mean().item()) if metrics.coplanar_neighbor_fraction.numel() > 0 else 0.0,
            float(metrics.qem_like.mean().item()) if metrics.qem_like.numel() > 0 else 0.0,
            float(metrics.flatness_score.mean().item()) if metrics.flatness_score.numel() > 0 else 0.0,
        )
    )

    debug_print_triangle_structure(metrics=metrics, triangle_ids=None, max_print=int(args.max_print))
