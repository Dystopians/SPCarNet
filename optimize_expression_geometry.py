import torch
from argparse import ArgumentParser
import numpy as np

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel


if __name__ == "__main__":
    parser = ArgumentParser(description="Expression-level planar geometry optimization")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to load. -1 loads latest.")
    parser.add_argument("--save_iteration", type=int, default=-1, help="Iteration id to save. -1 overwrites loaded iteration id.")

    parser.add_argument("--preset", type=str, default="parking_lot", choices=["default", "parking_lot"])
    parser.add_argument("--up_axis", type=str, default="auto", choices=["auto", "x", "y", "z"])
    parser.add_argument("--max_ground_tilt_deg", type=float, default=30.0)
    parser.add_argument("--max_neighbor_normal_deg", type=float, default=20.0)
    parser.add_argument("--max_neighbor_height_delta", type=float, default=0.10)
    parser.add_argument("--min_region_triangles", type=int, default=80)
    parser.add_argument("--min_region_area", type=float, default=0.05)
    parser.add_argument("--max_plane_residual", type=float, default=0.03)
    parser.add_argument("--residual_quantile", type=float, default=0.95)
    parser.add_argument("--snap_cell_size", type=float, default=0.05)
    parser.add_argument("--near_field_radius", type=float, default=12.0, help="Use <=0 to disable near-field filtering.")
    parser.add_argument("--enable_global_snap", action="store_true")
    parser.add_argument("--global_height_bin", type=float, default=0.05)
    parser.add_argument("--allow_boundary_snap", action="store_true")
    parser.add_argument("--boundary_snap_max_shift", type=float, default=0.02)
    parser.add_argument("--merge_mode", type=str, default="edge_collapse", choices=["edge_collapse", "snap"])
    parser.add_argument("--project_to_plane", action="store_true")
    parser.add_argument("--max_project_shift", type=float, default=0.015)
    parser.add_argument("--edge_collapse_length", type=float, default=0.03)
    parser.add_argument("--max_collapse_shift", type=float, default=0.02)
    parser.add_argument("--max_edges_per_region", type=int, default=6000)
    parser.add_argument("--max_candidate_triangles", type=int, default=800000)
    parser.add_argument("--quiet", action="store_true")

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

    if args.preset == "parking_lot":
        # Parking lot prior: mostly flat drivable ground, strict local height continuity.
        args.max_ground_tilt_deg = max(args.max_ground_tilt_deg, 25.0)
        args.max_neighbor_normal_deg = max(args.max_neighbor_normal_deg, 15.0)
        args.max_neighbor_height_delta = min(max(args.max_neighbor_height_delta, 0.06), 0.2)
        args.min_region_triangles = min(args.min_region_triangles, 120)
        args.min_region_area = min(args.min_region_area, 0.08)
        args.max_plane_residual = min(max(args.max_plane_residual, 0.02), 0.06)
        args.snap_cell_size = min(max(args.snap_cell_size, 0.03), 0.12)
        args.merge_mode = "edge_collapse"
        args.project_to_plane = True
        args.edge_collapse_length = max(args.edge_collapse_length, 0.02)
        args.max_collapse_shift = max(args.max_collapse_shift, 0.01)
        # quality-first default: do not snap boundaries in parking preset
        args.allow_boundary_snap = False

    near_center = None
    if args.near_field_radius > 0:
        c2ws = []
        for cam in scene.getTrainCameras():
            w2c = np.asarray(cam.world_view_transform.T.cpu().numpy(), dtype=np.float64)
            c2w = np.linalg.inv(w2c)
            c2ws.append(c2w[:3, 3])
        if len(c2ws) > 0:
            near_center = np.mean(np.stack(c2ws, axis=0), axis=0)

    with torch.no_grad():
        stats = scene.triangles.optimize_ground_planar_patches(
            up_axis=args.up_axis,
            max_ground_tilt_deg=args.max_ground_tilt_deg,
            max_neighbor_normal_deg=args.max_neighbor_normal_deg,
            max_neighbor_height_delta=args.max_neighbor_height_delta,
            min_region_triangles=args.min_region_triangles,
            min_region_area=args.min_region_area,
            max_plane_residual=args.max_plane_residual,
            residual_quantile=args.residual_quantile,
            snap_cell_size=args.snap_cell_size,
            near_center=near_center,
            near_radius=args.near_field_radius,
            enable_global_snap=args.enable_global_snap,
            global_height_bin=args.global_height_bin,
            allow_boundary_snap=args.allow_boundary_snap,
            boundary_snap_max_shift=args.boundary_snap_max_shift,
            merge_mode=args.merge_mode,
            project_to_plane=args.project_to_plane,
            max_project_shift=args.max_project_shift,
            edge_collapse_length=args.edge_collapse_length,
            max_collapse_shift=args.max_collapse_shift,
            max_edges_per_region=args.max_edges_per_region,
            max_candidate_triangles=args.max_candidate_triangles,
            verbose=not args.quiet,
        )

    loaded_iter = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    save_iter = loaded_iter if args.save_iteration < 0 else args.save_iteration
    save_path = scene.save(save_iter)

    print("[ExpressionOptimization] stats:", stats)
    print("[ExpressionOptimization] saved:", save_path)
