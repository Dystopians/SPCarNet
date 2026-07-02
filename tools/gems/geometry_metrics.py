#
# GEMS Stage One — geometry metrics (PROTOCOL.md v1.1.0 §4.3, g1–g4).
#
# This module is part of the single evaluation mouth (run_eval.py). It consumes
# only: the loaded checkpoint (via the EvalContext), camera poses, and the
# declared metric-only GT assets from tools/gems/scenes.py (COLMAP sparse
# model, GT depth maps, GT mesh, laser scans). It must NOT import any
# ELA/teacher/selector code (D4 purity).
#
# The `ctx` argument is duck-typed against the EvalContext contract
# (tools/gems/eval_context.py):
#   .triangles, .pipe, .bg, .train_cams, .test_cams, .spec, .out_dir,
#   .render_view(cam) -> render pkg dict,
#   .vertices() -> [V,3], .faces() -> [T,3] long.
# PROTOCOL 1.1.0 §4.3: the reconstructed surface for g4 = ALL checkpoint
# triangles (the model's opacity floor pins every triangle to >= 0.999
# render-time opacity; see eval_context.EvalContext.opaque_mask).
#

import os

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_next_bytes,
)

# ---------------------------------------------------------------------------
# Frozen protocol constants (PROTOCOL.md §4.3; edits bump MAJOR).
# ---------------------------------------------------------------------------
G1_DEPTH_RATIO = 0.95           # violation iff median depth < 0.95 * d_p
G1_ALPHA_MIN = 0.5              # ... AND rendered alpha >= 0.5
G1_MAX_PAIRS_COLMAP = 100_000   # colmap sparse pairs capped total, seed 0
G1_TOY_PIXELS_PER_VIEW = 20_000 # toy GT-depth pixels subsampled per view
G2_ALPHA_MIN = 0.5
G3_MAX_SUPPORT_VIEWS = 60       # evenly spaced train views for support pass
G3_FLOATER_MAX_SUPPORT = 1      # floater: every member triangle support <= 1
G3_FLOATER_MAX_SIZE = 10_000    # ... AND component size < 10,000 triangles
G4_RECON_SAMPLES = 1_000_000    # area-weighted samples on ALL checkpoint triangles
G4_GT_MESH_SAMPLES = 1_000_000  # toy GT mesh samples
G4_SCAN_SUBSAMPLE = 2_000_000   # courtyard scan subsample
G4_TAU_METERS = 0.05            # F-score threshold (meters)
G4_SCAN_BBOX_MARGIN_METERS = 0.3  # courtyard recon->scan bounding-region rule
SEED = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _units_per_meter(spec):
    upm = getattr(spec, "units_per_meter", None)
    return float(upm) if upm else 1.0


def _out_subdir(ctx):
    out_dir = os.path.join(str(ctx.out_dir), "geometry")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _image_key(name):
    """Match dataset_readers convention: basename without extension."""
    return os.path.basename(str(name)).split(".")[0]


def _spec_gt(spec):
    gt = getattr(spec, "gt", None)
    return gt if isinstance(gt, dict) else {}


# ---------------------------------------------------------------------------
# COLMAP sparse model (metric-only GT asset)
# ---------------------------------------------------------------------------

def _read_points3d_with_ids_binary(path):
    """Like scene.colmap_loader.read_points3D_binary but keeps point3D ids
    (needed to join per-image point2D->point3D tracks). Reuses the repo's
    read_next_bytes for the exact binary layout."""
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        ids = np.empty(num_points, dtype=np.int64)
        xyzs = np.empty((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            props = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            ids[i] = props[0]
            xyzs[i] = props[1:4]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(fid, num_bytes=8 * track_length,
                            format_char_sequence="ii" * track_length)
    return ids, xyzs


def _read_points3d_with_ids_text(path):
    ids, xyzs = [], []
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            ids.append(int(elems[0]))
            xyzs.append([float(elems[1]), float(elems[2]), float(elems[3])])
    return np.asarray(ids, dtype=np.int64), np.asarray(xyzs, dtype=np.float64)


def _find_colmap_sparse_dir(spec):
    gt = _spec_gt(spec)
    candidates = []
    if gt.get("colmap_sparse"):
        candidates.append(str(gt["colmap_sparse"]))
    source = getattr(spec, "source_path", "") or ""
    candidates += [os.path.join(source, "sparse", "0"), os.path.join(source, "sparse")]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "images.bin")) or \
           os.path.isfile(os.path.join(cand, "images.txt")):
            return cand
    return None


def _load_colmap_sparse(sparse_dir):
    images_bin = os.path.join(sparse_dir, "images.bin")
    points_bin = os.path.join(sparse_dir, "points3D.bin")
    if os.path.isfile(images_bin):
        images = read_extrinsics_binary(images_bin)
    else:
        images = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    if os.path.isfile(points_bin):
        pt_ids, pt_xyz = _read_points3d_with_ids_binary(points_bin)
    else:
        pt_ids, pt_xyz = _read_points3d_with_ids_text(os.path.join(sparse_dir, "points3D.txt"))
    return images, pt_ids, pt_xyz


# ---------------------------------------------------------------------------
# Projection with the repo's exact camera conventions
# ---------------------------------------------------------------------------

def _project_points(points_world, cam):
    """Project world points into camera `cam` using the repo's exact
    (CUDA-matching) transforms from triangle_renderer.

    Returns (depth [N], px [N], py [N], in_front [N]) as torch tensors on the
    points' device. Pixel coords follow ndc2Pix: pixel i's center is at i.
    """
    from triangle_renderer import transform_point_4x3, compute_image_2d_pytorch_exact

    p_cam = transform_point_4x3(points_world, cam.world_view_transform.to(points_world.device))
    depth = p_cam[:, 2]
    W = int(cam.image_width)
    H = int(cam.image_height)
    pix = compute_image_2d_pytorch_exact(
        points_world, cam.full_proj_transform.to(points_world.device), W, H)
    px = torch.round(pix[:, 0]).long()
    py = torch.round(pix[:, 1]).long()
    in_front = depth > float(getattr(cam, "znear", 0.01))
    return depth, px, py, in_front


# ---------------------------------------------------------------------------
# g1 — free-space violation rate
# ---------------------------------------------------------------------------

def _g1_pairs_from_colmap(ctx):
    """Build (train-camera index, world point) pairs from the COLMAP sparse
    model: points visible per image via the point2D -> point3D tracks,
    capped at G1_MAX_PAIRS_COLMAP total with seed 0."""
    sparse_dir = _find_colmap_sparse_dir(ctx.spec)
    if sparse_dir is None:
        return None, None, "skipped: no COLMAP sparse model found for scene"
    images, pt_ids, pt_xyz = _load_colmap_sparse(sparse_dir)
    id_to_row = {int(pid): row for row, pid in enumerate(pt_ids)}
    colmap_by_key = {}
    for img in images.values():
        colmap_by_key.setdefault(_image_key(img.name), img)

    cam_idx_chunks, xyz_chunks, pid_chunks = [], [], []
    matched_cams = 0
    for cam_idx, cam in enumerate(ctx.train_cams):
        img = colmap_by_key.get(_image_key(getattr(cam, "image_name", "")))
        if img is None:
            continue
        matched_cams += 1
        visible = img.point3D_ids[img.point3D_ids >= 0]
        if visible.size == 0:
            continue
        rows = np.asarray([id_to_row[int(pid)] for pid in visible if int(pid) in id_to_row],
                          dtype=np.int64)
        if rows.size == 0:
            continue
        cam_idx_chunks.append(np.full(rows.shape[0], cam_idx, dtype=np.int64))
        xyz_chunks.append(pt_xyz[rows])
        pid_chunks.append(pt_ids[rows])

    if matched_cams == 0:
        return None, None, "skipped: no train camera matched a COLMAP image name"
    if not cam_idx_chunks:
        return None, None, "skipped: COLMAP model has no visible points in train cameras"

    cam_indices = np.concatenate(cam_idx_chunks)
    xyz = np.concatenate(xyz_chunks, axis=0)
    pids = np.concatenate(pid_chunks)
    n_total = cam_indices.shape[0]
    if n_total > G1_MAX_PAIRS_COLMAP:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(n_total, size=G1_MAX_PAIRS_COLMAP, replace=False)
        keep.sort()
        cam_indices, xyz, pids = cam_indices[keep], xyz[keep], pids[keep]
    pairs = {"cam_indices": cam_indices, "xyz": xyz, "point3d_ids": pids}
    meta = {"source": "colmap_sparse", "sparse_dir": sparse_dir,
            "matched_train_cams": matched_cams, "n_pairs_before_cap": int(n_total)}
    return pairs, meta, None


@torch.no_grad()
def _g1_colmap(ctx, out_dir):
    pairs, meta, skip = _g1_pairs_from_colmap(ctx)
    if skip is not None:
        return {"skipped": skip}

    cam_indices = pairs["cam_indices"]
    xyz_all = torch.from_numpy(np.ascontiguousarray(pairs["xyz"])).float().cuda()

    records = {k: [] for k in ("cam_index", "point3d_id", "px", "py",
                               "point_depth", "rendered_depth", "rendered_alpha",
                               "violation")}
    n_dropped = 0
    for cam_idx in np.unique(cam_indices):
        cam = ctx.train_cams[int(cam_idx)]
        sel = np.nonzero(cam_indices == cam_idx)[0]
        pts = xyz_all[torch.from_numpy(sel).cuda()]
        depth_p, px, py, in_front = _project_points(pts, cam)
        W, H = int(cam.image_width), int(cam.image_height)
        ok = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        n_dropped += int((~ok).sum().item())
        if not bool(ok.any().item()):
            continue
        pkg = ctx.render_view(cam)
        depth_map = pkg["surf_depth"].detach()   # [1, H, W] median depth
        alpha_map = pkg["rend_alpha"].detach()   # [1, H, W]
        px_k, py_k = px[ok], py[ok]
        d_p = depth_p[ok]
        d_r = depth_map[0, py_k, px_k]
        a_r = alpha_map[0, py_k, px_k]
        viol = (d_r < G1_DEPTH_RATIO * d_p) & (a_r >= G1_ALPHA_MIN)
        sel_k = sel[ok.cpu().numpy()]
        records["cam_index"].append(np.full(sel_k.shape[0], cam_idx, dtype=np.int32))
        records["point3d_id"].append(pairs["point3d_ids"][sel_k])
        records["px"].append(px_k.cpu().numpy().astype(np.int32))
        records["py"].append(py_k.cpu().numpy().astype(np.int32))
        records["point_depth"].append(d_p.cpu().numpy().astype(np.float32))
        records["rendered_depth"].append(d_r.cpu().numpy().astype(np.float32))
        records["rendered_alpha"].append(a_r.cpu().numpy().astype(np.float32))
        records["violation"].append(viol.cpu().numpy())
        del pkg
        torch.cuda.empty_cache()

    if not records["violation"]:
        return {"skipped": "skipped: no COLMAP pairs projected inside any train view"}
    arrays = {k: np.concatenate(v) for k, v in records.items()}
    npz_path = os.path.join(out_dir, "g1_free_space_samples.npz")
    np.savez_compressed(npz_path, **arrays)
    violations = arrays["violation"]
    return {
        "value": float(violations.mean()),
        "n_samples": int(violations.shape[0]),
        "n_violations": int(violations.sum()),
        "n_pairs_dropped_out_of_frame": int(n_dropped),
        "per_sample_npz": npz_path,
        **meta,
    }


def _find_gt_depth_file(gt_depth_dir, image_name):
    stem = _image_key(image_name)
    for cand in (f"{stem}.npy", f"{image_name}.npy", f"depth_{stem}.npy"):
        path = os.path.join(gt_depth_dir, cand)
        if os.path.isfile(path):
            return path
    return None


@torch.no_grad()
def _g1_toy(ctx, out_dir, gt_depth_dir):
    upm = _units_per_meter(ctx.spec)
    rng = np.random.default_rng(SEED)
    records = {k: [] for k in ("cam_index", "px", "py", "point_depth",
                               "rendered_depth", "rendered_alpha", "violation")}
    missing = []
    for cam_idx, cam in enumerate(ctx.test_cams):
        depth_file = _find_gt_depth_file(gt_depth_dir, getattr(cam, "image_name", ""))
        if depth_file is None:
            missing.append(_image_key(getattr(cam, "image_name", "")))
            continue
        gt_depth = np.load(depth_file).astype(np.float32) * upm  # meters -> scene units
        H, W = int(cam.image_height), int(cam.image_width)
        if gt_depth.shape != (H, W):
            raise ValueError(
                f"g1: GT depth shape {gt_depth.shape} != render shape {(H, W)} "
                f"for view {getattr(cam, 'image_name', cam_idx)}")
        valid = np.isfinite(gt_depth) & (gt_depth > 0)
        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            continue
        if ys.size > G1_TOY_PIXELS_PER_VIEW:
            keep = rng.choice(ys.size, size=G1_TOY_PIXELS_PER_VIEW, replace=False)
            ys, xs = ys[keep], xs[keep]
        pkg = ctx.render_view(cam)
        depth_map = pkg["surf_depth"].detach()
        alpha_map = pkg["rend_alpha"].detach()
        py = torch.from_numpy(ys).long().cuda()
        px = torch.from_numpy(xs).long().cuda()
        d_p = torch.from_numpy(gt_depth[ys, xs]).float().cuda()
        d_r = depth_map[0, py, px]
        a_r = alpha_map[0, py, px]
        viol = (d_r < G1_DEPTH_RATIO * d_p) & (a_r >= G1_ALPHA_MIN)
        records["cam_index"].append(np.full(ys.shape[0], cam_idx, dtype=np.int32))
        records["px"].append(xs.astype(np.int32))
        records["py"].append(ys.astype(np.int32))
        records["point_depth"].append(gt_depth[ys, xs])
        records["rendered_depth"].append(d_r.cpu().numpy().astype(np.float32))
        records["rendered_alpha"].append(a_r.cpu().numpy().astype(np.float32))
        records["violation"].append(viol.cpu().numpy())
        del pkg
        torch.cuda.empty_cache()

    if not records["violation"]:
        return {"skipped": f"skipped: no GT depth files matched test views in {gt_depth_dir}"}
    arrays = {k: np.concatenate(v) for k, v in records.items()}
    npz_path = os.path.join(out_dir, "g1_free_space_samples.npz")
    np.savez_compressed(npz_path, **arrays)
    violations = arrays["violation"]
    out = {
        "value": float(violations.mean()),
        "n_samples": int(violations.shape[0]),
        "n_violations": int(violations.sum()),
        "source": "gt_depth",
        "per_sample_npz": npz_path,
    }
    if missing:
        out["views_missing_gt_depth"] = missing
    return out


def compute_g1(ctx):
    out_dir = _out_subdir(ctx)
    gt = _spec_gt(ctx.spec)
    if gt.get("gt_depth_dir") and os.path.isdir(str(gt["gt_depth_dir"])):
        return _g1_toy(ctx, out_dir, str(gt["gt_depth_dir"]))
    return _g1_colmap(ctx, out_dir)


# ---------------------------------------------------------------------------
# g2 — held-out depth L1 (toy only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_g2(ctx):
    gt = _spec_gt(ctx.spec)
    gt_depth_dir = gt.get("gt_depth_dir")
    if not gt_depth_dir or not os.path.isdir(str(gt_depth_dir)):
        return {"skipped": "skipped: no gt_depth_dir declared for scene (toy only)"}
    gt_depth_dir = str(gt_depth_dir)
    out_dir = _out_subdir(ctx)
    upm = _units_per_meter(ctx.spec)

    per_view_l1 = []
    view_names = []
    for cam in ctx.test_cams:
        depth_file = _find_gt_depth_file(gt_depth_dir, getattr(cam, "image_name", ""))
        if depth_file is None:
            continue
        gt_depth = np.load(depth_file).astype(np.float32)  # meters
        H, W = int(cam.image_height), int(cam.image_width)
        if gt_depth.shape != (H, W):
            raise ValueError(
                f"g2: GT depth shape {gt_depth.shape} != render shape {(H, W)} "
                f"for view {getattr(cam, 'image_name', '?')}")
        pkg = ctx.render_view(cam)
        depth_map = pkg["surf_depth"].detach()[0].cpu().numpy() / upm  # -> meters
        alpha_map = pkg["rend_alpha"].detach()[0].cpu().numpy()
        valid = np.isfinite(gt_depth) & (gt_depth > 0) & (alpha_map >= G2_ALPHA_MIN)
        if valid.sum() == 0:
            continue
        per_view_l1.append(float(np.abs(depth_map[valid] - gt_depth[valid]).mean()))
        view_names.append(_image_key(getattr(cam, "image_name", "")))
        del pkg
        torch.cuda.empty_cache()

    if not per_view_l1:
        return {"skipped": f"skipped: no test view had GT depth + valid pixels in {gt_depth_dir}"}
    per_view = np.asarray(per_view_l1, dtype=np.float32)
    npz_path = os.path.join(out_dir, "g2_depth_l1_per_view.npz")
    np.savez_compressed(npz_path, per_view_l1_m=per_view,
                        view_names=np.asarray(view_names))
    return {
        "value": float(per_view.mean()),
        "n_views": int(per_view.shape[0]),
        "per_sample_npz": npz_path,
    }


# ---------------------------------------------------------------------------
# g3 — floater score
# ---------------------------------------------------------------------------

def _mesh_component_labels(faces_np, n_vertices):
    """Connected components of triangles sharing a vertex, computed on the
    vertex graph (edges = triangle edges) so 11.5M faces stay tractable.
    Returns per-face component labels (labels over the vertex graph)."""
    e0 = faces_np[:, [0, 1]]
    e1 = faces_np[:, [1, 2]]
    e2 = faces_np[:, [2, 0]]
    edges = np.concatenate([e0, e1, e2], axis=0)
    graph = coo_matrix(
        (np.ones(edges.shape[0], dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices),
    )
    _, vertex_labels = connected_components(csgraph=graph, directed=False)
    # A triangle's 3 vertices are mutually connected, so any vertex works.
    return vertex_labels[faces_np[:, 0]]


@torch.no_grad()
def _support_counts(ctx, n_triangles, max_views=G3_MAX_SUPPORT_VIEWS):
    """Per-triangle count of train views where the triangle owns >= 1 pixel,
    read from rend_ids over <= max_views evenly spaced train views.

    rend_ids is at the supersampled render resolution and is UNINITIALIZED on
    pixels where no triangle reached the median-depth test (background), so it
    must be gated by depth_full > 0 (median depth at the same resolution).
    """
    n_views = min(int(max_views), len(ctx.train_cams))
    view_indices = np.unique(np.linspace(0, len(ctx.train_cams) - 1, n_views).round().astype(int))
    support = torch.zeros(n_triangles, dtype=torch.int32, device="cuda")
    for view_idx in view_indices:
        cam = ctx.train_cams[int(view_idx)]
        pkg = ctx.render_view(cam)
        ids = pkg["rend_ids"].detach()
        depth_full = pkg.get("depth_full")
        ids = ids.reshape(-1)
        valid = (ids >= 0) & (ids < n_triangles)
        if depth_full is not None:
            valid &= depth_full.detach().reshape(-1) > 0
        visible = torch.unique(ids[valid].round().long())
        support[visible] += 1
        del pkg, ids, valid, visible
        torch.cuda.empty_cache()
    return support.cpu().numpy(), view_indices


def _finite_faces_mask(ctx, faces_np):
    """Bool [T] over ORIGINAL face order: all-3-vertices-finite (PROTOCOL §4.3)."""
    fn = getattr(ctx, "finite_faces_mask", None)
    if callable(fn):
        return fn().detach().cpu().numpy().astype(bool)
    verts = ctx.vertices().detach().cpu().numpy()
    finite_v = np.isfinite(verts).all(axis=1)
    return finite_v[faces_np].all(axis=1)


def compute_g3(ctx):
    out_dir = _out_subdir(ctx)
    faces_np = ctx.faces().detach().cpu().numpy().astype(np.int64)
    n_vertices = int(ctx.vertices().shape[0])
    n_triangles = int(faces_np.shape[0])
    if n_triangles == 0:
        return {"skipped": "skipped: model has no triangles"}
    if n_triangles >= 2 ** 24:
        # rend_ids is float32; integer ids above 2^24 are not exactly representable.
        return {"skipped": "skipped: triangle count >= 2^24 exceeds rend_ids float32 precision"}

    # PROTOCOL §4.3: faces touching non-finite vertices are excluded from the
    # geometry surface (rasterizer culls them); indexing stays original.
    finite_mask = _finite_faces_mask(ctx, faces_np)
    n_nonfinite = int((~finite_mask).sum())

    face_labels = _mesh_component_labels(faces_np, n_vertices)
    support, view_indices = _support_counts(ctx, n_triangles)

    # Reduce per component (labels come from the vertex graph; only components
    # that actually contain triangles are counted). Non-finite faces are
    # excluded from component stats and can never be floaters.
    n_labels = int(face_labels.max()) + 1
    fin_labels = face_labels[finite_mask]
    comp_sizes = np.bincount(fin_labels, minlength=n_labels)
    comp_max_support = np.zeros(n_labels, dtype=np.int64)
    np.maximum.at(comp_max_support, fin_labels, support[finite_mask].astype(np.int64))
    has_faces = comp_sizes > 0

    is_floater = (
        has_faces
        & (comp_max_support <= G3_FLOATER_MAX_SUPPORT)
        & (comp_sizes < G3_FLOATER_MAX_SIZE)
    )
    floater_face_mask = is_floater[face_labels] & finite_mask
    floater_tri_ids = np.nonzero(floater_face_mask)[0].astype(np.int64)

    npz_path = os.path.join(out_dir, "g3_floaters.npz")
    comp_ids = np.nonzero(has_faces)[0]
    np.savez_compressed(
        npz_path,
        floater_tri_ids=floater_tri_ids,
        component_sizes=comp_sizes[comp_ids].astype(np.int64),
        component_max_support=comp_max_support[comp_ids],
        component_is_floater=is_floater[comp_ids],
        triangle_support=support.astype(np.int32),
        support_view_indices=view_indices.astype(np.int64),
    )
    return {
        "floater_component_count": int(is_floater.sum()),
        "floater_triangle_fraction": float(floater_face_mask.mean()),
        "n_components": int(has_faces.sum()),
        "n_triangles": n_triangles,
        "n_nonfinite_faces_excluded": n_nonfinite,
        "n_support_views": int(view_indices.shape[0]),
        "floater_tri_ids_npz": npz_path,
        "per_sample_npz": npz_path,
    }


# ---------------------------------------------------------------------------
# g4 — Chamfer-L1 / F-score@tau
# ---------------------------------------------------------------------------

def _sample_points_on_mesh(verts, faces, n_samples, rng):
    """Area-weighted uniform surface sampling. verts [V,3] float64 np,
    faces [T,3] int np. Returns [n_samples,3] float64."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("g4: mesh has zero total surface area")
    face_idx = rng.choice(faces.shape[0], size=n_samples, p=areas / total)
    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    return (w0[:, None] * a[face_idx]
            + w1[:, None] * b[face_idx]
            + w2[:, None] * c[face_idx])


def _crop_to_roi(points, roi):
    if not roi:
        return points, None
    lo = np.asarray(roi["min"], dtype=np.float64)
    hi = np.asarray(roi["max"], dtype=np.float64)
    keep = np.all((points >= lo[None, :]) & (points <= hi[None, :]), axis=1)
    return points[keep], keep


def _load_gt_points(spec, rng):
    """Returns (gt_points [N,3] float64, source_str) or raises FileNotFoundError."""
    gt = _spec_gt(spec)
    mesh_path = gt.get("mesh_path")
    scan_paths = gt.get("scan_paths")
    if mesh_path and os.path.isfile(str(mesh_path)):
        import trimesh
        mesh = trimesh.load(str(mesh_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate([g for g in mesh.dump()])
        pts = _sample_points_on_mesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            G4_GT_MESH_SAMPLES, rng)
        return pts, f"gt_mesh:{mesh_path}", "mesh"
    if scan_paths:
        import trimesh
        clouds = []
        for scan in scan_paths:
            if not os.path.isfile(str(scan)):
                raise FileNotFoundError(f"scan not found: {scan}")
            scan_obj = trimesh.load(str(scan), process=False)
            clouds.append(np.asarray(scan_obj.vertices, dtype=np.float64))
        pts = np.concatenate(clouds, axis=0)
        if pts.shape[0] > G4_SCAN_SUBSAMPLE:
            keep = rng.choice(pts.shape[0], size=G4_SCAN_SUBSAMPLE, replace=False)
            pts = pts[keep]
        return pts, "laser_scans:" + ",".join(str(s) for s in scan_paths), "scan"
    raise FileNotFoundError("no GT mesh or scan declared")


def _chamfer_fscore(recon_query, gt_pts, recon_targets, tau):
    """Chamfer-L1 and F@tau between point sets (all in scene units).

    recon_query : recon points used as queries recon->gt (may be a filtered
                  subset under the courtyard bounding-region rule)
    recon_targets : recon points used as NN targets for gt->recon
    Returns (chamfer, precision, recall, fscore, d_recon_to_gt, d_gt_to_recon).
    """
    gt_tree = cKDTree(gt_pts)
    d_recon, _ = gt_tree.query(recon_query, k=1, workers=-1)
    recon_tree = cKDTree(recon_targets)
    d_gt, _ = recon_tree.query(gt_pts, k=1, workers=-1)
    chamfer = 0.5 * (float(d_recon.mean()) + float(d_gt.mean()))
    precision = float((d_recon < tau).mean())
    recall = float((d_gt < tau).mean())
    fscore = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return chamfer, precision, recall, fscore, d_recon, d_gt


def _gt_is_scan(gt):
    """True when the GT surface asset that _load_gt_points would use is a
    laser scan (mesh takes precedence when its file exists)."""
    mesh_path = gt.get("mesh_path")
    if mesh_path and os.path.isfile(str(mesh_path)):
        return False
    return bool(gt.get("scan_paths"))


def compute_g4(ctx):
    roi = getattr(ctx.spec, "roi", None)
    gt = _spec_gt(ctx.spec)
    if not (gt.get("mesh_path") or gt.get("scan_paths")):
        return {"skipped": "skipped: no GT mesh/scan declared for scene", "roi": roi}
    # PROTOCOL 1.1.0 §4.3: when the GT is a scan and the scene ROI is not yet
    # frozen, g4 is skipped — never reported as a protocol number. This
    # mirrors the run_eval downstream (d1/d2) gate on spec.roi.
    if _gt_is_scan(gt) and roi is None:
        return {"skipped": "ROI not yet frozen in scenes.py", "roi": None}
    out_dir = _out_subdir(ctx)
    upm = _units_per_meter(ctx.spec)
    tau = G4_TAU_METERS * upm
    rng = np.random.default_rng(SEED)

    # PROTOCOL 1.1.0 §4.3: reconstructed surface = ALL checkpoint triangles
    # (opacity floor >= 0.999 makes every triangle part of the rendered
    # surface; the former sigmoid(weight) >= 0.5 mask is gone), minus faces
    # touching non-finite vertices (rasterizer culls those; count reported).
    faces = ctx.faces().detach().cpu().numpy().astype(np.int64)
    if faces.shape[0] == 0:
        return {"skipped": "skipped: model has no triangles", "roi": roi}
    verts = ctx.vertices().detach().cpu().numpy().astype(np.float64)
    finite_mask = _finite_faces_mask(ctx, faces)
    n_nonfinite = int((~finite_mask).sum())
    faces = faces[finite_mask]
    if faces.shape[0] == 0:
        return {"skipped": "skipped: no finite triangles", "roi": roi}
    recon_pts = _sample_points_on_mesh(verts, faces, G4_RECON_SAMPLES, rng)

    try:
        gt_pts, gt_source, gt_kind = _load_gt_points(ctx.spec, rng)
    except FileNotFoundError as exc:
        return {"skipped": f"skipped: GT asset missing ({exc})", "roi": roi}

    recon_pts, _ = _crop_to_roi(recon_pts, roi)
    gt_pts, _ = _crop_to_roi(gt_pts, roi)
    if recon_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return {"skipped": "skipped: ROI crop left an empty point cloud", "roi": roi}

    # Courtyard scan-direction rule (PROTOCOL §4.3 g4): recon->scan only counts
    # recon points within 0.3 m of the scan points' bounding region.
    recon_query = recon_pts
    n_excluded = 0
    if gt_kind == "scan":
        margin = G4_SCAN_BBOX_MARGIN_METERS * upm
        lo = gt_pts.min(axis=0) - margin
        hi = gt_pts.max(axis=0) + margin
        keep = np.all((recon_pts >= lo[None, :]) & (recon_pts <= hi[None, :]), axis=1)
        n_excluded = int((~keep).sum())
        recon_query = recon_pts[keep]
        if recon_query.shape[0] == 0:
            return {"skipped": "skipped: no recon points within the scan bounding region",
                    "roi": roi}

    chamfer, precision, recall, fscore, d_recon, d_gt = _chamfer_fscore(
        recon_query, gt_pts, recon_pts, tau)

    # PROTOCOL 1.1.0 §4.3 pairing rule: only the gt->recon per-sample
    # distances are a pairable bootstrap unit across models (GT sample points
    # are model-independent) and may feed paired_bootstrap_ci; the recon->gt
    # per-sample array is UNPAIRED (recon sample points differ per model) and
    # is stored only as an unpaired summary.
    npz_path = os.path.join(out_dir, "g4_chamfer_samples.npz")
    np.savez_compressed(
        npz_path,
        recon_to_gt_dist_unpaired=d_recon.astype(np.float32),
        gt_to_recon_dist=d_gt.astype(np.float32),
        units_per_meter=np.float64(upm),
    )
    return {
        "chamfer_l1_m": chamfer / upm,
        "fscore_at_tau": fscore,
        "precision_at_tau": precision,
        "recall_at_tau": recall,
        "tau_m": G4_TAU_METERS,
        "roi": roi,
        "n_recon_samples": int(recon_pts.shape[0]),
        "n_recon_query": int(recon_query.shape[0]),
        "n_recon_excluded_by_scan_region": n_excluded,
        "n_gt_samples": int(gt_pts.shape[0]),
        "n_surface_triangles": int(faces.shape[0]),
        "n_nonfinite_faces_excluded": n_nonfinite,
        "surface_definition": "all_checkpoint_triangles (PROTOCOL 1.1.0 §4.3)",
        "pairing": "only gt_to_recon_dist feeds paired_bootstrap_ci; "
                   "recon_to_gt_dist_unpaired is an unpaired summary",
        "gt_source": gt_source,
        "per_sample_npz": npz_path,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_geometry_metrics(ctx):
    """PROTOCOL.md §4.3 g1-g4. Returns a dict with subkeys g1..g4; each is a
    metric dict (scalar summary + per-sample npz path) or
    {'skipped': '<reason>'} when the required GT assets are absent."""
    return {
        "g1": compute_g1(ctx),
        "g2": compute_g2(ctx),
        "g3": compute_g3(ctx),
        "g4": compute_g4(ctx),
    }
