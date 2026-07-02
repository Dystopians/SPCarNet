"""GEMS Stage-One downstream proxy metrics (PROTOCOL.md §4.4, v1.1.0).

Implements
  d1 — occupancy agreement: voxelize the reconstructed triangle surface (ALL
       checkpoint triangles per PROTOCOL 1.1.0 §4.3) and the GT surface at
       0.10 m into the scene ROI (ground-relevant z-band). Reports
       false_free_rate = P(recon free | GT occupied)
       (safety-critical) and false_occupied_rate = P(recon occupied | GT free).
  d2 — collision-verdict agreement: 200 trajectories (numpy default_rng,
       seed 0), half straight lines / half constant-curvature arcs
       (radius ~ U[4, 20] m), random start pose inside the ROI ground area,
       length ~ U[5, 15] m, swept vehicle footprint 4.5 x 1.8 m in the height
       band [0.1, 1.5] m above the ground (ground = z_band[0]). The identical
       trajectory set is evaluated on BOTH occupancy grids (paired). Reports
       agreement_rate and unsafe_disagreement_rate =
       P(recon says free | GT says collision).

Purity: pure numpy (+ trimesh only for loading GT assets, with a plain
v/f OBJ-parser fallback). No renderer, no repo model imports — the module is
importable standalone and runs on CPU only.
"""

from __future__ import annotations

import numpy as np

VEHICLE_LENGTH_M = 4.5
VEHICLE_WIDTH_M = 1.8
VEHICLE_Z_BAND_M = (0.1, 1.5)  # above ground (= z_band[0])
TRAJ_LENGTH_RANGE_M = (5.0, 15.0)
ARC_RADIUS_RANGE_M = (4.0, 20.0)
# Max barycentric subdivisions per triangle in one lattice pass. NOT a silent
# cap (PROTOCOL 1.1.0 §4.4 forbids those): triangles that would need more
# subdivisions to reach voxel/2 point spacing are recursively midpoint-split
# first (_split_oversized_triangles); if that cannot converge (non-finite
# vertices) a ValueError is raised.
_MAX_TRI_SUBDIV = 1024
_MAX_SPLIT_ROUNDS = 64  # each round halves edge lengths; 64 is unreachable
                        # for finite meshes and guards infinite loops


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _load_obj_plain(path):
    """Minimal OBJ parser (plain 'v x y z' / 'f i j k' lines, 1-based or
    negative indices, 'f a/b/c' vertex/uv/normal syntax, polygon fan
    triangulation). Fallback when trimesh is unavailable."""
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):  # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return verts, faces


def _load_gt_mesh(mesh_path):
    """Load a GT mesh as (verts [N,3], faces [T,3]) numpy arrays."""
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        return (np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64))
    except ImportError:
        return _load_obj_plain(mesh_path)


# ---------------------------------------------------------------------------
# Voxel grid
# ---------------------------------------------------------------------------

class _VoxelGrid:
    """Axis-aligned occupancy grid: xy extent from roi min/max, z extent from
    roi z_band (the frozen ground-relevant band), cell size `voxel` meters.
    Cell (i,j,k) covers [origin + (i,j,k)*voxel, origin + (i+1,j+1,k+1)*voxel)."""

    def __init__(self, roi, voxel):
        rmin = np.asarray(roi["min"], dtype=np.float64)
        rmax = np.asarray(roi["max"], dtype=np.float64)
        z_lo, z_hi = float(roi["z_band"][0]), float(roi["z_band"][1])
        self.voxel = float(voxel)
        self.origin = np.array([rmin[0], rmin[1], z_lo], dtype=np.float64)
        upper = np.array([rmax[0], rmax[1], z_hi], dtype=np.float64)
        self.shape = np.maximum(
            np.ceil((upper - self.origin) / self.voxel - 1e-9).astype(np.int64), 1)
        self.z_lo, self.z_hi = z_lo, z_hi

    def new_occupancy(self):
        return np.zeros(tuple(self.shape), dtype=bool)

    def mark_points(self, occ, points):
        """Set occupied every cell containing one of `points` [N,3]."""
        if points.size == 0:
            return
        idx = np.floor((points - self.origin[None, :]) / self.voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < self.shape[None, :]), axis=1)
        idx = idx[ok]
        if idx.size:
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    def z_slice(self, band_lo, band_hi):
        """Half-open z-layer index range [k0, k1) intersecting absolute band
        [band_lo, band_hi]."""
        k0 = int(np.floor((band_lo - self.z_lo) / self.voxel + 1e-9))
        k1 = int(np.ceil((band_hi - self.z_lo) / self.voxel - 1e-9))
        return max(k0, 0), min(k1, int(self.shape[2]))


def _tri_max_edge(tri):
    """Longest edge length per triangle. tri [T,3,3] -> [T]."""
    edges = np.stack([tri[:, 1] - tri[:, 0],
                      tri[:, 2] - tri[:, 1],
                      tri[:, 0] - tri[:, 2]], axis=1)
    return np.linalg.norm(edges, axis=2).max(axis=1)


def _split_oversized_triangles(tri, spacing):
    """Recursively midpoint-4-split triangles whose barycentric lattice would
    need more than _MAX_TRI_SUBDIV subdivisions to reach `spacing` point
    spacing (PROTOCOL 1.1.0 §4.4: sampling spacing <= voxel/2 must hold for
    EVERY triangle; silently capping the subdivision count is forbidden).

    Each 4-split halves all edge lengths, so the required subdivision count
    halves per round and the recursion terminates for any finite mesh. Raises
    ValueError if it cannot converge (non-finite vertex coordinates).
    """
    limit = float(_MAX_TRI_SUBDIV) * spacing  # max_edge allowed per lattice pass
    ok_parts = []
    work = tri
    for _ in range(_MAX_SPLIT_ROUNDS):
        if work.shape[0] == 0:
            break
        max_edge = _tri_max_edge(work)
        if not np.isfinite(max_edge).all():
            raise ValueError(
                "downstream voxelization: non-finite triangle vertex "
                "coordinates; cannot guarantee spacing <= voxel/2 "
                "(PROTOCOL §4.4)")
        oversized = max_edge > limit
        ok_parts.append(work[~oversized])
        bad = work[oversized]
        if bad.shape[0] == 0:
            work = bad
            break
        a, b, c = bad[:, 0], bad[:, 1], bad[:, 2]
        m01, m12, m20 = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
        work = np.concatenate([
            np.stack([a, m01, m20], axis=1),
            np.stack([m01, b, m12], axis=1),
            np.stack([m20, m12, c], axis=1),
            np.stack([m01, m12, m20], axis=1),
        ], axis=0)
    else:
        raise ValueError(
            f"downstream voxelization: triangle subdivision did not converge "
            f"within {_MAX_SPLIT_ROUNDS} split rounds (max edge still > "
            f"{limit:.3g} m); cannot guarantee sampling spacing <= voxel/2 "
            f"(PROTOCOL §4.4 forbids silently capping the sampling density)")
    return np.concatenate(ok_parts, axis=0)


def _rasterize_triangles(grid, occ, verts, faces, chunk_points=4_000_000):
    """Conservatively voxelize triangle surfaces: sample each triangle on a
    barycentric grid at <= voxel/2 point spacing and mark containing cells.

    Triangles too large for a single <= _MAX_TRI_SUBDIV lattice are first
    recursively midpoint-split (never silently capped, PROTOCOL 1.1.0 §4.4).
    """
    if faces.size == 0:
        return
    tri = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]  # [T,3,3]

    # Cull triangles whose bbox misses the grid.
    lo = grid.origin
    hi = grid.origin + grid.shape * grid.voxel
    keep = np.all(tri.max(axis=1) >= lo[None, :], axis=1) & \
           np.all(tri.min(axis=1) <= hi[None, :], axis=1)
    tri = tri[keep]
    if tri.shape[0] == 0:
        return

    spacing = grid.voxel / 2.0
    tri = _split_oversized_triangles(tri, spacing)
    if tri.shape[0] == 0:
        return
    max_edge = _tri_max_edge(tri)
    n_sub = np.maximum(np.ceil(max_edge / spacing).astype(np.int64), 1)
    if (n_sub > _MAX_TRI_SUBDIV).any():
        raise ValueError(
            "downstream voxelization: internal error — a triangle still needs "
            f"more than {_MAX_TRI_SUBDIV} subdivisions after splitting "
            "(PROTOCOL §4.4 guard)")

    for n in np.unique(n_sub):
        sel = tri[n_sub == n]
        # Barycentric lattice {(i/n, j/n) : i+j <= n} — spacing <= voxel/2.
        ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = (ii + jj) <= n
        u = (ii[m] / n)[None, :, None]  # [1,M,1]
        v = (jj[m] / n)[None, :, None]
        m_pts = int(m.sum())
        step = max(1, chunk_points // m_pts)
        for s in range(0, sel.shape[0], step):
            t = sel[s:s + step]
            a = t[:, 0][:, None, :]
            pts = a + u * (t[:, 1][:, None, :] - a) + v * (t[:, 2][:, None, :] - a)
            grid.mark_points(occ, pts.reshape(-1, 3))


def _build_gt_occupancy(grid, gt):
    """Build the GT occupancy grid from gt['mesh_path'] (.obj mesh) or
    gt['scan_points'] (np [N,3] laser-scan points)."""
    occ = grid.new_occupancy()
    if gt.get("mesh_path"):
        verts, faces = _load_gt_mesh(gt["mesh_path"])
        _rasterize_triangles(grid, occ, verts, faces)
    elif gt.get("scan_points") is not None:
        pts = np.asarray(gt["scan_points"], dtype=np.float64).reshape(-1, 3)
        grid.mark_points(occ, pts)
    else:
        raise ValueError(
            "gt dict must provide 'mesh_path' (.obj) or 'scan_points' [N,3]")
    return occ


# ---------------------------------------------------------------------------
# d2 trajectories
# ---------------------------------------------------------------------------

def _sample_trajectories(rng, roi, n_traj):
    """Sample n_traj poses+shapes: first half straight, second half arcs.
    Returns list of dicts (consumed identically for both grids — paired)."""
    rmin = np.asarray(roi["min"], dtype=np.float64)
    rmax = np.asarray(roi["max"], dtype=np.float64)
    trajs = []
    n_straight = n_traj // 2
    for t in range(n_traj):
        start = rng.uniform(rmin[:2], rmax[:2])
        heading = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(*TRAJ_LENGTH_RANGE_M)
        if t < n_straight:
            trajs.append({"kind": "straight", "start": start,
                          "heading": heading, "length": length})
        else:
            radius = rng.uniform(*ARC_RADIUS_RANGE_M)
            turn = -1.0 if rng.uniform() < 0.5 else 1.0
            trajs.append({"kind": "arc", "start": start, "heading": heading,
                          "length": length, "radius": radius, "turn": turn})
    return trajs


def _swept_footprint_columns(traj, grid):
    """Unique (ix, iy) voxel columns covered by sweeping the vehicle footprint
    along the trajectory (centerline stepped at voxel/2, footprint sampled on
    a voxel/2 lattice, rotated to the local heading)."""
    step = grid.voxel / 2.0
    s = np.arange(0.0, traj["length"] + step, step)
    theta0 = traj["heading"]
    x0, y0 = traj["start"]
    if traj["kind"] == "straight":
        cx = x0 + s * np.cos(theta0)
        cy = y0 + s * np.sin(theta0)
        th = np.full_like(s, theta0)
    else:
        r, sgn = traj["radius"], traj["turn"]
        phi = sgn * s / r
        th = theta0 + phi
        # circle center perpendicular-left (sgn=+1) / right (sgn=-1) of pose
        ccx = x0 - sgn * r * np.sin(theta0)
        ccy = y0 + sgn * r * np.cos(theta0)
        cx = ccx + sgn * r * np.sin(th)
        cy = ccy - sgn * r * np.cos(th)

    nu = int(np.ceil(VEHICLE_LENGTH_M / step)) + 1
    nv = int(np.ceil(VEHICLE_WIDTH_M / step)) + 1
    u = np.linspace(-VEHICLE_LENGTH_M / 2.0, VEHICLE_LENGTH_M / 2.0, nu)
    v = np.linspace(-VEHICLE_WIDTH_M / 2.0, VEHICLE_WIDTH_M / 2.0, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel()[None, :], vv.ravel()[None, :]  # [1,F]

    cth, sth = np.cos(th)[:, None], np.sin(th)[:, None]  # [S,1]
    px = cx[:, None] + uu * cth - vv * sth
    py = cy[:, None] + uu * sth + vv * cth
    ix = np.floor((px.ravel() - grid.origin[0]) / grid.voxel).astype(np.int64)
    iy = np.floor((py.ravel() - grid.origin[1]) / grid.voxel).astype(np.int64)
    ok = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
    if not ok.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    flat = np.unique(ix[ok] * grid.shape[1] + iy[ok])
    return flat // grid.shape[1], flat % grid.shape[1]


def _collision_verdicts(trajs, grid, occ):
    """Per-trajectory collision verdict against `occ` within the vehicle
    height band [0.1, 1.5] m above the ground (= z_band[0])."""
    k0, k1 = grid.z_slice(grid.z_lo + VEHICLE_Z_BAND_M[0],
                          grid.z_lo + VEHICLE_Z_BAND_M[1])
    band2d = occ[:, :, k0:k1].any(axis=2) if k1 > k0 else \
        np.zeros(tuple(grid.shape[:2]), dtype=bool)
    verdicts = np.zeros(len(trajs), dtype=bool)
    for i, traj in enumerate(trajs):
        ix, iy = _swept_footprint_columns(traj, grid)
        verdicts[i] = bool(band2d[ix, iy].any()) if ix.size else False
    return verdicts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_downstream_metrics(verts_np, faces_np, gt, roi,
                               voxel=0.10, n_traj=200, seed=0):
    """PROTOCOL §4.4 downstream proxies.

    Args:
        verts_np: [V,3] float array — reconstructed vertex positions.
        faces_np: [T,3] int array — ALL checkpoint triangle indices
            (PROTOCOL 1.1.0 §4.3: the model's opacity floor makes every
            triangle opaque; no opacity mask is applied).
        gt: dict with 'mesh_path' (.obj GT mesh) OR 'scan_points' (np [N,3]).
        roi: dict with 'min' [x,y(,z)], 'max' [x,y(,z)], 'z_band' [z_lo, z_hi]
            (frozen in tools/gems/scenes.py; z extent comes from z_band,
            z_band[0] is treated as ground level for d2).
        voxel: grid cell size, meters (protocol: 0.10).
        n_traj: number of trajectories (protocol: 200).
        seed: numpy default_rng seed (protocol: 0).

    Returns dict with 'd1' and 'd2' sub-dicts: scalar rates + counts + the
    per-sample boolean arrays that feed tools/gems/paired_bootstrap.py.
    """
    verts_np = np.asarray(verts_np, dtype=np.float64).reshape(-1, 3)
    faces_np = np.asarray(faces_np, dtype=np.int64).reshape(-1, 3)

    grid = _VoxelGrid(roi, voxel)
    recon_occ = grid.new_occupancy()
    _rasterize_triangles(grid, recon_occ, verts_np, faces_np)
    gt_occ = _build_gt_occupancy(grid, gt)

    # ---- d1: occupancy agreement over the z-band ROI voxels ----
    gt_occupied = gt_occ.ravel()
    recon_occupied = recon_occ.ravel()
    free_at_gt_occ = ~recon_occupied[gt_occupied]      # per GT-occupied voxel
    occ_at_gt_free = recon_occupied[~gt_occupied]      # per GT-free voxel
    n_gt_occ = int(gt_occupied.sum())
    n_gt_free = int(gt_occupied.size - n_gt_occ)
    d1 = {
        "false_free_rate": float(free_at_gt_occ.mean()) if n_gt_occ else float("nan"),
        "false_occupied_rate": float(occ_at_gt_free.mean()) if n_gt_free else float("nan"),
        "n_gt_occupied": n_gt_occ,
        "n_gt_free": n_gt_free,
        "n_voxels": int(gt_occupied.size),
        "grid_shape": [int(x) for x in grid.shape],
        "voxel_m": float(voxel),
        "recon_free_at_gt_occupied": free_at_gt_occ,
        "recon_occ_at_gt_free": occ_at_gt_free,
    }

    # ---- d2: paired collision verdicts on both grids ----
    rng = np.random.default_rng(seed)
    trajs = _sample_trajectories(rng, roi, n_traj)
    v_recon = _collision_verdicts(trajs, grid, recon_occ)
    v_gt = _collision_verdicts(trajs, grid, gt_occ)
    agree = v_recon == v_gt
    n_gt_coll = int(v_gt.sum())
    unsafe = float((~v_recon[v_gt]).mean()) if n_gt_coll else float("nan")
    d2 = {
        "agreement_rate": float(agree.mean()),
        "unsafe_disagreement_rate": unsafe,
        "n_traj": int(n_traj),
        "n_gt_collision": n_gt_coll,
        "n_recon_collision": int(v_recon.sum()),
        "recon_verdicts": v_recon,
        "gt_verdicts": v_gt,
        "agreement": agree,
        "seed": int(seed),
    }
    return {"d1": d1, "d2": d2}


# ---------------------------------------------------------------------------
# Self-test (CPU only, synthetic analytic scene)
# ---------------------------------------------------------------------------

def _make_plane(half=15.0, z=0.0):
    v = np.array([[-half, -half, z], [half, -half, z],
                  [half, half, z], [-half, half, z]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, f


def _make_box(cmin, cmax):
    x0, y0, z0 = cmin
    x1, y1, z1 = cmax
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                 dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2],            # bottom
                  [4, 5, 6], [4, 6, 7],            # top
                  [0, 1, 5], [0, 5, 4],            # y = y0
                  [2, 3, 7], [2, 7, 6],            # y = y1
                  [1, 2, 6], [1, 6, 5],            # x = x1
                  [3, 0, 4], [3, 4, 7]],           # x = x0
                 dtype=np.int64)
    return v, f


def _concat_meshes(a, b):
    va, fa = a
    vb, fb = b
    return np.vstack([va, vb]), np.vstack([fa, fb + len(va)])


def test_downstream_smoke():
    """Synthetic scene: 30x30 m ground plane at z=0 plus a 2x2x1 m box at
    x,y in [4,6], z in [0,1]; ROI 10x10 m, z_band [0, 2].

    (a) recon == GT           -> false_free = false_occupied = 0, agreement 1.
    (b) recon misses the box  -> box voxels all false-free (~1.0 on a box-only
        sub-ROI), unsafe_disagreement > 0 for trajectories through the box.
    """
    gt_mesh = _concat_meshes(_make_plane(), _make_box((4.0, 4.0, 0.0),
                                                      (6.0, 6.0, 1.0)))
    roi = {"min": [0.0, 0.0], "max": [10.0, 10.0], "z_band": [0.0, 2.0]}

    # --- (a) perfect recon ---
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        obj = os.path.join(td, "gt.obj")
        with open(obj, "w") as fh:
            for p in gt_mesh[0]:
                fh.write(f"v {p[0]} {p[1]} {p[2]}\n")
            for f in gt_mesh[1]:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        # Exercise both the trimesh and plain-parser GT loaders.
        v_tm, f_tm = _load_gt_mesh(obj)
        v_pl, f_pl = _load_obj_plain(obj)
        assert np.allclose(v_tm, v_pl) and np.array_equal(f_tm, f_pl), \
            "trimesh and plain OBJ loaders disagree"

        res_a = compute_downstream_metrics(gt_mesh[0], gt_mesh[1],
                                           {"mesh_path": obj}, roi)
    assert res_a["d1"]["n_gt_occupied"] > 0
    assert res_a["d1"]["false_free_rate"] == 0.0, res_a["d1"]
    assert res_a["d1"]["false_occupied_rate"] == 0.0, res_a["d1"]
    assert res_a["d2"]["agreement_rate"] == 1.0, res_a["d2"]
    assert res_a["d2"]["n_gt_collision"] > 0, \
        "expected some GT collisions with the box"
    assert res_a["d2"]["unsafe_disagreement_rate"] == 0.0
    print("[a] recon==GT:",
          {k: res_a["d1"][k] for k in
           ("false_free_rate", "false_occupied_rate", "n_gt_occupied")},
          {k: res_a["d2"][k] for k in
           ("agreement_rate", "unsafe_disagreement_rate", "n_gt_collision")})

    # --- (b) recon = ground only (box missing) ---
    recon = _make_plane()
    res_b = compute_downstream_metrics(
        recon[0], recon[1], {"scan_points": _sample_gt_points(gt_mesh)}, roi)
    assert 0.0 < res_b["d1"]["false_free_rate"] < 1.0  # ground found, box missed
    assert res_b["d2"]["n_gt_collision"] > 0
    assert res_b["d2"]["unsafe_disagreement_rate"] > 0.0, res_b["d2"]
    assert res_b["d2"]["agreement_rate"] < 1.0

    # Box-only sub-ROI (z-band above ground, xy around the box): every
    # GT-occupied voxel there must be false-free in the box-less recon.
    roi_box = {"min": [3.8, 3.8], "max": [6.2, 6.2], "z_band": [0.15, 0.95]}
    res_box = compute_downstream_metrics(
        recon[0], recon[1], {"scan_points": _sample_gt_points(gt_mesh)}, roi_box)
    assert res_box["d1"]["n_gt_occupied"] > 0
    assert res_box["d1"]["false_free_rate"] > 0.99, res_box["d1"]
    print("[b] recon misses box:",
          {"false_free_rate_full_roi": res_b["d1"]["false_free_rate"],
           "false_free_rate_box_roi": res_box["d1"]["false_free_rate"],
           "agreement_rate": res_b["d2"]["agreement_rate"],
           "unsafe_disagreement_rate": res_b["d2"]["unsafe_disagreement_rate"],
           "n_gt_collision": res_b["d2"]["n_gt_collision"]})
    print("test_downstream_smoke PASSED")


def test_oversized_triangle_guard():
    """PROTOCOL 1.1.0 §4.4: triangles whose lattice would exceed
    _MAX_TRI_SUBDIV subdivisions are recursively split (spacing <= voxel/2
    preserved), never silently capped; non-finite meshes raise ValueError."""
    roi = {"min": [0.0, 0.0], "max": [10.0, 10.0], "z_band": [0.0, 1.0]}
    grid = _VoxelGrid(roi, 0.10)
    # Giant ground square: 120 m edges (169.7 m diagonal) need ~3394
    # subdivisions at voxel/2 = 0.05 m spacing — beyond _MAX_TRI_SUBDIV, so
    # the recursive split path must engage.
    v, f = _make_plane(half=60.0, z=0.05)
    assert _tri_max_edge(v[f]).max() > _MAX_TRI_SUBDIV * (grid.voxel / 2.0), \
        "test premise broken: triangle not oversized"
    occ = grid.new_occupancy()
    _rasterize_triangles(grid, occ, v, f)
    # The plane crosses every xy column of the ROI at z=0.05 (layer 0):
    # with guaranteed <= voxel/2 sampling no column may be missed.
    assert occ[:, :, 0].all(), \
        f"giant-triangle voxelization left holes ({int((~occ[:, :, 0]).sum())} columns)"
    assert not occ[:, :, 1:].any(), "plane marked voxels above its z layer"

    # Non-finite vertices: must raise, not silently degrade (PROTOCOL 4.4).
    v_bad = v.copy()
    v_bad[0, 0] = np.inf
    try:
        _rasterize_triangles(grid, grid.new_occupancy(), v_bad, f)
        raise AssertionError("expected ValueError for non-finite vertices")
    except ValueError as exc:
        print(f"[guard] non-finite mesh raised as required: {exc}")
    print("test_oversized_triangle_guard PASSED")


def _sample_gt_points(mesh, spacing=0.04):
    """Dense surface points from a (verts, faces) mesh — used to exercise the
    'scan_points' GT path in the self-test."""
    pts = []
    verts, faces = mesh
    tri = verts[faces]
    for t in tri:
        n = max(1, int(np.ceil(max(np.linalg.norm(t[1] - t[0]),
                                   np.linalg.norm(t[2] - t[1]),
                                   np.linalg.norm(t[0] - t[2])) / spacing)))
        n = min(n, _MAX_TRI_SUBDIV)
        ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = (ii + jj) <= n
        u, v = (ii[m] / n)[:, None], (jj[m] / n)[:, None]
        pts.append(t[0] + u * (t[1] - t[0]) + v * (t[2] - t[0]))
    return np.vstack(pts)


if __name__ == "__main__":
    test_downstream_smoke()
    test_oversized_triangle_guard()
