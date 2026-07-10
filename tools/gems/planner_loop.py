#!/usr/bin/env python
"""GEMS Stage-1R R3.c — planner closed loop v0 (LEDGER GOAL #R-03).

Hybrid-A*-lite planner consuming the R3.a occupancy grids (LEDGER GOAL #R-02):

  planner   — A* on an SE(2) lattice (cell 0.10 m = the d1 voxel, 16 heading
              bins) with straight + arc motion primitives (arc radii
              {4, 8, 20} m, one-bin heading sweep, forward/reverse; straights
              0.5 m fwd/rev). Reverse costs x2. Heuristic = max(Euclidean,
              obstacle-aware 2D Dijkstra distance x cos 22.5deg) - goal
              tolerance (admissible).
  costmap   — per occupancy grid: footprint layer = z-band collapse over the
              d2 vehicle band [z_lo+0.1, z_lo+1.5] (identical to
              downstream_metrics._collision_verdicts); ESDF = scipy 2D
              distance transform x 0.10 m; lethal iff ESDF <= half-width
              0.9 m + safety 0.1 m = 1.0 m.
  footprint — vehicle 4.5 x 1.8 m; planning-time check = medial spine sampled
              every 0.05 m against the inflated (lethal) grid. The swept
              stadium (radius 1.0 m) covers the exact rectangle plus a 0.1 m
              lateral margin; the ~0.56 m longitudinal overhang per end is
              structural conservatism, identical across all cells.
  GT check  — the PLANNED path is swept with the EXACT 4.5 x 1.8 m rectangle
              (0.05 m pose/footprint sampling, d2 semantics) against the GT
              footprint layer -> collisions-per-100-plans.

Grids are REUSED from /data/peilincai/gems_stage1/analysis/r3a_occupancy_routes
(<cell>/grids_and_per_sample.npz; occ_route_i / occ_route_ii). GT occupancy is
not stored there; it is rebuilt via the identical downstream_metrics code path
and VERIFIED bit-exact against the stored per-voxel indicator arrays and the
stored seed-0 d2 GT verdicts before any planning (mismatch = abort).

Pre-registered predictions (LEDGER GOAL #R-03, frozen before any plan):
  P1 (preservation): B50 route-(i) grids produce <= the collisions-per-100-
     plans of clean route-(i) grids on both scenes.
  P2: route-(ii) grids produce MORE GT-collisions-per-100-plans than
     route-(i) grids per (scene x model).
Conservatism (path-length inflation, spurious infeasibility) reported for
both routes either way.

Usage:
    python tools/gems/planner_loop.py                 # full study (CPU only)
    python tools/gems/planner_loop.py --selftest      # numpy/scipy-only tests

Durable outputs: /data/peilincai/gems_stage1/analysis/r3c_planner/
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.gems.downstream_metrics import (          # noqa: E402
    VEHICLE_LENGTH_M,
    VEHICLE_WIDTH_M,
    VEHICLE_Z_BAND_M,
    _VoxelGrid,
    _build_gt_occupancy,
    _collision_verdicts,
    _sample_trajectories,
)
from tools.gems.occupancy_routes import build_gt_arg  # noqa: E402
from tools.gems.paired_bootstrap import paired_bootstrap_ci  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen study constants (pre-registered in LEDGER GOAL #R-03)
# ---------------------------------------------------------------------------
VOXEL_M = 0.10
N_THETA = 16
DTH = 2.0 * math.pi / N_THETA          # one heading bin = 22.5 deg

STRAIGHT_LEN_M = 0.5
ARC_RADII_M = (4.0, 8.0, 20.0)
REVERSE_COST_MULT = 2.0

HALF_WIDTH_M = VEHICLE_WIDTH_M / 2.0   # 0.9
SAFETY_M = 0.1
INFLATE_R_M = HALF_WIDTH_M + SAFETY_M  # 1.0 — costmap inflation radius
# Spine half-length so the radius-INFLATE_R stadium covers the exact vehicle
# rectangle: sqrt((L/2 - h)^2 + (W/2)^2) <= R  =>  h = L/2 - sqrt(R^2-(W/2)^2)
SPINE_HALF_M = VEHICLE_LENGTH_M / 2.0 - math.sqrt(
    INFLATE_R_M ** 2 - HALF_WIDTH_M ** 2)          # 1.81411 m
SAMPLE_STEP_M = 0.05                   # pose stepping + footprint sampling

GOAL_XY_TOL_M = 0.30
GOAL_TH_TOL_BINS = 1
MAX_EXPANSIONS = 500_000
MAX_SECONDS = 90.0
COS_22P5 = math.cos(math.pi / 8.0)     # 8-connected over-estimation correction

N_PROBLEMS = 100
SEED = 0
SEPARATION_RANGE_M = (5.0, 25.0)

N_TRAJ_D2 = 200                        # for the GT-grid bit-exact verification

R3A_ROOT = "/data/peilincai/gems_stage1/analysis/r3a_occupancy_routes"
OUT_ROOT_DEFAULT = "/data/peilincai/gems_stage1/analysis/r3c_planner"

# (scene, model label) cells with saved R3.a grids.
CELLS = [
    ("toy_parking", "clean30k"),
    ("toy_parking", "B50_importance_ft_e1v2_40000"),
    ("toy_parking", "B25_importance_ft_e1v2_40000"),
    ("courtyard", "clean30k"),
    ("courtyard", "B50_importance_ft_e1v2_40000"),
]
ROUTES = ("i", "ii")
GTREF_LABEL = "GTREF"

# Okabe-Ito (colorblind-safe) panel colors.
C_LETHAL = "#d9d9d9"     # inflated margin (light gray)
C_OCC = "#4d4d4d"        # model-grid obstacles (dark gray)
C_GT = "#0072B2"         # GT obstacles overlay (blue)
C_SWEEP = "#E69F00"      # swept footprint (orange)
C_COLL = "#D55E00"       # swept cells hitting GT (vermillion)
C_START = "#009E73"      # start marker (green)
C_GOAL = "#CC79A7"       # goal marker (purple)


# ---------------------------------------------------------------------------
# Motion primitives (grid-independent; anchored at a cell center, per bin)
# ---------------------------------------------------------------------------

def make_primitives():
    """14 primitives: straight fwd/rev + arcs {4,8,20} m x {L,R} x {fwd,rev}.
    Arcs sweep exactly one heading bin. Returns list of dicts."""
    prims = []
    for d in (1, -1):
        prims.append({"kind": "straight", "dir": d, "radius": None, "turn": 0,
                      "length": STRAIGHT_LEN_M, "dbin": 0,
                      "cost": STRAIGHT_LEN_M * (1.0 if d > 0 else REVERSE_COST_MULT)})
    for r in ARC_RADII_M:
        for sg in (1, -1):
            for d in (1, -1):
                L = r * DTH
                prims.append({"kind": "arc", "dir": d, "radius": r, "turn": sg,
                              "length": L, "dbin": (sg * d) % N_THETA,
                              "cost": L * (1.0 if d > 0 else REVERSE_COST_MULT)})
    return prims


def primitive_poses(prim, theta0, step=SAMPLE_STEP_M):
    """Vehicle-center poses along the primitive starting at (0,0,theta0).
    Returns (cx [P], cy [P], th [P]) including both endpoints."""
    L, d = prim["length"], prim["dir"]
    n = max(1, int(math.ceil(L / step)))
    s = d * np.linspace(0.0, L, n + 1)          # signed arc length
    if prim["kind"] == "straight":
        cx = s * math.cos(theta0)
        cy = s * math.sin(theta0)
        th = np.full_like(s, theta0)
    else:
        r, sg = prim["radius"], prim["turn"]
        th = theta0 + sg * s / r
        cx = (r / sg) * (np.sin(th) - math.sin(theta0))
        cy = -(r / sg) * (np.cos(th) - math.cos(theta0))
    return cx, cy, th


def _cells_of_points(px, py):
    """Cell offsets (relative to the anchor cell) containing points relative
    to the anchor cell CENTER: di = floor(px/v + 0.5) (half-open cells)."""
    di = np.floor(px / VOXEL_M + 0.5).astype(np.int32)
    dj = np.floor(py / VOXEL_M + 0.5).astype(np.int32)
    return di, dj


def _unique_offsets(di, dj):
    key = di.astype(np.int64) * 1_000_003 + dj.astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    return np.stack([di[idx], dj[idx]], axis=1)     # [K, 2] int32


def spine_offsets(prim, theta0):
    """Swept SPINE sample cells (planning-time collision model)."""
    cx, cy, th = primitive_poses(prim, theta0)
    nu = int(math.ceil(2.0 * SPINE_HALF_M / SAMPLE_STEP_M)) + 1
    u = np.linspace(-SPINE_HALF_M, SPINE_HALF_M, nu)
    px = cx[:, None] + u[None, :] * np.cos(th)[:, None]
    py = cy[:, None] + u[None, :] * np.sin(th)[:, None]
    return _unique_offsets(*_cells_of_points(px.ravel(), py.ravel()))


def rect_offsets(prim, theta0):
    """Swept EXACT-rectangle cells (GT collision check, d2 semantics)."""
    cx, cy, th = primitive_poses(prim, theta0)
    nu = int(math.ceil(VEHICLE_LENGTH_M / SAMPLE_STEP_M)) + 1
    nv = int(math.ceil(VEHICLE_WIDTH_M / SAMPLE_STEP_M)) + 1
    u = np.linspace(-VEHICLE_LENGTH_M / 2.0, VEHICLE_LENGTH_M / 2.0, nu)
    v = np.linspace(-VEHICLE_WIDTH_M / 2.0, VEHICLE_WIDTH_M / 2.0, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel()[None, :], vv.ravel()[None, :]
    cth, sth = np.cos(th)[:, None], np.sin(th)[:, None]
    px = cx[:, None] + uu * cth - vv * sth
    py = cy[:, None] + uu * sth + vv * cth
    return _unique_offsets(*_cells_of_points(px.ravel(), py.ravel()))


def stationary_spine_offsets(theta0):
    nu = int(math.ceil(2.0 * SPINE_HALF_M / SAMPLE_STEP_M)) + 1
    u = np.linspace(-SPINE_HALF_M, SPINE_HALF_M, nu)
    return _unique_offsets(*_cells_of_points(u * math.cos(theta0),
                                             u * math.sin(theta0)))


class PrimitiveTable:
    """All grid-independent precomputation: per (theta bin, primitive):
    endpoint cell displacement + new bin + cost/length, swept spine offsets
    (planning) and swept rectangle offsets (GT check); per bin: stationary
    spine offsets (pose validity)."""

    def __init__(self):
        self.prims = make_primitives()
        self.n_prims = len(self.prims)
        self.end = np.zeros((N_THETA, self.n_prims, 3), dtype=np.int32)  # di,dj,nt
        self.cost = np.zeros(self.n_prims, dtype=np.float64)
        self.length = np.zeros(self.n_prims, dtype=np.float64)
        self.spine = [[None] * self.n_prims for _ in range(N_THETA)]
        self.rect = [[None] * self.n_prims for _ in range(N_THETA)]
        self.stat_spine = [None] * N_THETA
        for p, prim in enumerate(self.prims):
            self.cost[p] = prim["cost"]
            self.length[p] = prim["length"]
        for t in range(N_THETA):
            th = t * DTH
            self.stat_spine[t] = stationary_spine_offsets(th)
            for p, prim in enumerate(self.prims):
                cx, cy, _ = primitive_poses(prim, th)
                di = int(math.floor(cx[-1] / VOXEL_M + 0.5))
                dj = int(math.floor(cy[-1] / VOXEL_M + 0.5))
                nt = (t + prim["dbin"]) % N_THETA
                self.end[t, p] = (di, dj, nt)
                self.spine[t][p] = spine_offsets(prim, th)
                self.rect[t][p] = rect_offsets(prim, th)
        # Plain-python copies for the A* inner loop (numpy scalar overhead).
        self.end_list = [[tuple(int(v) for v in self.end[t, p])
                          for p in range(self.n_prims)] for t in range(N_THETA)]
        self.cost_list = [float(c) for c in self.cost]


# ---------------------------------------------------------------------------
# Per-grid maps (ESDF, inflation, swept-blocked maps, heuristic graph)
# ---------------------------------------------------------------------------

def footprint_layer(occ3d, grid):
    """d2 vehicle-band collapse — identical semantics to _collision_verdicts."""
    k0, k1 = grid.z_slice(grid.z_lo + VEHICLE_Z_BAND_M[0],
                          grid.z_lo + VEHICLE_Z_BAND_M[1])
    if k1 <= k0:
        return np.zeros(tuple(grid.shape[:2]), dtype=bool)
    return occ3d[:, :, k0:k1].any(axis=2)


def _blocked_map(lethal_f32, offsets, nx, ny):
    """B[i,j] = OR over offsets o of lethal[i+o]; outside grid = free.
    Computed by FFT cross-correlation (kernel = offset indicator)."""
    oi, oj = offsets[:, 0], offsets[:, 1]
    oi_min, oi_max = int(oi.min()), int(oi.max())
    oj_min, oj_max = int(oj.min()), int(oj.max())
    K = np.zeros((oi_max - oi_min + 1, oj_max - oj_min + 1), dtype=np.float32)
    K[oi - oi_min, oj - oj_min] = 1.0
    C = fftconvolve(lethal_f32, K[::-1, ::-1], mode="full")
    return C[oi_max: oi_max + nx, oj_max: oj_max + ny] > 0.5


class GridMaps:
    """Everything the planner needs for ONE occupancy grid."""

    def __init__(self, band2d, ptab):
        t0 = time.time()
        self.band2d = band2d
        self.nx, self.ny = band2d.shape
        if band2d.any():
            self.esdf = distance_transform_edt(~band2d) * VOXEL_M
        else:  # no obstacles: EDT of an all-foreground image is ill-defined
            self.esdf = np.full(band2d.shape, np.inf)
        self.lethal = self.esdf <= INFLATE_R_M + 1e-12
        lf = self.lethal.astype(np.float32)
        # Pose-validity (stationary spine) per bin, flattened for fast lookup.
        self.stat_block = [
            _blocked_map(lf, ptab.stat_spine[t], self.nx, self.ny).ravel()
            for t in range(N_THETA)]
        # Swept-blocked maps per (bin, primitive), flattened.
        self.swept_block = [
            [_blocked_map(lf, ptab.spine[t][p], self.nx, self.ny).ravel()
             for p in range(ptab.n_prims)]
            for t in range(N_THETA)]
        # Relaxed point-robot free space for the admissible heuristic:
        # valid vehicle-center cells always satisfy esdf > R - 0.1 (1-Lipschitz
        # ESDF, sample-to-center distance < 0.1) -> superset => admissible.
        self.relaxed_free = self.esdf > (INFLATE_R_M - VOXEL_M)
        self._graph = self._build_graph()
        self.precompute_sec = time.time() - t0

    def _build_graph(self):
        nx, ny = self.nx, self.ny
        free = self.relaxed_free
        idx = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
        rows, cols, wts = [], [], []
        shifts = [(1, 0, VOXEL_M), (0, 1, VOXEL_M),
                  (1, 1, VOXEL_M * math.sqrt(2.0)),
                  (1, -1, VOXEL_M * math.sqrt(2.0))]
        for dx, dy, w in shifts:
            sx = slice(max(0, -dx), nx - max(0, dx))
            sy = slice(max(0, -dy), ny - max(0, dy))
            tx = slice(max(0, dx), nx - max(0, -dx))
            ty = slice(max(0, dy), ny - max(0, -dy))
            m = free[sx, sy] & free[tx, ty]
            a, b = idx[sx, sy][m], idx[tx, ty][m]
            rows.append(a)
            cols.append(b)
            wts.append(np.full(a.shape[0], w))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        wts = np.concatenate(wts)
        return coo_matrix((wts, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()

    def dijkstra_from(self, goal_cells_flat):
        """Geodesic 8-connected distance maps (meters) from each goal cell.
        Returns [len(goal_cells), nx*ny] float64 (inf = unreachable)."""
        return csgraph_dijkstra(self._graph, directed=False,
                                indices=np.asarray(goal_cells_flat, dtype=np.int64))

    def pose_valid(self, ix, iy, t):
        return not self.stat_block[t][ix * self.ny + iy]


# ---------------------------------------------------------------------------
# A* search on the SE(2) lattice
# ---------------------------------------------------------------------------

def astar(maps, ptab, start, goal, dist_map,
          max_expansions=MAX_EXPANSIONS, max_seconds=MAX_SECONDS,
          cell_cost_mult=None):
    """A* from start=(ix,iy,t) to goal=(gx,gy,gt) on one grid.

    dist_map: flattened [nx*ny] geodesic distances (m) from the GOAL cell on
    maps.relaxed_free (heuristic component; inf = point-robot unreachable).

    Returns dict: found, reason, segments [(ix,iy,t,prim_id)], n_expansions,
    wall_s, path_length_m, n_switches.
    """
    nx, ny = maps.nx, maps.ny
    n_prims = ptab.n_prims
    sx, sy, st = start
    gx, gy, gt = goal
    t_begin = time.time()

    out = {"found": False, "reason": None, "segments": None,
           "n_expansions": 0, "wall_s": 0.0, "path_length_m": None,
           "n_switches": None}

    if not maps.pose_valid(sx, sy, st):
        out["reason"] = "start_invalid"
        out["wall_s"] = time.time() - t_begin
        return out
    if not maps.pose_valid(gx, gy, gt):
        out["reason"] = "goal_invalid"
        out["wall_s"] = time.time() - t_begin
        return out
    if not np.isfinite(dist_map[sx * ny + sy]):
        out["reason"] = "disconnected"
        out["wall_s"] = time.time() - t_begin
        return out

    ns = nx * ny * N_THETA
    g = np.full(ns, np.inf, dtype=np.float64)
    parent = np.full(ns, -1, dtype=np.int64)
    parent_prim = np.full(ns, -1, dtype=np.int16)

    end = ptab.end_list
    cost = ptab.cost_list
    swept = maps.swept_block
    v = VOXEL_M
    tol2_cells = (GOAL_XY_TOL_M / v) ** 2

    def h(cix, ciy):
        dx, dy = (cix - gx) * v, (ciy - gy) * v
        eu = math.hypot(dx, dy)
        d8 = dist_map[cix * ny + ciy]
        hh = max(eu, d8 * COS_22P5) - GOAL_XY_TOL_M
        return hh if hh > 0.0 else 0.0

    sid0 = (sx * ny + sy) * N_THETA + st
    g[sid0] = 0.0
    heap = [(h(sx, sy), sid0)]
    n_exp = 0
    goal_sid = -1

    while heap:
        f, sid = heapq.heappop(heap)
        gc = g[sid]
        cell = sid // N_THETA
        t = sid - cell * N_THETA
        ix = cell // ny
        iy = cell - ix * ny
        # stale heap entry?
        if f > gc + h(ix, iy) + 1e-9:
            continue
        # goal test
        ddx, ddy = ix - gx, iy - gy
        dt = abs(t - gt)
        if ddx * ddx + ddy * ddy <= tol2_cells + 1e-9 and \
                min(dt, N_THETA - dt) <= GOAL_TH_TOL_BINS:
            goal_sid = sid
            break
        n_exp += 1
        if n_exp >= max_expansions:
            out["reason"] = "budget_expansions"
            break
        if (n_exp & 1023) == 0 and time.time() - t_begin > max_seconds:
            out["reason"] = "budget_time"
            break
        swept_t = swept[t]
        end_t = end[t]
        for p in range(n_prims):
            if swept_t[p][cell]:
                continue
            di, dj, nt = end_t[p]
            nix = ix + di
            niy = iy + dj
            if nix < 0 or nix >= nx or niy < 0 or niy >= ny:
                continue
            # DS-1 hook (default None = exact legacy behavior): per-cell
            # traversal-cost multiplier >= 1 at the primitive END cell —
            # keeps h admissible (costs only inflate).
            if cell_cost_mult is None:
                ng = gc + cost[p]
            else:
                ng = gc + cost[p] * cell_cost_mult[nix * ny + niy]
            nsid = (nix * ny + niy) * N_THETA + nt
            if ng < g[nsid] - 1e-12:
                g[nsid] = ng
                parent[nsid] = sid
                parent_prim[nsid] = p
                heapq.heappush(heap, (ng + h(nix, niy), nsid))

    out["n_expansions"] = n_exp
    out["wall_s"] = time.time() - t_begin
    if goal_sid < 0:
        if out["reason"] is None:
            out["reason"] = "exhausted"   # provably infeasible on this lattice
        return out

    # Reconstruct path: segments anchored at their START state.
    segs = []
    sid = goal_sid
    while parent[sid] >= 0:
        psid = int(parent[sid])
        p = int(parent_prim[sid])
        pcell = psid // N_THETA
        pt = psid - pcell * N_THETA
        pix = pcell // ny
        piy = pcell - pix * ny
        segs.append((pix, piy, pt, p))
        sid = psid
    segs.reverse()
    dirs = [ptab.prims[p]["dir"] for (_, _, _, p) in segs]
    out.update({
        "found": True, "reason": "ok", "segments": segs,
        "path_length_m": float(sum(ptab.length[p] for (_, _, _, p) in segs)),
        "n_switches": int(sum(1 for a, b in zip(dirs[:-1], dirs[1:]) if a != b)),
    })
    return out


def gt_sweep_collision(segments, start, ptab, gt_band2d, collect_mask=False):
    """Sweep the EXACT vehicle rectangle along the planned path against the GT
    footprint layer. Empty plans (start within goal tolerance) check the
    stationary rectangle at the start pose."""
    nx, ny = gt_band2d.shape
    mask = np.zeros_like(gt_band2d) if collect_mask else None
    hit = np.zeros_like(gt_band2d)
    if segments:
        items_offs = [((ix, iy), ptab.rect[t][p])
                      for (ix, iy, t, p) in segments]
    else:  # degenerate plan: stationary rectangle at the start pose
        sx, sy, st = start
        offs = _unique_offsets(*_cells_of_points(*_stationary_rect_points(st)))
        items_offs = [((sx, sy), offs)]
    collided = False
    for (ix, iy), offs in items_offs:
        ci = offs[:, 0] + ix
        cj = offs[:, 1] + iy
        ok = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
        ci, cj = ci[ok], cj[ok]
        if collect_mask:
            mask[ci, cj] = True
        gvals = gt_band2d[ci, cj]
        if gvals.any():
            collided = True
            hit[ci[gvals], cj[gvals]] = True
    n_coll_cells = int(hit.sum())
    return {"collision": bool(collided), "n_collision_cells": n_coll_cells,
            "swept_mask": mask, "hit_mask": hit if collect_mask else None}


def _stationary_rect_points(tbin):
    th = tbin * DTH
    nu = int(math.ceil(VEHICLE_LENGTH_M / SAMPLE_STEP_M)) + 1
    nv = int(math.ceil(VEHICLE_WIDTH_M / SAMPLE_STEP_M)) + 1
    u = np.linspace(-VEHICLE_LENGTH_M / 2.0, VEHICLE_LENGTH_M / 2.0, nu)
    w = np.linspace(-VEHICLE_WIDTH_M / 2.0, VEHICLE_WIDTH_M / 2.0, nv)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    px = uu.ravel() * math.cos(th) - ww.ravel() * math.sin(th)
    py = uu.ravel() * math.sin(th) + ww.ravel() * math.cos(th)
    return px, py


# ---------------------------------------------------------------------------
# GT rebuild + bit-exact verification against the saved R3.a arrays
# ---------------------------------------------------------------------------

def load_r3a_cell(scene, label):
    path = os.path.join(R3A_ROOT, f"{scene}__{label}", "grids_and_per_sample.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}, path


def verify_gt_grid(scene, grid, gt_occ, cells_npz):
    """GT grid is rebuilt (not stored in R3.a npz). Verify bit-exact:
    the stored per-voxel indicators of EVERY cell must be reproduced from
    (rebuilt GT, stored route grids), and the stored seed-0 d2 GT verdicts
    must equal _collision_verdicts on the rebuilt GT grid."""
    gt_flat = gt_occ.ravel()
    checks = {}
    for label, arrs in cells_npz.items():
        for route in ROUTES:
            occ = arrs[f"occ_route_{route}"]
            ff = ~occ.ravel()[gt_flat]
            fo = occ.ravel()[~gt_flat]
            ok = (np.array_equal(ff, arrs[f"free_at_gt_occ_route_{route}"]) and
                  np.array_equal(fo, arrs[f"occ_at_gt_free_route_{route}"]))
            checks[f"{label}/route_{route}"] = bool(ok)
    # d2 GT verdicts (seed 0, 200 trajectories — unchanged machinery)
    from tools.gems.scenes import SCENES
    roi = SCENES[scene].roi
    trajs = _sample_trajectories(np.random.default_rng(SEED), roi, N_TRAJ_D2)
    v_gt = _collision_verdicts(trajs, grid, gt_occ)
    any_arrs = next(iter(cells_npz.values()))
    checks["d2_verdicts_gt"] = bool(np.array_equal(v_gt, any_arrs["d2_verdicts_gt"]))
    all_ok = all(checks.values())
    return all_ok, checks


# ---------------------------------------------------------------------------
# Problem sampling (seed 0, GT-grid free space, paired across grids)
# ---------------------------------------------------------------------------

def sample_problems(maps_gt, rng, n_problems=N_PROBLEMS):
    """Start/goal lattice poses: valid on the GT inflated costmap, separation
    within SEPARATION_RANGE_M, point-robot-connected on the GT relaxed map."""
    nx, ny = maps_gt.nx, maps_gt.ny
    lo, hi = SEPARATION_RANGE_M
    problems = []
    attempts = 0
    while len(problems) < n_problems and attempts < 400:
        attempts += 1
        k = 512
        sx = rng.integers(0, nx, k)
        sy = rng.integers(0, ny, k)
        st = rng.integers(0, N_THETA, k)
        gx = rng.integers(0, nx, k)
        gy = rng.integers(0, ny, k)
        gt = rng.integers(0, N_THETA, k)
        sep = np.hypot((sx - gx) * VOXEL_M, (sy - gy) * VOXEL_M)
        cand = []
        for i in range(k):
            if not (lo <= sep[i] <= hi):
                continue
            if maps_gt.stat_block[st[i]][sx[i] * ny + sy[i]]:
                continue
            if maps_gt.stat_block[gt[i]][gx[i] * ny + gy[i]]:
                continue
            cand.append((int(sx[i]), int(sy[i]), int(st[i]),
                         int(gx[i]), int(gy[i]), int(gt[i])))
        if not cand:
            continue
        goal_cells = [c[3] * ny + c[4] for c in cand]
        dmat = maps_gt.dijkstra_from(goal_cells)
        for row, c in enumerate(cand):
            if len(problems) >= n_problems:
                break
            if np.isfinite(dmat[row, c[0] * ny + c[1]]):
                problems.append({"start": c[:3], "goal": c[3:]})
    if len(problems) < n_problems:
        raise RuntimeError(
            f"problem sampler found only {len(problems)}/{n_problems} "
            f"GT-feasible problems after {attempts} batches")
    return problems


# ---------------------------------------------------------------------------
# Panels (matplotlib top-down; Okabe-Ito colorblind-safe layers)
# ---------------------------------------------------------------------------

def _hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


def draw_panel(png_path, title, grid, maps_cell, gt_band2d, problem, result,
               sweep, ptab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    nx, ny = maps_cell.nx, maps_cell.ny
    img = np.ones((nx, ny, 3))
    for mask, color, alpha in [
            (maps_cell.lethal, _hex_rgb(C_LETHAL), 1.0),
            (maps_cell.band2d, _hex_rgb(C_OCC), 1.0),
            (gt_band2d, _hex_rgb(C_GT), 0.55)]:
        img[mask] = (1 - alpha) * img[mask] + alpha * color
    if sweep is not None and sweep.get("swept_mask") is not None:
        m = sweep["swept_mask"]
        img[m] = 0.55 * img[m] + 0.45 * _hex_rgb(C_SWEEP)
        hm = sweep.get("hit_mask")
        if hm is not None and hm.any():
            img[hm] = _hex_rgb(C_COLL)

    x0 = grid.origin[0]
    y0 = grid.origin[1]
    extent = [x0, x0 + nx * VOXEL_M, y0, y0 + ny * VOXEL_M]

    fig, ax = plt.subplots(figsize=(9, 9 * ny / nx))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=extent,
              interpolation="nearest")

    def pose_xy(p):
        return (x0 + (p[0] + 0.5) * VOXEL_M, y0 + (p[1] + 0.5) * VOXEL_M,
                p[2] * DTH)

    sxm, sym, sth = pose_xy(problem["start"])
    gxm, gym, gth = pose_xy(problem["goal"])
    if result is not None and result.get("segments"):
        pts = []
        for (ix, iy, t, p) in result["segments"]:
            cx, cy, _ = primitive_poses(ptab.prims[p], t * DTH)
            ax_x = x0 + (ix + 0.5) * VOXEL_M + cx
            ax_y = y0 + (iy + 0.5) * VOXEL_M + cy
            pts.append(np.stack([ax_x, ax_y], axis=1))
        poly = np.concatenate(pts, axis=0)
        ax.plot(poly[:, 0], poly[:, 1], color="black", lw=1.6, zorder=5)
    for (xm, ym, th, col) in [(sxm, sym, sth, C_START), (gxm, gym, gth, C_GOAL)]:
        ax.plot([xm], [ym], "o", color=col, ms=8, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.annotate("", xy=(xm + 1.4 * math.cos(th), ym + 1.4 * math.sin(th)),
                    xytext=(xm, ym), zorder=6,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
    legend = [
        Patch(facecolor=C_OCC, label="model grid (footprint layer)"),
        Patch(facecolor=C_LETHAL, label="inflated lethal (ESDF <= 1.0 m)"),
        Patch(facecolor=C_GT, alpha=0.55, label="GT footprint layer"),
        Patch(facecolor=C_SWEEP, alpha=0.6, label="planned footprint sweep"),
        Patch(facecolor=C_COLL, label="sweep cells hitting GT"),
        Line2D([0], [0], color="black", lw=1.6, label="planned centerline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_START,
               markeredgecolor="white", ms=8, label="start pose"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GOAL,
               markeredgecolor="white", ms=8, label="goal pose"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def plan_cell(cell_key, maps, maps_gt_band, problems, ptab, dist_maps,
              cell_cost_mult=None):
    """Plan all problems on one grid; GT-sweep each found plan."""
    per_problem = []
    for k, prob in enumerate(problems):
        res = astar(maps, ptab, prob["start"], prob["goal"], dist_maps[k],
                    cell_cost_mult=cell_cost_mult)
        rec = {"problem": k, "found": res["found"], "reason": res["reason"],
               "n_expansions": res["n_expansions"],
               "wall_s": round(res["wall_s"], 4),
               "path_length_m": res["path_length_m"],
               "n_segments": (len(res["segments"]) if res["segments"] else None),
               "n_switches": res["n_switches"]}
        if res["found"]:
            sw = gt_sweep_collision(res["segments"], prob["start"], ptab,
                                    maps_gt_band, collect_mask=False)
            rec["gt_collision"] = sw["collision"]
            rec["n_collision_cells"] = sw["n_collision_cells"]
            rec["_segments"] = res["segments"]
        else:
            rec["gt_collision"] = None
        per_problem.append(rec)
    n = len(per_problem)
    found = [r for r in per_problem if r["found"]]
    coll = [r for r in found if r["gt_collision"]]
    reasons = {}
    for r in per_problem:
        if not r["found"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    metrics = {
        "n_problems": n,
        "plans_found": len(found),
        "found_rate": len(found) / n,
        "spurious_infeasibility_rate": 1.0 - len(found) / n,
        "infeasibility_reasons": reasons,
        "n_gt_collisions": len(coll),
        "collisions_per_100_plans": (100.0 * len(coll) / len(found)
                                     if found else None),
        "mean_path_length_m": (float(np.mean([r["path_length_m"] for r in found]))
                               if found else None),
        "mean_wall_s_per_plan": float(np.mean([r["wall_s"] for r in per_problem])),
        "median_wall_s_per_plan": float(np.median([r["wall_s"] for r in per_problem])),
        "mean_expansions": float(np.mean([r["n_expansions"] for r in per_problem])),
        "grid_precompute_sec": maps.precompute_sec,
        "band_occupied_fraction": float(maps.band2d.mean()),
        "lethal_fraction": float(maps.lethal.mean()),
    }
    return metrics, per_problem


def path_length_inflation(per_problem, per_problem_ref):
    both = [(a["path_length_m"], b["path_length_m"])
            for a, b in zip(per_problem, per_problem_ref)
            if a["found"] and b["found"] and b["path_length_m"] and
            b["path_length_m"] > 0]
    if not both:
        return {"n_common_found": 0}
    infl = np.array([(a - b) / b for a, b in both])
    return {"n_common_found": len(both),
            "mean_inflation": float(infl.mean()),
            "median_inflation": float(np.median(infl)),
            "p90_inflation": float(np.percentile(infl, 90))}


def paired_collision_ci(per_a, per_b):
    """Paired bootstrap on GT-collision indicators over commonly-found
    problems: mean_diff = P(coll | a) - P(coll | b)."""
    a, b = [], []
    for ra, rb in zip(per_a, per_b):
        if ra["found"] and rb["found"]:
            a.append(1.0 if ra["gt_collision"] else 0.0)
            b.append(1.0 if rb["gt_collision"] else 0.0)
    if len(a) < 2:
        return {"n_common_found": len(a), "unavailable": True}
    ci = paired_bootstrap_ci(np.array(a), np.array(b))
    ci["n_common_found"] = len(a)
    ci["rate_a"] = float(np.mean(a))
    ci["rate_b"] = float(np.mean(b))
    return ci


def paired_found_ci(per_a, per_b):
    a = np.array([1.0 if r["found"] else 0.0 for r in per_a])
    b = np.array([1.0 if r["found"] else 0.0 for r in per_b])
    ci = paired_bootstrap_ci(a, b)
    ci["rate_a"] = float(a.mean())
    ci["rate_b"] = float(b.mean())
    return ci


def run_study(out_root):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "panels"), exist_ok=True)
    t_study = time.time()

    print("[r3c] building primitive table (grid-independent) ...")
    t0 = time.time()
    ptab = PrimitiveTable()
    print(f"[r3c] primitive table: {ptab.n_prims} prims x {N_THETA} bins "
          f"({time.time() - t0:.1f}s)")

    from tools.gems.scenes import SCENES

    summary = {
        "goal": "LEDGER GOAL #R-03 (Stage-1R R3.c planner closed loop v0)",
        "pre_registered_predictions": {
            "P1": ("preservation: B50 route-(i) collisions-per-100-plans <= "
                   "clean route-(i) on both scenes (violated only if B50-clean"
                   " > 0 with paired CI excl. 0)"),
            "P2": ("route-(ii) grids produce MORE GT-collisions-per-100-plans "
                   "than route-(i) per (scene x model)"),
        },
        "constants": {
            "voxel_m": VOXEL_M, "n_theta": N_THETA,
            "straight_len_m": STRAIGHT_LEN_M, "arc_radii_m": ARC_RADII_M,
            "reverse_cost_mult": REVERSE_COST_MULT,
            "inflate_r_m": INFLATE_R_M, "spine_half_m": SPINE_HALF_M,
            "sample_step_m": SAMPLE_STEP_M, "goal_xy_tol_m": GOAL_XY_TOL_M,
            "goal_th_tol_bins": GOAL_TH_TOL_BINS,
            "max_expansions": MAX_EXPANSIONS, "max_seconds": MAX_SECONDS,
            "n_problems": N_PROBLEMS, "seed": SEED,
            "separation_range_m": SEPARATION_RANGE_M,
            "vehicle_m": [VEHICLE_LENGTH_M, VEHICLE_WIDTH_M],
        },
        "gt_verification": {}, "scenes": {}, "cells": {}, "comparisons": {},
        "caveats": [],
    }

    scene_names = sorted({s for s, _ in CELLS},
                         key=lambda s: [c[0] for c in CELLS].index(s))
    panel_examples = {}

    for scene in scene_names:
        labels = [l for s, l in CELLS if s == scene]
        spec = SCENES[scene]
        grid = _VoxelGrid(spec.roi, VOXEL_M)

        print(f"[r3c] === scene {scene}: loading R3.a grids ===")
        cells_npz = {}
        for label in labels:
            arrs, path = load_r3a_cell(scene, label)
            cells_npz[label] = arrs
            print(f"[r3c]   loaded {path}")

        print(f"[r3c] rebuilding GT occupancy ({scene}) via downstream_metrics ...")
        gt_occ = _build_gt_occupancy(grid, build_gt_arg(spec))
        ok, checks = verify_gt_grid(scene, grid, gt_occ, cells_npz)
        summary["gt_verification"][scene] = {"all_bit_exact": ok, **checks}
        if not ok:
            raise RuntimeError(f"GT grid verification FAILED for {scene}: {checks}")
        print(f"[r3c]   GT verification bit-exact: {ok} "
              f"({len(checks)} checks)")

        gt_band = footprint_layer(gt_occ, grid)
        maps_gt = GridMaps(gt_band, ptab)
        print(f"[r3c]   GT maps: {maps_gt.nx}x{maps_gt.ny}, lethal "
              f"{maps_gt.lethal.mean():.1%}, precompute "
              f"{maps_gt.precompute_sec:.1f}s")

        rng = np.random.default_rng(SEED)
        problems = sample_problems(maps_gt, rng)
        seps = [math.hypot((p["start"][0] - p["goal"][0]) * VOXEL_M,
                           (p["start"][1] - p["goal"][1]) * VOXEL_M)
                for p in problems]
        summary["scenes"][scene] = {
            "grid_shape": [maps_gt.nx, maps_gt.ny],
            "gt_lethal_fraction": float(maps_gt.lethal.mean()),
            "gt_occupied_fraction": float(gt_band.mean()),
            "n_problems": len(problems),
            "mean_separation_m": float(np.mean(seps)),
        }
        print(f"[r3c]   sampled {len(problems)} paired problems "
              f"(mean separation {np.mean(seps):.1f} m)")

        # Grids to plan on: GT reference + each (model x route).
        grids_to_plan = [(GTREF_LABEL, None, maps_gt)]
        for label in labels:
            for route in ROUTES:
                band = footprint_layer(cells_npz[label][f"occ_route_{route}"],
                                       grid)
                grids_to_plan.append((label, route, GridMaps(band, ptab)))

        per_cell_records = {}
        for label, route, maps in grids_to_plan:
            key = (f"{scene}__{GTREF_LABEL}" if route is None
                   else f"{scene}__{label}__route_{route}")
            t0 = time.time()
            goal_cells = [p["goal"][0] * maps.ny + p["goal"][1]
                          for p in problems]
            dist_maps = maps.dijkstra_from(goal_cells)
            metrics, per_problem = plan_cell(key, maps, gt_band, problems,
                                             ptab, dist_maps)
            metrics["cell_overhead_sec"] = round(
                time.time() - t0 - sum(r["wall_s"] for r in per_problem), 1)
            per_cell_records[key] = (metrics, per_problem, maps)
            print(f"[r3c]   {key}: found {metrics['plans_found']}/"
                  f"{metrics['n_problems']}, GT-collisions "
                  f"{metrics['n_gt_collisions']} "
                  f"({metrics['collisions_per_100_plans']}) per-100, "
                  f"mean len {metrics['mean_path_length_m']}, "
                  f"{metrics['mean_wall_s_per_plan']:.2f}s/plan")

        # Inflation vs GT reference + write per-cell json.
        ref_key = f"{scene}__{GTREF_LABEL}"
        _, per_ref, _ = per_cell_records[ref_key]
        for key, (metrics, per_problem, maps) in per_cell_records.items():
            if key != ref_key:
                metrics["path_length_inflation_vs_gtref"] = \
                    path_length_inflation(per_problem, per_ref)
            cdir = os.path.join(out_root, key)
            os.makedirs(cdir, exist_ok=True)
            clean_pp = [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in per_problem]
            with open(os.path.join(cdir, "cell_metrics.json"), "w") as f:
                json.dump(_json_safe({"cell": key, "metrics": metrics,
                                      "per_problem": clean_pp}), f, indent=1)
            summary["cells"][key] = _json_safe(metrics)

        # ---- pre-registered comparisons ----
        comp = {}
        for label in labels:
            ki = f"{scene}__{label}__route_i"
            kii = f"{scene}__{label}__route_ii"
            comp[f"P2_route_ii_minus_i__{label}"] = _json_safe(
                paired_collision_ci(per_cell_records[kii][1],
                                    per_cell_records[ki][1]))
        clean_label = "clean30k"
        for label in labels:
            if label.startswith("B50") or label.startswith("B25"):
                ka = f"{scene}__{label}__route_i"
                kb = f"{scene}__{clean_label}__route_i"
                comp[f"P1_{label}_minus_clean__route_i"] = _json_safe(
                    paired_collision_ci(per_cell_records[ka][1],
                                        per_cell_records[kb][1]))
                comp[f"found_{label}_minus_clean__route_i"] = _json_safe(
                    paired_found_ci(per_cell_records[ka][1],
                                    per_cell_records[kb][1]))
        for label in labels:
            comp[f"found_route_ii_minus_i__{label}"] = _json_safe(
                paired_found_ci(per_cell_records[f"{scene}__{label}__route_ii"][1],
                                per_cell_records[f"{scene}__{label}__route_i"][1]))
        summary["comparisons"][scene] = comp

        # ---- panels: success / GT-collision-if-any / infeasible-if-any ----
        pref = [f"{scene}__{clean_label}__route_i"] + \
               [f"{scene}__{l}__route_i" for l in labels] + \
               [f"{scene}__{l}__route_ii" for l in labels]
        pref_coll = [f"{scene}__{l}__route_ii" for l in labels] + \
                    [f"{scene}__{l}__route_i" for l in labels] + \
                    [ref_key]  # GTREF fallback: shows the planner's own floor

        def find_example(keys, pred):
            for key in keys:
                if key not in per_cell_records:
                    continue
                metrics, per_problem, maps = per_cell_records[key]
                for r in per_problem:
                    if pred(r):
                        return key, r, maps
            return None

        examples = {
            "success": find_example(
                pref, lambda r: r["found"] and not r["gt_collision"]),
            "gt_collision": find_example(
                pref_coll, lambda r: r["found"] and r["gt_collision"]),
            "infeasible": find_example(
                pref, lambda r: not r["found"]),
        }
        panel_paths = {}
        for kind, ex in examples.items():
            if ex is None:
                panel_paths[kind] = None
                continue
            key, r, maps = ex
            prob = problems[r["problem"]]
            sweep = None
            if r["found"]:
                sweep = gt_sweep_collision(r["_segments"], prob["start"], ptab,
                                           gt_band, collect_mask=True)
            plen = (f"{r['path_length_m']:.1f} m" if r["path_length_m"]
                    is not None else "n/a")
            title = (f"{key} | problem {r['problem']} | {kind.upper()} | "
                     f"reason={r['reason']} len={plen}")
            png = os.path.join(out_root, "panels", f"{scene}__{kind}.png")
            draw_panel(png, title, grid, maps, gt_band, prob, r, sweep, ptab)
            panel_paths[kind] = png
            print(f"[r3c]   panel {kind}: {png} ({key} problem {r['problem']})")
        panel_examples[scene] = panel_paths

    summary["panels"] = panel_examples

    # ---- verdicts vs pre-registered predictions ----
    p1_checks, p2_checks = {}, {}
    for scene in scene_names:
        comp = summary["comparisons"][scene]
        for name, c in comp.items():
            if name.startswith("P1_"):
                # violated only if B50/B25-minus-clean > 0 with CI excl. 0
                p1_checks[f"{scene}:{name}"] = (
                    (not (c["mean_diff"] > 0 and c["ci_lo"] > 0))
                    if "mean_diff" in c else
                    f"unevaluable (n_common_found={c.get('n_common_found')})")
            if name.startswith("P2_"):
                p2_checks[f"{scene}:{name}"] = (
                    bool(c["mean_diff"] > 0) if "mean_diff" in c else
                    f"unevaluable (n_common_found={c.get('n_common_found')})")
    p1_eval = [v for k, v in p1_checks.items()
               if "B50" in k and isinstance(v, bool)]
    p2_eval = [v for v in p2_checks.values() if isinstance(v, bool)]
    summary["verdict"] = {
        "P1_preservation_per_comparison_ok": p1_checks,
        "P1_PASS": (all(p1_eval) if p1_eval else None),
        "P1_n_unevaluable": sum(1 for k, v in p1_checks.items()
                                if "B50" in k and not isinstance(v, bool)),
        "P2_route_ii_more_collisions_per_cell": p2_checks,
        "P2_PASS": (all(p2_eval) if p2_eval else None),
        "P2_n_unevaluable": sum(1 for v in p2_checks.values()
                                if not isinstance(v, bool)),
        "note": ("P1 bar: no B50-vs-clean route-(i) collision increase with "
                 "paired CI excl. 0 on both scenes (B25 reported, not part of "
                 "the bar); P2 bar: route-(ii) collision rate > route-(i) on "
                 "every (scene x model) cell, paired CIs reported; a "
                 "comparison is unevaluable when fewer than 2 problems are "
                 "solved by BOTH arms (collision rates are conditional on a "
                 "plan existing)"),
    }
    summary["caveats"] = [
        "lattice snap error <= 0.071 m per primitive step; absorbed by the "
        "0.1 m safety inflation along the vehicle SIDES; the GT sweep checks "
        "the exact snapped segments the planner validated",
        "stadium collision model overhangs the vehicle ~0.56 m longitudinally "
        "per end (structural conservatism, identical across all cells); the "
        "covering circle passes EXACTLY through the rectangle corners (zero "
        "corner margin), so lattice snap can clip corners: the GTREF row "
        "measures this planner-own collision floor",
        "ESDF and occupancy are cell-center quantities at 0.10 m: effective "
        "clearances are +-0.07 m of nominal",
        "out-of-ROI space is treated as free (same semantics as d2); "
        "problems/goals are sampled inside the ROI",
        "courtyard GT = laser-scan points; unscanned voxels count GT-free, so "
        "GT-collision counts are a lower bound there (affects all cells "
        "equally); courtyard z_band per LEDGER GOAL #008 derivation",
        "GT-reference plans use the same planner on the GT grid; its "
        "collisions-per-100-plans is the planner's own floor (expected ~0)",
        "problems are GT-feasible by construction (valid + point-robot-"
        "connected on the GT costmap), so every not-found on a model grid is "
        "spurious infeasibility relative to GT",
        "no retraining, no test views consumed anywhere: grids are R3.a "
        "train-evidence artifacts (D4)",
    ]
    summary["wallclock_sec_total"] = time.time() - t_study
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=1)
    print(f"[r3c] wrote {os.path.join(out_root, 'summary.json')} "
          f"({summary['wallclock_sec_total']:.0f}s total)")
    return summary


# ---------------------------------------------------------------------------
# Self-test (numpy/scipy only; no scene assets, no R3.a files)
# ---------------------------------------------------------------------------

def selftest():
    ptab = PrimitiveTable()

    # 1. arc endpoint analytic check: r=4, theta0=0, left, fwd ->
    #    (r sin 22.5, r (1-cos 22.5)), heading +1 bin.
    p_idx = next(i for i, pr in enumerate(ptab.prims)
                 if pr["kind"] == "arc" and pr["radius"] == 4.0
                 and pr["turn"] == 1 and pr["dir"] == 1)
    cx, cy, th = primitive_poses(ptab.prims[p_idx], 0.0)
    assert abs(cx[-1] - 4.0 * math.sin(DTH)) < 1e-9
    assert abs(cy[-1] - 4.0 * (1.0 - math.cos(DTH))) < 1e-9
    assert ptab.end[0, p_idx, 2] == 1
    # reverse-left arc endpoint mirrors behind, heading -1 bin
    p_rl = next(i for i, pr in enumerate(ptab.prims)
                if pr["kind"] == "arc" and pr["radius"] == 4.0
                and pr["turn"] == 1 and pr["dir"] == -1)
    cx2, cy2, _ = primitive_poses(ptab.prims[p_rl], 0.0)
    assert abs(cx2[-1] + 4.0 * math.sin(DTH)) < 1e-9
    assert abs(cy2[-1] - 4.0 * (1.0 - math.cos(DTH))) < 1e-9  # curves left behind
    assert ptab.end[0, p_rl, 2] == (N_THETA - 1)

    # 2. stadium coverage: every point of the exact vehicle rectangle is
    #    within INFLATE_R of the spine segment (corners exactly at R).
    u = np.linspace(-VEHICLE_LENGTH_M / 2, VEHICLE_LENGTH_M / 2, 181)
    w = np.linspace(-VEHICLE_WIDTH_M / 2, VEHICLE_WIDTH_M / 2, 73)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    du = np.clip(np.abs(uu) - SPINE_HALF_M, 0.0, None)
    dist = np.hypot(du, ww)
    assert dist.max() <= INFLATE_R_M + 1e-9, dist.max()
    corner = math.hypot(VEHICLE_LENGTH_M / 2 - SPINE_HALF_M, VEHICLE_WIDTH_M / 2)
    assert abs(corner - INFLATE_R_M) < 1e-9

    # 3. FFT blocked-map == brute force on random grids.
    rng = np.random.default_rng(1)
    for trial in range(3):
        nx, ny = 46, 39
        lethal = rng.random((nx, ny)) < 0.08
        t = int(rng.integers(0, N_THETA))
        p = int(rng.integers(0, ptab.n_prims))
        offs = ptab.spine[t][p]
        bf = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                ci = offs[:, 0] + i
                cj = offs[:, 1] + j
                ok = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
                bf[i, j] = lethal[ci[ok], cj[ok]].any()
        fftmap = _blocked_map(lethal.astype(np.float32), offs, nx, ny)
        assert np.array_equal(bf, fftmap), f"FFT map mismatch (trial {trial})"

    # 4. plan on an empty grid: straight-ahead problem along bin 0.
    nx = ny = 200  # 20 x 20 m
    empty = np.zeros((nx, ny), dtype=bool)
    maps = GridMaps(empty, ptab)
    start = (30, 100, 0)
    goal = (170, 100, 0)
    dist_map = maps.dijkstra_from([goal[0] * ny + goal[1]])[0]
    res = astar(maps, ptab, start, goal, dist_map)
    assert res["found"], res
    d_true = (170 - 30) * VOXEL_M
    assert d_true - GOAL_XY_TOL_M - 1e-6 <= res["path_length_m"] <= d_true + 1.0, res
    sw = gt_sweep_collision(res["segments"], start, ptab, empty)
    assert not sw["collision"]

    # 5. wall between start and goal -> disconnected; wall with a wide gap ->
    #    found, and the path must be longer than the straight line.
    wall = empty.copy()
    wall[100, :] = True
    maps_w = GridMaps(wall, ptab)
    dmap_w = maps_w.dijkstra_from([goal[0] * ny + goal[1]])[0]
    res_w = astar(maps_w, ptab, start, goal, dmap_w)
    assert not res_w["found"] and res_w["reason"] == "disconnected", res_w
    gap = wall.copy()
    gap[100, 130:190] = False   # 6 m gap, off-axis -> detour
    maps_g = GridMaps(gap, ptab)
    dmap_g = maps_g.dijkstra_from([goal[0] * ny + goal[1]])[0]
    res_g = astar(maps_g, ptab, start, goal, dmap_g)
    assert res_g["found"], res_g
    assert res_g["path_length_m"] > d_true + 1.0, res_g["path_length_m"]
    # planned path avoids its OWN grid but must collide with a DIFFERENT
    # "GT" wall that has no gap (this is exactly the false-free mechanism).
    sw_gt = gt_sweep_collision(res_g["segments"], start, ptab, wall)
    assert sw_gt["collision"] and sw_gt["n_collision_cells"] > 0

    # 6. plan on the gap grid but sweep against the gap grid itself: the
    #    planner's 0.1 m safety margin must keep the exact footprint clear.
    sw_own = gt_sweep_collision(res_g["segments"], start, ptab, gap)
    assert not sw_own["collision"], "planner cleared lethal but footprint hits own grid"

    # 7. pose validity: pose right next to the wall is invalid, far is valid.
    assert not maps_w.pose_valid(98, 100, 0)
    assert maps_w.pose_valid(30, 100, 0)

    print("planner_loop selftest PASSED")


def main():
    ap = argparse.ArgumentParser(description="GEMS R3.c planner closed loop v0")
    ap.add_argument("--out", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run_study(args.out)


if __name__ == "__main__":
    main()
