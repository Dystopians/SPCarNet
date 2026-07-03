"""GEMS Stage One scene registry (PROTOCOL.md section 1 / section 4).

This file is part of the protocol: paths, ingestion configs, splits, GT
assets, ROIs, and units are FROZEN here. Editing a frozen field (e.g. a
courtyard ROI once set) is a metric-definition change and bumps the
protocol MAJOR version.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SceneSpec:
    """Frozen per-scene evaluation configuration.

    split:
        'llff8' -> COLMAP cameras sorted by image name, idx % 8 == 0 is test
                   (exact rule of scene/dataset_readers.py::readColmapSceneInfo
                   with eval=True).
        'file5' -> file split (every 5th view is test). If
                   <source_path>/split.json exists it is consumed via the
                   repo's split_strategy='file' loader; otherwise the
                   deterministic idx % 5 == 0 rule on the sorted camera list
                   is applied (llffhold=5).
    gt: metric-only GT assets. Optional keys:
        'mesh_path', 'scan_paths', 'gt_depth_dir', 'colmap_sparse'.
    roi: axis-aligned box {'min': [x,y,z], 'max': [x,y,z], 'z_band': [lo, hi]}
         or None if not yet frozen (geometry/downstream metrics that need an
         ROI are skipped until it is frozen here).
    units_per_meter: scene units per metric meter (None = unknown/not metric).
    """
    name: str
    source_path: str
    images: str
    resolution: int
    white_background: bool
    split: str  # 'llff8' | 'file5'
    gt: dict = field(default_factory=dict)
    roi: Optional[dict] = None
    units_per_meter: Optional[float] = None
    exists: bool = True


SCENES: dict = {
    # dev_real_A (PROTOCOL section 1): Mip-NeRF360 garden at the clean-baseline
    # ingestion config `--images images_4 -r -1 --eval`.
    "garden": SceneSpec(
        name="garden",
        source_path="/data/peilincai/mesh_datasets/mipnerf360/garden",
        images="images_4",
        resolution=-1,
        white_background=False,
        split="llff8",
        gt={
            "colmap_sparse": "/data/peilincai/mesh_datasets/mipnerf360/garden/sparse/0",
        },
        roi=None,
        units_per_meter=None,  # COLMAP scale, not metric
        exists=True,
    ),
    # dev_drive_A (PROTOCOL section 1): ETH3D courtyard at the stageR precedent
    # ingestion config `--images images -r 8 --eval`.
    "courtyard": SceneSpec(
        name="courtyard",
        source_path="/data/peilincai/mesh_datasets/eth3d_colmap/courtyard",
        images="images",
        resolution=8,
        white_background=False,
        split="llff8",
        gt={
            "scan_paths": [
                "/data/peilincai/mesh_datasets/eth3d/courtyard/courtyard/scan_clean/scan1.ply",
                "/data/peilincai/mesh_datasets/eth3d/courtyard/courtyard/scan_clean/scan2.ply",
            ],
            # ETH3D scan_alignment.mlp per-scan rigid transforms into the
            # calibration/camera frame (verified: median sparse->scan distance
            # 0.036 m transformed vs 1.19 m raw). Order matches scan_paths.
            "scan_transforms": [
                [[0.996548, 0.0830159, -5.48683e-05, 3.81589],
                 [-0.0830159, 0.996548, 0.000287318, -7.23808],
                 [7.85309e-05, -0.000281772, 1.0, 1.86659],
                 [0.0, 0.0, 0.0, 1.0]],
                [[1.0, -2.38419e-07, -5.96047e-08, 3.59469],
                 [2.38419e-07, 1.0, 2.38419e-07, 0.562818],
                 [5.96046e-08, -2.38419e-07, 1.0, 1.80679],
                 [0.0, 0.0, 0.0, 1.0]],
            ],
            "colmap_sparse": "/data/peilincai/mesh_datasets/eth3d_colmap/courtyard/sparse/0",
        },
        # ROI FROZEN 2026-07-02 (PROTOCOL 4.3 g4, changelog 1.1.1): combined
        # TRANSFORMED scan_clean AABB expanded 0.3 m (37.8M pts after
        # scan_alignment.mlp: min [-3.722,-13.925,0.118],
        # max [28.264,10.002,18.449]). The earlier same-day raw-AABB freeze was
        # voided together with the misaligned g4 row it produced
        # (courtyard_clean30k_v2; see LEDGER).
        # z_band FROZEN 2026-07-02 (M5 up-axis derivation, LEDGER GOAL #008):
        # +z is up in the calibration frame (mean camera up-vector
        # [-0.025,-0.013,0.9996], all 38 cams dot>=0.908); ground = P1 of
        # transformed-scan z within the camera xy-footprint+1m = 0.146 m
        # (cameras then sit 1.55 m above ground - consistent with handheld
        # DSLR). z_band = [ground, ground+1.5]. Never edit (MAJOR bump).
        roi={
            "min": [-4.022, -14.225, -0.182],
            "max": [28.564, 10.302, 18.749],
            "z_band": [0.146, 1.646],
        },
        units_per_meter=1.0,  # ETH3D laser scans are metric
        exists=True,
    ),
    # toy_parking (PROTOCOL section 1.1): procedural scene, built in M1b by
    # tools/gems/build_toy_parking.py (seed 0). Root is /data per LEDGER
    # DEC-008 (storage resolved; durable root moved from /home to /data).
    "toy_parking": SceneSpec(
        name="toy_parking",
        source_path="/data/peilincai/gems_stage1/datasets/toy_parking",
        images="images",
        resolution=-1,
        white_background=False,
        split="file5",
        gt={
            "mesh_path": "/data/peilincai/gems_stage1/datasets/toy_parking/gt/mesh.obj",
            "gt_depth_dir": "/data/peilincai/gems_stage1/datasets/toy_parking/gt/depth",
            "colmap_sparse": "/data/peilincai/gems_stage1/datasets/toy_parking/sparse/0",
        },
        # ROI frozen at dataset build time (M1b), z_band[0] corrected to the true
        # ground level 0.0 BEFORE any d1/d2 row existed (PROTOCOL 4.4 convention:
        # ground = z_band[0]; d2 vehicle band = [z_band[0]+0.1, z_band[0]+1.5]).
        roi={
            "min": [-17.025, -17.025, -0.1],
            "max": [17.025, 17.025, 3.2],
            "z_band": [0.0, 1.5],
        },
        units_per_meter=1.0,  # built in meters by spec 1.1
        exists=True,
    ),
}

# ---------------------------------------------------------------------------
# Stage Two S-GEO (MATRIX cell D-2, GEMS_Stage2_Prompt §3): toy_parking
# generated variants, built 2026-07-03 by the SAME generator
# (tools/gems/build_toy_parking.py) with layout parameters exposed as CLI args
# (LEDGER GOAL #016). Frozen ingestion identical to toy_parking: 90 views
# @1000x750, file5 split (every 5th view test), GT mesh + per-view GT depth
# exported at build time, images/-r -1.
#   toy_parking_v2   (altered layout): --seed 1 --car-yaws-deg 180,0
#                    --car-offsets -0.35,0.4,0.3,-0.3 --wall-x 3.0 --fence-x -9.0
#   toy_parking_occl (added occlusion): --seed 0 + 2 extra vehicles parked at
#                    (±2.4, 9.7, yaw 90°) directly behind the bays, occluding
#                    the northern ring cameras' sightlines to cars/bays/curb.
# ROIs recomputed per variant from each GT mesh extent at build time (both:
# xy ±17.025 exactly, z ∈ [-0.02, 3.0]) -> identical to the toy_parking ROI;
# ground z = 0 by construction, z_band = [0.0, 1.5] (PROTOCOL 4.4).
for _vname in ["toy_parking_v2", "toy_parking_occl"]:
    _vsrc = f"/data/peilincai/gems_stage1/datasets/{_vname}"
    SCENES[_vname] = SceneSpec(
        name=_vname,
        source_path=_vsrc,
        images="images",
        resolution=-1,
        white_background=False,
        split="file5",
        gt={
            "mesh_path": f"{_vsrc}/gt/mesh.obj",
            "gt_depth_dir": f"{_vsrc}/gt/depth",
            "colmap_sparse": f"{_vsrc}/sparse/0",
        },
        roi={
            "min": [-17.025, -17.025, -0.1],
            "max": [17.025, 17.025, 3.2],
            "z_band": [0.0, 1.5],
        },
        units_per_meter=1.0,
        exists=True,
    )

# Stage Two S-REND expansion (MATRIX.md): the remaining Mip-NeRF360 scenes at
# the clean-baseline ingestion configs (outdoor images_4, indoor images_2,
# llff8, -r -1). COLMAP sparse declared for g1; no GT mesh (g4/d skipped).
_M360 = "/data/peilincai/mesh_datasets/mipnerf360"
for _name, _imgdir in [
    ("bicycle", "images_4"), ("flowers", "images_4"), ("stump", "images_4"),
    ("treehill", "images_4"), ("room", "images_2"), ("counter", "images_2"),
    ("kitchen", "images_2"), ("bonsai", "images_2"),
]:
    SCENES[_name] = SceneSpec(
        name=_name,
        source_path=f"{_M360}/{_name}",
        images=_imgdir,
        resolution=-1,
        white_background=False,
        split="llff8",
        gt={"colmap_sparse": f"{_M360}/{_name}/sparse/0"},
        roi=None,
        units_per_meter=None,
        exists=True,
    )

# ---------------------------------------------------------------------------
# Stage Two S-GEO (MATRIX cell D-1b): SS3DM CARLA street sequences, converted
# by tools/gems_train/ss3dm_ingest.py from
# /data/peilincai/mesh_datasets/SS3DM (150_streetsurf, 4 towns).
# Frozen ingestion policy (see each dataset_manifest.json): 3 front cameras x
# 150 frames = 450 images at 1920x1080; resolution=2 (-r 2 -> 960x540) keeps
# the trainer's cuda data_device GT-image footprint at ~2.8 GB (450 full-res
# would be ~11.2 GB, 900 imgs of 6 cams ~22.4 GB).
# split='file5' consumes <source_path>/split.json; its content is a WHOLE-FRAME
# holdout (frame idx % 8 == 0 -> test, all 3 cams of that frame; 393/57), not
# every-5th (the registry code only keys on the literal 'file5').
# Trainer/COLMAP world = diag(1,-1,1) @ raw SS3DM world (raw CARLA world is
# left-handed, c2w rotation det=-1); LiDAR evidence npz (recorded in the
# manifests) stay in the RAW frame -> negate y when consuming.
# gt.mesh_path OBJ units are CENTIMETERS (x0.01 -> m, schema
# town_mesh_unit_scale=0.01); mesh->trainer-world alignment frozen 2026-07-03
# (S-GEO activation, LEDGER GOAL #R-08) as gt.mesh_transform =
# diag(0.01,-0.01,0.01) (cm->m scale THEN the diag(1,-1,1) world mirror;
# diagonal matrices commute; zero translation). VERIFIED per town in
# /data/peilincai/gems_stage1/analysis/sgeo_roi_freeze/<town>.json:
# (B1) median NN distance of 100k in-ROI LiDAR points (converted frame) to 2M
# transformed-mesh surface samples = 0.073/0.103/0.076 m (towns 01/02/03;
# GT-sample-density-limited) vs the WRONG (unmirrored) diag(0.01,0.01,0.01)
# control: 32.98 m / "no faces reach the ROI" / 0.13 m (town03's control is
# weak because CARLA's near-symmetric road grid puts road planes under the
# mirrored ROI too; the B2 projection check disambiguates);
# (B2, prescribed) per-16x16px-bin min-depth mesh samples projected into a
# mid-test front camera vs the raw GT depth PNG: median ratio
# 1.0028/1.0045/1.0130 (~=1 within 1.3%; p10 0.998-0.999, p90 tail =
# occlusion, expected for a crude z-buffer).
# gt.gt_depth_dir (g1/g2 activation): per-TEST-VIEW float32 z-depth .npy at
# the -r 2 render resolution 960x540, exported by
# tools/gems_train/ss3dm_export_gt_depth.py (decode raw*1000/65535 m,
# sky/far-plane saturation raw>=65534 -> 0 invalid, nearest at target pixel
# centers d[1::2,1::2]). Verified: npy vs LiDAR projection median ratio
# 0.9975 (94% within 5%); B0 rendered median depth / GT ratio ~=0.99 in the
# 10-20 m band (larger deviations at <5 m and >40 m are real model error).
# gt.mesh_roi_presample_crop: SS3DM town meshes cover ~4-11 km^2 (far
# terrain/skirt geometry) vs ~0.02-0.03 km^2 ROI; g4's frozen 1M-on-full-mesh
# sampling would leave only ~3k in-ROI GT points (NN spacing ~3 m >> tau).
# The flag (defined at this first SS3DM geometry eval, before any valid g4
# row existed; toy rows untouched) restricts GT sampling to ROI-intersecting
# faces; the posterior ROI crop of both clouds is unchanged. Even so, 1M GT
# samples over ~3e4 m^2 -> NN spacing ~0.09 m: absolute F@5cm/chamfer are
# GT-sampling-density-limited; paired clean-vs-B50 deltas are unaffected.
# ROIs FROZEN 2026-07-03 (derivation:
# analysis/sgeo_roi_freeze/derive_roi_and_verify_mesh.py, per-town json):
# xy = camera-centers xy-bbox +/- 30 m; up-axis verified +z in the mirrored
# frame (mean camera up [~0,~0,+0.99], min dot(+z) 0.966, courtyard-M5
# pattern); ground = P1 of LiDAR z (converted frame, range filter
# [0.5,120] m) within the camera xy-footprint+1m = -0.069/0.150/-0.073/-0.071
# (towns 01/02/03/06; cameras sit 2.25-2.47 m above -- vehicle roof, flat:
# per-trajectory-fifth ground spread <= 7 mm every town);
# z_band = [ground, ground+1.5] (PROTOCOL 4.4 convention);
# ROI z = [ground-0.3, P99.9(in-ROI-xy LiDAR z)+0.3] (0.3 m margin mirrors
# the courtyard scan-AABB+0.3 rule). Never edit (MAJOR bump).
# RAM (measured 2026-07-03, exact g4 trimesh load, scratchpad mesh_rss.log):
# Town01 13.3 / Town02 5.1 / Town03 11.1 / Town06 24.7 GB peak RSS.
# Town06 EXCEEDS the pre-registered 20 GB bar -> g4/d1/d2 INFEASIBLE-for-now
# (mesh_path=None below; do NOT silently decimate; g1/g2/g3 still active).
_SS3DM_DS = "/data/peilincai/gems_stage1/datasets"
_SS3DM_MESH = "/data/peilincai/mesh_datasets/SS3DM/meshes/mesh"
_SS3DM_MESH_TRANSFORM = [
    [0.01, 0.0, 0.0, 0.0],
    [0.0, -0.01, 0.0, 0.0],
    [0.0, 0.0, 0.01, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
_SS3DM_ROI = {
    "Town01": {"min": [57.874, -99.822, -0.369], "max": [119.037, 12.164, 7.433],
               "z_band": [-0.069, 1.431]},
    "Town02": {"min": [91.471, -218.128, -0.15], "max": [204.169, -156.987, 9.394],
               "z_band": [0.15, 1.65]},
    "Town03": {"min": [-87.206, 163.657, -0.373], "max": [23.888, 225.838, 9.619],
               "z_band": [-0.073, 1.427]},
    "Town06": {"min": [-35.165, -14.956, -0.371], "max": [76.793, 46.578, 8.407],
               "z_band": [-0.071, 1.429]},
}
for _town in ["Town01", "Town02", "Town03", "Town06"]:
    _key = f"ss3dm_{_town.lower()}"
    _gt = {
        "colmap_sparse": f"{_SS3DM_DS}/{_key}/sparse/0",
        "gt_depth_dir": f"{_SS3DM_DS}/{_key}/gt/depth",
    }
    if _town == "Town06":
        # Measured 24.7 GB peak RSS > the 20 GB bar (see block comment):
        # surface-GT metrics stay gated until a streaming loader exists.
        _gt["mesh_path"] = None
        _gt["mesh_infeasible_note"] = (
            "g4/d1/d2 INFEASIBLE-for-now: trimesh load of Town06_obj.obj "
            "(4.5 GB ASCII OBJ) measured 24.7 GB peak RSS > pre-registered "
            "20 GB bar (LEDGER #R-08); never silently decimate")
    else:
        _gt["mesh_path"] = f"{_SS3DM_MESH}/{_town}_obj.obj"
        _gt["mesh_transform"] = _SS3DM_MESH_TRANSFORM
        _gt["mesh_roi_presample_crop"] = True
    SCENES[_key] = SceneSpec(
        name=_key,
        source_path=f"{_SS3DM_DS}/{_key}",
        images="images",
        resolution=2,
        white_background=False,
        split="file5",
        gt=_gt,
        roi=_SS3DM_ROI[_town],
        units_per_meter=1.0,  # CARLA is metric
        exists=True,
    )
