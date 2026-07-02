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
            "colmap_sparse": "/data/peilincai/mesh_datasets/eth3d_colmap/courtyard/sparse/0",
        },
        # ROI frozen at first courtyard geometry eval (PROTOCOL 4.3 g4);
        # not yet set -> g4/d1/d2 report skipped until then.
        roi=None,
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
