"""Data discovery and manifest utilities for SS3DM prior experiments."""

from ss3dm_prior.data.corruptions import CorruptionResult, apply_patch_corruptions
from ss3dm_prior.data.discovery import (
    EXPECTED_CAMERA_NAMES,
    EXPECTED_LIDAR_NAMES,
    EXPECTED_TOWN_IDS,
    SequenceDiscoveryRecord,
    discover_ss3dm_sequences,
    parse_num_frames_from_sequence_name,
)
from ss3dm_prior.data.manifest import (
    build_manifest,
    summarize_manifest,
    validate_manifest,
)
from ss3dm_prior.data.observed_fusion import (
    build_sequence_observed_cache,
    generate_tile_centers,
    save_observed_cache,
)
from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.data.patch_index import read_patch_index_jsonl, write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.data.raw_sequence import RawSequence, select_manifest_entries_by_split
from ss3dm_prior.data.scenario_loader import ScenarioData, load_scenario
from ss3dm_prior.data.teacher_patch_builder import (
    build_teacher_patches_for_sequence,
    load_sequence_observed_cache,
)
from ss3dm_prior.data.town_mesh_cache import TownMeshCache, load_town_mesh_cache

__all__ = [
    "CorruptionResult",
    "EXPECTED_CAMERA_NAMES",
    "EXPECTED_LIDAR_NAMES",
    "EXPECTED_TOWN_IDS",
    "PatchIndexRecord",
    "RawSequence",
    "ScenarioData",
    "SequenceDiscoveryRecord",
    "TeacherPatchSample",
    "TownMeshCache",
    "apply_patch_corruptions",
    "build_manifest",
    "build_sequence_observed_cache",
    "build_teacher_patches_for_sequence",
    "convert_obj_to_cache",
    "discover_ss3dm_sequences",
    "generate_tile_centers",
    "load_sequence_observed_cache",
    "load_town_mesh_cache",
    "load_scenario",
    "parse_num_frames_from_sequence_name",
    "read_patch_index_jsonl",
    "save_observed_cache",
    "select_manifest_entries_by_split",
    "summarize_manifest",
    "validate_manifest",
    "write_patch_index_jsonl",
]
