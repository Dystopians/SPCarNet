"""Convenience wrapper around a manifest entry and its raw sequence assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ss3dm_prior.data.scenario_loader import ScenarioData, load_scenario


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


@dataclass
class RawSequence:
    manifest_entry: dict[str, Any]
    scenario: ScenarioData

    @classmethod
    def from_manifest_entry(cls, manifest_entry: dict[str, Any]) -> "RawSequence":
        return cls(
            manifest_entry=manifest_entry,
            scenario=load_scenario(manifest_entry["sequence_root"]),
        )

    @property
    def town_id(self) -> str:
        return str(self.manifest_entry["town_id"])

    @property
    def sequence_id(self) -> str:
        return str(self.manifest_entry["sequence_id"])

    @property
    def sequence_root(self) -> Path:
        return Path(self.manifest_entry["sequence_root"])

    @property
    def num_frames(self) -> int:
        return int(
            self.scenario.num_frames
            if self.scenario.num_frames is not None
            else self.manifest_entry["num_frames_from_name"]
        )

    def camera_names(self) -> list[str]:
        names = _ordered_unique(
            list(self.manifest_entry.get("camera_names", [])) + list(self.scenario.cameras.keys())
        )
        return sorted(names)

    def lidar_names(self) -> list[str]:
        names = _ordered_unique(
            list(self.manifest_entry.get("lidar_names", [])) + list(self.scenario.lidars.keys())
        )
        return sorted(names)

    def lidar_frame_path(self, lidar_name: str, frame_idx: int) -> Path:
        lidar_dir_map = self.manifest_entry.get("lidar_dir_map", {})
        if lidar_name in lidar_dir_map:
            return Path(lidar_dir_map[lidar_name]) / f"{frame_idx:08d}.npz"
        return self.sequence_root / "lidars" / lidar_name / f"{frame_idx:08d}.npz"

    def iter_frame_indices(self, frame_stride: int) -> list[int]:
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        frame_indices = list(range(0, self.num_frames, frame_stride))
        if not frame_indices:
            frame_indices = [0]
        if frame_indices[-1] != self.num_frames - 1:
            frame_indices.append(self.num_frames - 1)
        return frame_indices


def select_manifest_entries_by_split(
    manifest: dict[str, Any],
    split_config: dict[str, Any],
    subsets: tuple[str, ...] = ("train", "val", "test"),
) -> list[dict[str, Any]]:
    selected_towns: set[str] = set()
    for subset in subsets:
        selected_towns.update(split_config.get(f"{subset}_towns", []))
    entries = [
        entry
        for entry in manifest.get("entries", [])
        if entry.get("town_id") in selected_towns
    ]
    return sorted(entries, key=lambda entry: str(entry["sequence_id"]))
