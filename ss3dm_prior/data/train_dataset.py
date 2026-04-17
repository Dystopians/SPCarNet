"""Training dataset for online-corrupted teacher patch samples."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from ss3dm_prior.data.corruptions import apply_patch_corruptions
from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.patch_types import load_patch_npz
from ss3dm_prior.utils.io import load_yaml

import torch
from torch.utils.data import Dataset


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


class TeacherPatchTrainDataset(Dataset):
    def __init__(
        self,
        *,
        patch_index_path: str | Path,
        records: list[dict[str, Any]] | None = None,
        split_config: str | Path | dict[str, Any] | None = None,
        subsets: tuple[str, ...] = ("train",),
        corruption_config: dict[str, Any] | None = None,
        seed: int = 0,
        dynamic_corruption: bool = True,
    ) -> None:
        self.patch_index_path = Path(patch_index_path).expanduser().resolve()
        self.records = list(records) if records is not None else read_patch_index_jsonl(self.patch_index_path)
        self.seed = int(seed)
        self.corruption_config = corruption_config or {}
        self._sample_visit_counts: dict[str, int] = defaultdict(int)
        self.dynamic_corruption = bool(dynamic_corruption)

        if split_config is not None:
            if isinstance(split_config, (str, Path)):
                split_data = load_yaml(split_config)
            else:
                split_data = split_config
            selected_towns: set[str] = set()
            for subset in subsets:
                selected_towns.update(split_data.get(f"{subset}_towns", []))
            self.records = [
                record for record in self.records if str(record["town_id"]) in selected_towns
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        patch = load_patch_npz(record["patch_file"])
        clean_points = np.asarray(patch["clean_points"], dtype=np.float32)
        clean_normals = np.asarray(patch["clean_normals"], dtype=np.float32)
        observed_points = np.asarray(patch["observed_points"], dtype=np.float32)
        surface_query_points = np.asarray(patch.get("surface_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        surface_query_labels = np.asarray(patch.get("surface_query_labels", np.zeros((0,), dtype=np.int8)))
        free_query_points = np.asarray(patch.get("free_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        free_query_labels = np.asarray(patch.get("free_query_labels", np.zeros((0,), dtype=np.int8)))
        unknown_query_points = np.asarray(patch.get("unknown_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        query_points_all = np.asarray(patch.get("query_points_all", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        query_labels_all = np.asarray(patch.get("query_labels_all", np.zeros((0,), dtype=np.int8)))
        query_ignore_mask = np.asarray(patch.get("query_ignore_mask", np.zeros((0,), dtype=bool)), dtype=bool)
        patch_center_world = np.asarray(patch["patch_center_world"], dtype=np.float32)
        patch_id = str(_scalar(patch["patch_id"]))
        town_id = str(_scalar(patch["town_id"]))
        sequence_id = str(_scalar(patch["sequence_id"]))
        patch_cache_format_version = int(_scalar(patch.get("patch_cache_format_version", np.asarray(1, dtype=np.int32))))
        difficulty_components_json = str(_scalar(patch.get("difficulty_components_json", np.asarray("{}"))))
        patch_metadata_json = str(_scalar(patch.get("patch_metadata_json", np.asarray("{}"))))
        if self.dynamic_corruption:
            visit_count = self._sample_visit_counts[patch_id]
            self._sample_visit_counts[patch_id] += 1
            sample_key = f"{patch_id}__visit_{visit_count}"
        else:
            sample_key = patch_id

        corruption = apply_patch_corruptions(
            clean_points=clean_points,
            clean_normals=clean_normals,
            observed_points=observed_points,
            config=self.corruption_config,
            seed=self.seed,
            sample_key=sample_key,
        )
        sample = {
            "clean_points": torch.from_numpy(clean_points),
            "clean_normals": torch.from_numpy(clean_normals),
            "observed_points": torch.from_numpy(observed_points),
            "corrupted_points": torch.from_numpy(corruption.corrupted_points),
            "corrupted_normals": torch.from_numpy(corruption.corrupted_normals),
            "point_defect_target": torch.from_numpy(corruption.point_defect_target),
            "corruption_score_target": torch.tensor(
                corruption.corruption_score_target,
                dtype=torch.float32,
            ),
            "surface_query_points": torch.from_numpy(surface_query_points),
            "surface_query_labels": torch.from_numpy(surface_query_labels),
            "free_query_points": torch.from_numpy(free_query_points),
            "free_query_labels": torch.from_numpy(free_query_labels),
            "unknown_query_points": torch.from_numpy(unknown_query_points),
            "query_points_all": torch.from_numpy(query_points_all),
            "query_labels_all": torch.from_numpy(query_labels_all),
            "query_ignore_mask": torch.from_numpy(query_ignore_mask),
            "camera_support_count": torch.tensor(
                float(_scalar(patch.get("camera_support_count", np.asarray(0, dtype=np.int32)))),
                dtype=torch.float32,
            ),
            "lidar_support_count": torch.tensor(
                float(_scalar(patch.get("lidar_support_count", np.asarray(0, dtype=np.int32)))),
                dtype=torch.float32,
            ),
            "visible_surface_fraction": torch.tensor(
                float(_scalar(patch.get("visible_surface_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "free_space_fraction": torch.tensor(
                float(_scalar(patch.get("free_space_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "unknown_fraction": torch.tensor(
                float(_scalar(patch.get("unknown_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "intrinsic_patch_difficulty_target": torch.tensor(
                float(
                    _scalar(
                        patch.get(
                            "intrinsic_patch_difficulty_target",
                            np.asarray(0.0, dtype=np.float32),
                        )
                    )
                ),
                dtype=torch.float32,
            ),
            "patch_center_world": torch.from_numpy(patch_center_world),
            "town_id": town_id,
            "sequence_id": sequence_id,
            "patch_id": patch_id,
            "patch_cache_format_version": patch_cache_format_version,
            "difficulty_components": json.loads(difficulty_components_json),
            "patch_metadata": json.loads(patch_metadata_json),
            "corruption_metadata": corruption.metadata,
        }
        return sample
