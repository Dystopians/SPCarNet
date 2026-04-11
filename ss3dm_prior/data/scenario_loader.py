"""Load and summarize raw SS3DM sequence scenario metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np


def _append_warning(warning_list: list[str], message: str) -> None:
    warning_list.append(message)
    warnings.warn(message, stacklevel=2)


def _maybe_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def _extract_num_frames(data: dict[str, Any], fallback: int | None = None) -> int | None:
    raw_value = data.get("num_frames", fallback)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except Exception:
        return fallback


@dataclass
class ObserverRecord:
    name: str
    class_name: str
    num_frames: int | None
    data_keys: list[str]
    intr: np.ndarray | None = None
    c2w: np.ndarray | None = None
    hw: np.ndarray | None = None
    l2v: np.ndarray | None = None
    sensor_v2w: np.ndarray | None = None
    timestamp: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def center_array(self) -> np.ndarray:
        if self.c2w is not None and self.c2w.ndim == 3 and self.c2w.shape[-2:] == (4, 4):
            return np.asarray(self.c2w[:, :3, 3], dtype=np.float32)
        if self.sensor_v2w is not None and self.sensor_v2w.ndim == 3 and self.sensor_v2w.shape[-2:] == (4, 4):
            return np.asarray(self.sensor_v2w[:, :3, 3], dtype=np.float32)
        if self.l2v is not None and self.l2v.ndim == 3 and self.l2v.shape[-2:] == (4, 4):
            return np.asarray(self.l2v[:, :3, 3], dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "num_frames": self.num_frames,
            "data_keys": self.data_keys,
            "intr_shape": list(self.intr.shape) if self.intr is not None else None,
            "c2w_shape": list(self.c2w.shape) if self.c2w is not None else None,
            "hw_shape": list(self.hw.shape) if self.hw is not None else None,
            "l2v_shape": list(self.l2v.shape) if self.l2v is not None else None,
            "sensor_v2w_shape": list(self.sensor_v2w.shape) if self.sensor_v2w is not None else None,
            "timestamp_shape": list(self.timestamp.shape) if self.timestamp is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class ScenarioData:
    scene_id: str | None
    num_frames: int | None
    cameras: dict[str, ObserverRecord]
    lidars: dict[str, ObserverRecord]
    metas: dict[str, Any]
    source_format: str
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "num_frames": self.num_frames,
            "source_format": self.source_format,
            "camera_names": sorted(self.cameras.keys()),
            "lidar_names": sorted(self.lidars.keys()),
            "num_cameras": len(self.cameras),
            "num_lidars": len(self.lidars),
            "warnings": list(self.warnings),
            "camera_summaries": {
                name: record.summary_dict() for name, record in sorted(self.cameras.items())
            },
            "lidar_summaries": {
                name: record.summary_dict() for name, record in sorted(self.lidars.items())
            },
        }


def _load_with_torch(scenario_pt_path: Path, warning_list: list[str]) -> dict[str, Any] | None:
    try:
        import torch  # type: ignore
    except Exception as exc:
        _append_warning(
            warning_list,
            f"torch.load unavailable for {scenario_pt_path}: {exc}. Falling back to pickle.load.",
        )
        return None

    try:
        raw_obj = torch.load(scenario_pt_path, map_location="cpu", weights_only=False)
        if isinstance(raw_obj, dict):
            return raw_obj
        _append_warning(
            warning_list,
            f"torch.load returned non-dict object for {scenario_pt_path}: {type(raw_obj)}",
        )
        return None
    except Exception as exc:
        _append_warning(
            warning_list,
            f"torch.load failed for {scenario_pt_path}: {exc}. Falling back to pickle.load.",
        )
        return None


def _load_with_pickle(scenario_pt_path: Path, warning_list: list[str]) -> dict[str, Any] | None:
    try:
        with scenario_pt_path.open("rb") as handle:
            raw_obj = pickle.load(handle)
    except Exception as exc:
        _append_warning(warning_list, f"pickle.load failed for {scenario_pt_path}: {exc}")
        return None

    if not isinstance(raw_obj, dict):
        _append_warning(
            warning_list,
            f"pickle.load returned non-dict object for {scenario_pt_path}: {type(raw_obj)}",
        )
        return None
    return raw_obj


def _parse_scenario_txt_summary(scenario_txt_path: Path, warning_list: list[str]) -> dict[str, Any]:
    try:
        text = scenario_txt_path.read_text(encoding="utf-8")
    except Exception as exc:
        _append_warning(warning_list, f"Failed to read scenario.txt from {scenario_txt_path}: {exc}")
        return {}

    scene_id_match = re.search(r"'scene_id':\s*'([^']+)'", text)
    num_frames_match = re.search(r"'num_frames':\s*(\d+)", text)
    camera_names = sorted(set(re.findall(r"'(camera_[A-Z_]+)'", text)))
    lidar_names = sorted(set(re.findall(r"'(lidar_[A-Z_]+)'", text)))
    return {
        "scene_id": scene_id_match.group(1) if scene_id_match else None,
        "num_frames": int(num_frames_match.group(1)) if num_frames_match else None,
        "camera_names": camera_names,
        "lidar_names": lidar_names,
    }


def _build_observer_record(
    name: str,
    payload: dict[str, Any],
    fallback_num_frames: int | None,
    warning_list: list[str],
) -> ObserverRecord:
    class_name = str(payload.get("class_name", "Unknown"))
    num_frames = _extract_num_frames(payload, fallback_num_frames)
    data = payload.get("data", {})
    if not isinstance(data, dict):
        _append_warning(warning_list, f"Observer {name} has non-dict data payload")
        data = {}

    record = ObserverRecord(
        name=name,
        class_name=class_name,
        num_frames=num_frames,
        data_keys=sorted(data.keys()),
        intr=_maybe_numpy(data.get("intr")),
        c2w=_maybe_numpy(data.get("c2w")),
        hw=_maybe_numpy(data.get("hw")),
        l2v=_maybe_numpy(data.get("l2v")),
        sensor_v2w=_maybe_numpy(data.get("sensor_v2w")),
        timestamp=_maybe_numpy(data.get("timestamp")),
    )

    if class_name == "Camera":
        if record.intr is None:
            message = f"Camera observer {name} is missing intr"
            record.warnings.append(message)
            _append_warning(warning_list, message)
        if record.c2w is None:
            message = f"Camera observer {name} is missing c2w"
            record.warnings.append(message)
            _append_warning(warning_list, message)
        if record.hw is None:
            message = f"Camera observer {name} is missing hw"
            record.warnings.append(message)
            _append_warning(warning_list, message)
    if "Lidar" in class_name or name.startswith("lidar_"):
        if record.l2v is None and record.sensor_v2w is None:
            message = f"Lidar observer {name} is missing l2v/sensor_v2w"
            record.warnings.append(message)
            _append_warning(warning_list, message)

    return record


def load_scenario(sequence_root: str | Path) -> ScenarioData:
    sequence_path = Path(sequence_root).expanduser().resolve()
    scenario_pt_path = sequence_path / "scenario.pt"
    scenario_txt_path = sequence_path / "scenario.txt"

    warning_list: list[str] = []
    raw_obj = _load_with_torch(scenario_pt_path, warning_list)
    source_format = "torch.load"
    if raw_obj is None:
        raw_obj = _load_with_pickle(scenario_pt_path, warning_list)
        source_format = "pickle.load"

    if raw_obj is None:
        summary = _parse_scenario_txt_summary(scenario_txt_path, warning_list)
        return ScenarioData(
            scene_id=summary.get("scene_id"),
            num_frames=summary.get("num_frames"),
            cameras={
                name: ObserverRecord(
                    name=name,
                    class_name="Camera",
                    num_frames=summary.get("num_frames"),
                    data_keys=[],
                )
                for name in summary.get("camera_names", [])
            },
            lidars={
                name: ObserverRecord(
                    name=name,
                    class_name="RaysLidar",
                    num_frames=summary.get("num_frames"),
                    data_keys=[],
                )
                for name in summary.get("lidar_names", [])
            },
            metas={},
            source_format="scenario.txt_summary_only",
            warnings=warning_list,
            raw_payload=None,
        )

    observers = raw_obj.get("observers", {})
    metas = raw_obj.get("metas", {})
    scene_id = raw_obj.get("scene_id")
    num_frames = _extract_num_frames(metas, None)
    if num_frames is None and isinstance(observers, dict) and observers:
        first_observer = next(iter(observers.values()))
        if isinstance(first_observer, dict):
            num_frames = _extract_num_frames(first_observer, None)

    cameras: dict[str, ObserverRecord] = {}
    lidars: dict[str, ObserverRecord] = {}
    if not isinstance(observers, dict):
        _append_warning(warning_list, f"Scenario observers payload is not a dict for {sequence_path}")
        observers = {}

    for name, payload in observers.items():
        if not isinstance(payload, dict):
            _append_warning(warning_list, f"Observer {name} has non-dict payload")
            continue
        record = _build_observer_record(name, payload, num_frames, warning_list)
        if record.class_name == "Camera" or name.startswith("camera_"):
            cameras[name] = record
        elif "Lidar" in record.class_name or name.startswith("lidar_"):
            lidars[name] = record

    return ScenarioData(
        scene_id=str(scene_id) if scene_id is not None else None,
        num_frames=num_frames,
        cameras=cameras,
        lidars=lidars,
        metas=metas if isinstance(metas, dict) else {},
        source_format=source_format,
        warnings=warning_list,
        raw_payload=raw_obj,
    )
