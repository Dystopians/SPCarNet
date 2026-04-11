"""Small file IO helpers for SS3DM prior tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def dump_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_yaml(path: str | Path, payload: Any) -> None:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
