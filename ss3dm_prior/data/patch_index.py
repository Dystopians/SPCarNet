"""Helpers for writing and reading teacher patch index files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ss3dm_prior.data.patch_types import PatchIndexRecord


def write_patch_index_jsonl(index_path: str | Path, records: Iterable[PatchIndexRecord]) -> Path:
    index_path = Path(index_path).expanduser().resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")
    return index_path


def read_patch_index_jsonl(index_path: str | Path) -> list[dict]:
    index_path = Path(index_path).expanduser().resolve()
    records: list[dict] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
