#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any


FORBIDDEN_SELECTION_SPLITS = {"test", "target_test", "heldout_test"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float,)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def file_sha1(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha1()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if max_bytes > 0 and total >= int(max_bytes):
                digest.update(f":truncated:{path.stat().st_size}".encode("utf-8"))
                break
    return digest.hexdigest()


def path_record(path: Path | str, *, hash_file: bool = False) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    row: dict[str, Any] = {
        "path": str(p),
        "exists": bool(exists),
        "is_file": bool(p.is_file()),
        "is_dir": bool(p.is_dir()),
    }
    if exists:
        try:
            stat = p.stat()
            row["mtime"] = float(stat.st_mtime)
            row["size_bytes"] = int(stat.st_size) if p.is_file() else None
        except OSError:
            pass
    if hash_file and p.is_file():
        row["sha1"] = file_sha1(p)
    return row


def command_record(name: str, cmd: list[str], *, log_path: Path | None = None) -> dict[str, Any]:
    return {
        "name": str(name),
        "cmd": [str(x) for x in cmd],
        "cmd_string": " ".join(str(x) for x in cmd),
        "log_path": str(log_path) if log_path is not None else None,
        "returncode": None,
        "elapsed_sec": None,
    }


def make_protocol_audit(
    *,
    fit_split: str = "train",
    policy_val_split: str = "train",
    target_split: str = "test",
    teacher_uses_gt: bool,
    selection_uses_test_gt: bool = False,
    capacity_selected_on: str = "train_policy_val",
    thresholds_selected_on: str = "train_policy_val",
) -> dict[str, Any]:
    forbidden = set()
    for key, value in {
        "fit_split": fit_split,
        "policy_val_split": policy_val_split,
        "capacity_selected_on": capacity_selected_on,
        "thresholds_selected_on": thresholds_selected_on,
    }.items():
        if str(value).lower() in FORBIDDEN_SELECTION_SPLITS:
            forbidden.add(key)
    passed = (not forbidden) and not bool(selection_uses_test_gt)
    return {
        "passed": bool(passed),
        "fit_split": str(fit_split),
        "policy_val_split": str(policy_val_split),
        "target_split": str(target_split),
        "teacher_uses_train_gt": bool(teacher_uses_gt),
        "selection_uses_test_gt": bool(selection_uses_test_gt),
        "capacity_selected_on": str(capacity_selected_on),
        "thresholds_selected_on": str(thresholds_selected_on),
        "forbidden_selection_fields": sorted(forbidden),
    }


def make_run_manifest(
    *,
    method: str,
    scene: str,
    run_root: Path,
    inputs: dict[str, Any],
    settings: dict[str, Any],
    commands: list[dict[str, Any]],
    protocol_audit: dict[str, Any],
    status: str = "PLANNED",
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "schema_version": 1,
        "method": str(method),
        "scene": str(scene),
        "status": str(status),
        "created_at": now,
        "updated_at": now,
        "run_root": str(run_root),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "inputs": json_safe(inputs),
        "settings": json_safe(settings),
        "commands": json_safe(commands),
        "protocol_audit": json_safe(protocol_audit),
        "errors": errors or [],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_vnext_report(path: Path, manifest: dict[str, Any]) -> None:
    audit = manifest.get("protocol_audit", {}) or {}
    inputs = manifest.get("inputs", {}) or {}
    settings = manifest.get("settings", {}) or {}
    lines = [
        "# vNext Certified Residual Surface Texture Run",
        "",
        f"- method: `{manifest.get('method', '')}`",
        f"- scene: `{manifest.get('scene', '')}`",
        f"- status: `{manifest.get('status', '')}`",
        f"- run root: `{manifest.get('run_root', '')}`",
        f"- protocol audit passed: `{audit.get('passed', False)}`",
        f"- target split: `{audit.get('target_split', '')}`",
        f"- selection uses test GT: `{audit.get('selection_uses_test_gt', None)}`",
        f"- capacity selected on: `{audit.get('capacity_selected_on', '')}`",
        f"- thresholds selected on: `{audit.get('thresholds_selected_on', '')}`",
        "",
        "## Inputs",
        "",
    ]
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(value, dict) and "path" in value:
            lines.append(f"- {key}: `{value.get('path')}` exists=`{value.get('exists')}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Settings", ""])
    for key in sorted(settings):
        lines.append(f"- {key}: `{settings[key]}`")
    lines.extend(["", "## Commands", ""])
    for command in manifest.get("commands", []) or []:
        lines.extend(
            [
                f"### {command.get('name', '')}",
                "",
                "```bash",
                str(command.get("cmd_string", "")),
                "```",
                "",
                f"- returncode: `{command.get('returncode')}`",
                f"- elapsed_sec: `{command.get('elapsed_sec')}`",
                f"- log: `{command.get('log_path')}`",
                "",
            ]
        )
    errors = manifest.get("errors", []) or []
    lines.extend(["## Errors", ""])
    if errors:
        for error in errors:
            lines.append(f"- `{error}`")
    else:
        lines.append("- none")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
