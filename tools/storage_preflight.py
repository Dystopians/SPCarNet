#!/usr/bin/env python
"""GEMS D6 storage preflight.

Checks free space on the volumes backing the given paths and refuses (exit 1)
if any is below the floor. Also flags /dev/shm as banned for run artifacts.

Usage:
    python tools/storage_preflight.py PATH [PATH ...] [--min-free-gb 50]

Exit codes: 0 = all volumes OK, 1 = at least one volume below floor or a
target resolves to /dev/shm.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def volume_report(path: str, min_free_gb: float) -> dict:
    resolved = os.path.realpath(path)
    probe = resolved
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1e9
    on_shm = resolved.startswith("/dev/shm")
    return {
        "path": path,
        "resolved": resolved,
        "checked_mount_of": probe,
        "total_gb": round(usage.total / 1e9, 1),
        "free_gb": round(free_gb, 1),
        "min_free_gb": min_free_gb,
        "dev_shm": on_shm,
        "ok": (free_gb >= min_free_gb) and not on_shm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="target output paths for the upcoming run")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    args = parser.parse_args()

    reports = [volume_report(p, args.min_free_gb) for p in args.paths]
    all_ok = all(r["ok"] for r in reports)
    print(json.dumps({"ok": all_ok, "volumes": reports}, indent=1))
    if not all_ok:
        for r in reports:
            if not r["ok"]:
                reason = "on /dev/shm (banned)" if r["dev_shm"] else (
                    f"only {r['free_gb']} GB free < floor {r['min_free_gb']} GB"
                )
                print(f"PREFLIGHT FAIL: {r['path']} -> {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
