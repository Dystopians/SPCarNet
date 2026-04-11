"""Validate a JSON manifest produced for SS3DM_raw discovery."""

from __future__ import annotations

import argparse

from ss3dm_prior.data.manifest import validate_manifest
from ss3dm_prior.utils.io import load_json


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SS3DM_raw manifest paths, sensor completeness, and counts."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the manifest JSON file.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    manifest = load_json(args.manifest)
    report = validate_manifest(manifest)

    print(f"num_entries: {report['num_entries']}")
    print("town_sequence_counts:")
    for town_id, count in report["town_sequence_counts"].items():
        print(f"  - {town_id}: {count}")
    print(
        "complete_sensor_sets: "
        f"cameras={report['complete_camera_entries']}/{report['num_entries']}, "
        f"lidars={report['complete_lidar_entries']}/{report['num_entries']}"
    )
    print(
        "frame_consistency: "
        f"{report['frame_consistent_entries']}/{report['num_entries']} entries fully consistent"
    )

    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    else:
        print("warnings: none")

    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
        return 1

    print("errors: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
