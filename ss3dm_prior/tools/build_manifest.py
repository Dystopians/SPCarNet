"""Build a JSON manifest for SS3DM_raw sequence discovery."""

from __future__ import annotations

import argparse

from ss3dm_prior.data.manifest import build_manifest, summarize_manifest
from ss3dm_prior.utils.io import dump_json


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan SS3DM_raw and write a sequence-level JSON manifest."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the SS3DM_raw root directory.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output manifest JSON file.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    manifest = build_manifest(args.root)
    dump_json(args.out, manifest, indent=2)
    print(summarize_manifest(manifest))
    print(f"manifest_out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
