#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import apply_edit_to_checkpoint_copy
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    edit = MeshEdit(**json.loads(Path(args.edit_json).read_text(encoding="utf-8")))
    report = apply_edit_to_checkpoint_copy(args.checkpoint_path, edit, args.output_dir)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
