#!/usr/bin/env python3
"""Smoke-test the SPCarNet qualitative panel exporter with synthetic images."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    gt_dir = root / "gt"
    baseline_dir = root / "baseline"
    ours_dir = root / "ours"
    gt_dir.mkdir()
    baseline_dir.mkdir()
    ours_dir.mkdir()

    for index in range(4):
        frame = f"{index:05d}.png"
        gt = Image.new("RGB", (96, 64), (40 + index * 20, 80, 120))
        draw = ImageDraw.Draw(gt)
        draw.rectangle((16 + index * 3, 14, 54 + index * 3, 42), fill=(210, 210, 210))
        draw.line((0, 8 + index * 6, 95, 20 + index * 6), fill=(30, 30, 30), width=2)
        gt.save(gt_dir / frame)

        baseline = gt.copy()
        draw = ImageDraw.Draw(baseline)
        draw.rectangle((18 + index * 3, 16, 56 + index * 3, 44), outline=(255, 100, 70), width=4)
        baseline.save(baseline_dir / frame)

        ours = gt.copy()
        draw = ImageDraw.Draw(ours)
        if index in (1, 3):
            draw.rectangle((20 + index * 3, 18, 48 + index * 3, 38), outline=(80, 220, 100), width=2)
        ours.save(ours_dir / frame)

    return gt_dir, baseline_dir, ours_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "car_model" / "build_vnext_qualitative_panels.py"
    with tempfile.TemporaryDirectory(prefix="spcarnet_panel_smoke_") as temp:
        root = Path(temp)
        gt_dir, baseline_dir, ours_dir = _write_fixture(root)
        output_dir = root / "out"
        cmd = [
            sys.executable,
            str(script),
            "--gt_dir",
            str(gt_dir),
            "--method",
            f"baseline={baseline_dir}",
            "--method",
            f"ours={ours_dir}",
            "--reference_label",
            "baseline",
            "--candidate_label",
            "ours",
            "--output_dir",
            str(output_dir),
            "--panel_name",
            "smoke_panel",
            "--num_views",
            "3",
            "--tile_width",
            "96",
            "--selection_mode",
            "largest_candidate_reference_delta",
        ]
        subprocess.run(cmd, check=True)

        manifest_path = output_dir / "smoke_panel_manifest.json"
        panel_path = output_dir / "smoke_panel.png"
        summary_path = output_dir / "smoke_panel_summary.md"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert panel_path.is_file(), panel_path
        assert summary_path.is_file(), summary_path
        assert manifest["common_frame_count"] == 4, manifest
        assert manifest["selected_count"] == 3, manifest
        assert manifest["reference_label"] == "baseline", manifest
        assert manifest["candidate_label"] == "ours", manifest
        print(json.dumps({"status": "ok", "panel": str(panel_path)}, indent=2))


if __name__ == "__main__":
    main()
