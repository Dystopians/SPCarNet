#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_facelocal_sh1_delta import collect_samples


def _save_rgb(path: Path, chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hwc = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(hwc).save(path)


def _write_base_view(path: Path, *, parent: np.ndarray, gt: np.ndarray, face_id: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = gt - parent
    np.savez_compressed(
        path,
        face_id=np.full(parent.shape[1:], int(face_id), dtype=np.int32),
        residual_l1=np.mean(np.abs(residual), axis=0).astype(np.float16),
        residual_rgb=residual.astype(np.float16),
        alpha=np.ones(parent.shape[1:], dtype=np.float16),
        texture=np.ones(parent.shape[1:], dtype=np.float16),
        camera_center=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        rgb_render=parent.astype(np.float16),
        rgb_gt=gt.astype(np.float16),
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base_evidence"
        teacher_dir = root / "teacher_renders"
        out = root / "teacher_evidence"
        parent = np.full((3, 4, 4), 0.20, dtype=np.float32)
        gt = np.full((3, 4, 4), 0.80, dtype=np.float32)

        _write_base_view(base / "views" / "00000.npz", parent=parent, gt=gt)
        _write_base_view(base / "views" / "00001.npz", parent=parent, gt=gt)
        (base / "top_residual_supports.csv").write_text(
            "rank,face_id,score,pixel_count,view_hits,mean_l1_error,mean_texture,residual_consistency,mean_residual_r,mean_residual_g,mean_residual_b\n"
            "1,1,1.0,16,1,0.6,0.0,1.0,0.6,0.6,0.6\n",
            encoding="utf-8",
        )
        (base / "surface_evidence_summary.json").write_text(
            json.dumps({"per_view_npz_fields": ["face_id", "residual_l1", "residual_rgb", "alpha", "rgb_render", "rgb_gt"]}) + "\n",
            encoding="utf-8",
        )

        # Good teacher: closer to GT than parent and sufficiently different.
        _save_rgb(teacher_dir / "00000.png", np.full((3, 4, 4), 0.75, dtype=np.float32))
        # Bad teacher: worse than parent, so it must be masked out.
        _save_rgb(teacher_dir / "00001.png", np.full((3, 4, 4), 0.05, dtype=np.float32))

        cmd = [
            sys.executable,
            str(ROOT / "scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py"),
            "--base_evidence_dir",
            str(base),
            "--teacher_render_dir",
            str(teacher_dir),
            "--out_dir",
            str(out),
            "--teacher_parent_delta_min",
            "0.01",
            "--teacher_render_error_margin",
            "0.0",
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        with np.load(out / "views" / "00000.npz") as good:
            assert "teacher_residual_rgb" in good.files
            assert "teacher_residual_l1" in good.files
            assert float(good["teacher_better_mask"].mean()) == 1.0
            assert float(good["teacher_residual_l1"].mean()) > 0.50
        with np.load(out / "views" / "00001.npz") as bad:
            assert float(bad["teacher_better_mask"].mean()) == 0.0
            assert float(bad["teacher_residual_l1"].mean()) == 0.0

        summary = json.loads((out / "teacher_surface_evidence_summary.json").read_text(encoding="utf-8"))
        assert summary["processed_views"] == 2, summary
        assert summary["top_support_rebuild"]["rows"] == 1, summary

        samples = collect_samples(
            [out / "views" / "00000.npz"],
            [1],
            {1: {"score": 1.0, "consistency": 1.0}},
            high_error_quantile=0.0,
            min_alpha=0.1,
            barycentric_tolerance=0.01,
            max_samples_per_face_view=16,
            max_total_samples=16,
            uniform_barycentric=True,
            residual_rgb_key="teacher_residual_rgb",
            residual_l1_key="teacher_residual_l1",
        )
        assert samples.count == 16, samples.count
        mean_rgb = samples.residual_rgb.mean(axis=0)
        assert np.allclose(mean_rgb, np.asarray([0.55, 0.55, 0.55], dtype=np.float32), atol=0.02), mean_rgb

    print("[teacher surface evidence cache smoke] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
