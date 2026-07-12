#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.edit import make_mask_variants as masks


class MaskVariantsTest(unittest.TestCase):
    def synthetic_stale(self):
        stale = np.zeros((32, 32), dtype=bool)
        stale[10:14, 11:16] = True
        stale[20, 22] = True
        return stale

    def assert_partition_exact(self, stale):
        arr = masks.stale_to_mask(stale)
        valid = arr == 255
        stale_px = arr == 0
        self.assertFalse(np.logical_and(valid, stale_px).any())
        self.assertTrue(np.logical_or(valid, stale_px).all())
        self.assertEqual(set(np.unique(arr).tolist()), {0, 255})

    def test_transform_invariants(self):
        stale = self.synthetic_stale()
        stale1 = masks.dilate_stale(stale, 1)
        stale4 = masks.dilate_stale(stale, 4)
        stale16 = masks.dilate_stale(stale, 16)
        box = masks.box_stale(stale)

        self.assertTrue(np.all(stale4 | ~stale1))
        self.assertTrue(np.all(stale16 | ~stale4))
        self.assertTrue(np.all(box | ~stale))
        self.assertGreaterEqual(int(stale4.sum()), int(stale1.sum()))
        self.assertGreaterEqual(int(stale16.sum()), int(stale4.sum()))
        self.assertGreaterEqual(int(box.sum()), int(stale.sum()))

        for transformed in (stale1, stale4, stale16, box):
            self.assert_partition_exact(transformed)

    def test_build_variant_manifest_and_mask(self):
        stale = self.synthetic_stale()
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            src = root / "edited"
            for rel_dir in ("renders", "gt", "depths", "masks"):
                (src / rel_dir).mkdir(parents=True)
            (src / "renders" / "00000.png").write_bytes(b"render")
            (src / "gt" / "00000.png").write_bytes(b"gt")
            (src / "depths" / "00000.npy").write_bytes(b"depth")
            (src / "camera_index.json").write_text("{}", encoding="utf-8")
            Image.fromarray(masks.stale_to_mask(stale), mode="L").save(
                src / "masks" / "00000.png")
            manifest = {
                "edit": {
                    "masks": {"00000": "masks/00000.png"},
                    "policy": "base policy",
                },
                "sizes": {"files": {}, "n_files": 0, "cache_mb_raw": 0.0},
            }
            (src / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")

            out_root = root / "variants"
            out_manifest = masks.build_variant(src, out_root, manifest, "box2d")

            self.assertEqual(out_manifest["edit"]["masks"],
                             {"00000": "masks/00000.png"})
            self.assertIn("mask_variant=box2d", out_manifest["edit"]["policy"])
            self.assertIn("masks/00000.png", out_manifest["sizes"]["files"])
            self.assertNotIn("manifest.json", out_manifest["sizes"]["files"])
            self.assertTrue((out_root / "box2d" / "manifest.json").is_file())

            out_stale = masks.mask_to_stale(out_root / "box2d" / "masks" /
                                            "00000.png")
            np.testing.assert_array_equal(out_stale, masks.box_stale(stale))
            self.assert_partition_exact(out_stale)


if __name__ == "__main__":
    unittest.main()
