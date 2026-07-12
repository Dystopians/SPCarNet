#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.gems import build_toy_parking as toy


class DropElementsTest(unittest.TestCase):
    def test_argparse_drop_elements(self):
        parser = toy.build_arg_parser()
        default_args = parser.parse_args([])
        self.assertEqual(toy.parse_drop_elements(default_args.drop_elements), ())

        args = parser.parse_args(["--drop-elements", "car_0, car_1,,car_0"])
        self.assertEqual(toy.parse_drop_elements(args.drop_elements),
                         ("car_0", "car_1"))

    def test_filter_mesh_by_elements(self):
        verts = np.arange(15, dtype=np.float32).reshape(5, 3)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [1, 3, 4],
        ], dtype=np.int64)
        colors = np.ones((5, 3), dtype=np.float32)
        normals = np.ones((5, 3), dtype=np.float32)
        albedo = np.ones((5, 3), dtype=np.float32)
        element_names = ["ground", "car_0", "car_1"]
        element_of_face = np.array([0, 1, 2, 1], dtype=np.int16)

        out = toy.filter_mesh_by_elements(
            verts, faces, colors, normals, albedo, element_of_face,
            element_names, ("car_0",))
        out_verts, out_faces, out_colors, out_normals, out_albedo, out_eof, \
            out_names, drop_counts = out

        self.assertIs(out_verts, verts)
        self.assertIs(out_colors, colors)
        self.assertIs(out_normals, normals)
        self.assertIs(out_albedo, albedo)
        np.testing.assert_array_equal(out_faces, faces[[0, 2]])
        np.testing.assert_array_equal(out_eof, np.array([0, 1], dtype=np.int16))
        self.assertEqual(out_names, ["ground", "car_1"])
        self.assertEqual(drop_counts, {"car_0": 2})
        self.assertEqual(toy.element_face_counts(out_eof, out_names),
                         {"ground": 1, "car_1": 1})

    def test_filter_noop_and_unknown(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        colors = np.zeros((3, 3), dtype=np.float32)
        normals = np.zeros((3, 3), dtype=np.float32)
        albedo = np.zeros((3, 3), dtype=np.float32)
        element_of_face = np.array([0], dtype=np.int16)
        element_names = ["ground"]

        out = toy.filter_mesh_by_elements(
            verts, faces, colors, normals, albedo, element_of_face,
            element_names, ())
        self.assertIs(out[0], verts)
        self.assertIs(out[1], faces)
        self.assertEqual(out[6], element_names)
        self.assertEqual(out[7], {})

        with self.assertRaises(ValueError):
            toy.filter_mesh_by_elements(
                verts, faces, colors, normals, albedo, element_of_face,
                element_names, ("car_0",))


if __name__ == "__main__":
    unittest.main()
