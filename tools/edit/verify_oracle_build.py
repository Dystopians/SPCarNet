#!/usr/bin/env python
"""Verify a toy_parking element-drop oracle build against its original build."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def same_bytes(a: Path, b: Path) -> bool:
    if not a.is_file() or not b.is_file():
        return False
    return a.read_bytes() == b.read_bytes()


class Results:
    def __init__(self):
        self.failed = False

    def pass_(self, name: str, detail: str):
        print(f"PASS {name}: {detail}")

    def fail(self, name: str, detail: str):
        self.failed = True
        print(f"FAIL {name}: {detail}")

    def skip(self, name: str, detail: str):
        print(f"SKIP {name}: {detail}")


def parse_name_list(values) -> list[str]:
    out = []
    for value in values or []:
        for raw in str(value).split(","):
            item = raw.strip()
            if item:
                out.append(item)
    return out


def normalized_view_name(name: str) -> str:
    base = os.path.basename(str(name))
    stem, ext = os.path.splitext(base)
    return stem if ext else base


def image_path(root: Path, view: str) -> Path:
    base = os.path.basename(str(view))
    if os.path.splitext(base)[1]:
        return root / "images" / base
    return root / "images" / f"{base}.png"


def compare_required_files(original: Path, oracle: Path, rels: list[str],
                           results: Results, check_name: str):
    missing = []
    mismatched = []
    for rel in rels:
        a = original / rel
        b = oracle / rel
        if not a.is_file() or not b.is_file():
            missing.append(rel)
        elif not same_bytes(a, b):
            mismatched.append(rel)
    if missing or mismatched:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if mismatched:
            detail.append(f"mismatched={mismatched}")
        results.fail(check_name, "; ".join(detail))
    else:
        results.pass_(check_name, f"byte-identical: {rels}")


def split_files_to_compare(original: Path, oracle: Path) -> tuple[list[str], list[str]]:
    names = ["split_test.txt", "split.json"]
    present = [name for name in names if (original / name).exists() or
               (oracle / name).exists()]
    missing = [name for name in present
               if not (original / name).is_file() or not (oracle / name).is_file()]
    return present, missing


def load_split_tests(root: Path) -> list[str]:
    for rel in ("split_test.txt", "split.json"):
        path = root / rel
        if path.is_file():
            payload = load_json(path)
            return [normalized_view_name(v) for v in payload.get("test", [])]
    return []


def load_element_counts(root: Path) -> dict[str, int]:
    sidecar = root / "gt" / "mesh_colors.npz"
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing mesh sidecar: {sidecar}")
    with np.load(sidecar, allow_pickle=False) as data:
        names = [str(x) for x in data["element_names"].tolist()]
        eof = np.asarray(data["element_of_face"], dtype=np.int64)
    counts = np.bincount(eof, minlength=len(names))
    return {name: int(counts[i]) for i, name in enumerate(names)}


def check_manifest_and_face_counts(original: Path, oracle: Path, dropped: list[str],
                                   results: Results):
    try:
        orig_man = load_json(original / "dataset_manifest.json")
        oracle_man = load_json(oracle / "dataset_manifest.json")
    except Exception as exc:
        results.fail("manifest_face_counts", f"could not read manifests: {exc}")
        return

    if not dropped:
        dropped = [str(x) for x in oracle_man.get("dropped_elements", [])]
    if not dropped:
        results.fail("manifest_face_counts",
                     "oracle manifest has no dropped_elements and none were provided")
        return

    oracle_elements = [str(x) for x in oracle_man.get("elements", [])]
    leaked = [name for name in dropped if name in oracle_elements]
    if leaked:
        results.fail("manifest_face_counts",
                     f"oracle manifest still lists dropped elements: {leaked}")
        return

    try:
        orig_counts = load_element_counts(original)
        oracle_counts = load_element_counts(oracle)
    except Exception as exc:
        results.fail("manifest_face_counts", str(exc))
        return

    missing_original = [name for name in dropped if name not in orig_counts]
    if missing_original:
        results.fail("manifest_face_counts",
                     f"dropped elements absent from original sidecar: {missing_original}")
        return

    count_errors = []
    for name, count in sorted(orig_counts.items()):
        if name in dropped:
            if oracle_counts.get(name, 0) != 0:
                count_errors.append(
                    f"{name}: oracle has {oracle_counts.get(name)} dropped faces")
        elif oracle_counts.get(name) != count:
            count_errors.append(
                f"{name}: original={count} oracle={oracle_counts.get(name)}")

    dropped_faces = sum(orig_counts[name] for name in dropped)
    face_delta = sum(orig_counts.values()) - sum(oracle_counts.values())
    if face_delta != dropped_faces:
        count_errors.append(
            f"face delta={face_delta}, expected dropped faces={dropped_faces}")

    if count_errors:
        results.fail("manifest_face_counts", "; ".join(count_errors))
    else:
        results.pass_(
            "manifest_face_counts",
            f"dropped={dropped}, removed_faces={dropped_faces}, other counts unchanged")


def count_from_record(record, element: str, elements: list[str] | None = None):
    if isinstance(record, dict):
        if element in record:
            return int(record[element])
        for key in ("counts", "coverage", "pixels", "elements"):
            if key in record:
                got = count_from_record(record[key], element, elements)
                if got is not None:
                    return got
        return None
    if elements is not None and isinstance(record, list) and element in elements:
        idx = elements.index(element)
        if idx < len(record):
            return int(record[idx])
    return None


def extract_per_view_counts(manifest: dict, element: str) -> dict[str, int] | None:
    candidate_keys = (
        "coverage_per_view",
        "per_view_coverage",
        "coverage_census",
        "per_view_coverage_census",
    )
    for key in candidate_keys:
        if key not in manifest:
            continue
        block = manifest[key]
        elements = [str(x) for x in block.get("elements", [])] \
            if isinstance(block, dict) else None
        if isinstance(block, dict):
            for view_key in ("views", "per_view", "counts_by_view"):
                if view_key in block:
                    block = block[view_key]
                    break
        out = {}
        if isinstance(block, dict):
            for view, record in block.items():
                count = count_from_record(record, element, elements)
                if count is not None:
                    out[normalized_view_name(view)] = count
        elif isinstance(block, list):
            for record in block:
                if not isinstance(record, dict):
                    continue
                view = (record.get("name") or record.get("view") or
                        record.get("image") or record.get("image_name"))
                if view is None:
                    continue
                count = count_from_record(record, element, elements)
                if count is not None:
                    out[normalized_view_name(view)] = count
        if out:
            return out
    return None


def compare_images_for_views(original: Path, oracle: Path, views: list[str],
                             results: Results, check_name: str):
    mismatched = []
    missing = []
    for view in views:
        a = image_path(original, view)
        b = image_path(oracle, view)
        if not a.is_file() or not b.is_file():
            missing.append(view)
        elif not same_bytes(a, b):
            mismatched.append(view)
    if missing or mismatched:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if mismatched:
            detail.append(f"mismatched={mismatched}")
        results.fail(check_name, "; ".join(detail))
    else:
        results.pass_(check_name, f"{len(views)} image(s) byte-identical")


def check_zero_pixel_images(original: Path, oracle: Path, dropped: list[str],
                            expect_identical: list[str], results: Results):
    element = dropped[0] if dropped else "car_0"
    try:
        manifest = load_json(original / "dataset_manifest.json")
    except Exception as exc:
        results.fail("zero_pixel_images", f"could not read original manifest: {exc}")
        return
    per_view = extract_per_view_counts(manifest, element)
    test_views = set(load_split_tests(original))
    if per_view:
        views = sorted(v for v, count in per_view.items()
                       if count == 0 and (not test_views or v in test_views))
        compare_images_for_views(original, oracle, views, results,
                                 "zero_pixel_images")
        return

    results.skip(
        "zero_pixel_images",
        "original dataset_manifest.json has no per-view element coverage detail")
    expected = [normalized_view_name(v) for v in expect_identical]
    if len(expected) != 3:
        results.fail(
            "expect_identical_images",
            "per-view census absent; pass exactly 3 views with --expect-identical")
        return
    compare_images_for_views(original, oracle, expected, results,
                             "expect_identical_images")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--dropped-elements", default="",
                    help="optional comma-separated override; defaults to the "
                         "oracle manifest's dropped_elements")
    ap.add_argument("--expect-identical", action="append", default=[],
                    help="fallback view name(s), comma-separated or repeatable, "
                         "used when per-view coverage is absent")
    args = ap.parse_args()

    original = Path(args.original)
    oracle = Path(args.oracle)
    results = Results()

    compare_required_files(
        original, oracle,
        ["sparse/0/images.txt", "sparse/0/cameras.txt"],
        results, "colmap_camera_text")

    split_rels, split_missing = split_files_to_compare(original, oracle)
    if split_missing:
        results.fail("split_files", f"missing={split_missing}")
    elif not split_rels:
        results.fail("split_files", "no split_test.txt or split.json found")
    else:
        compare_required_files(original, oracle, split_rels, results, "split_files")

    dropped = parse_name_list([args.dropped_elements])
    if not dropped:
        try:
            oracle_man = load_json(oracle / "dataset_manifest.json")
            dropped = [str(x) for x in oracle_man.get("dropped_elements", [])]
        except Exception:
            dropped = []
    check_manifest_and_face_counts(original, oracle, dropped, results)
    check_zero_pixel_images(
        original, oracle, dropped, parse_name_list(args.expect_identical), results)
    return 1 if results.failed else 0


if __name__ == "__main__":
    sys.exit(main())
