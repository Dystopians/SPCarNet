#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat


def load_extrinsics(scene_path: Path):
    bin_path = scene_path / "sparse/0/images.bin"
    txt_path = scene_path / "sparse/0/images.txt"
    if bin_path.exists():
        extrinsics = read_extrinsics_binary(str(bin_path))
    elif txt_path.exists():
        extrinsics = read_extrinsics_text(str(txt_path))
    else:
        raise FileNotFoundError(f"Cannot find COLMAP images.bin/images.txt under {scene_path}/sparse/0")
    return extrinsics


def camera_center_from_extrinsic(extr):
    # COLMAP stores world->camera as (R, t); camera center is -R^T t.
    r_wc = qvec2rotmat(extr.qvec)
    t_wc = np.array(extr.tvec, dtype=np.float64)
    center = -r_wc.T @ t_wc
    return center


def build_spatial_ball_split(names, centers, test_ratio, gap_ratio):
    n = len(names)
    test_count = max(1, int(round(n * test_ratio)))
    gap_count = max(0, int(round(n * gap_ratio)))

    best = None
    for anchor in range(n):
        d = np.linalg.norm(centers - centers[anchor], axis=1)
        order = np.argsort(d)
        test_idx = order[:test_count]
        dropped_idx = order[test_count : test_count + gap_count]
        train_idx = order[test_count + gap_count :]

        if len(train_idx) == 0:
            continue

        # Higher margin means test is more separated from train.
        cross = np.linalg.norm(
            centers[test_idx][:, None, :] - centers[train_idx][None, :, :], axis=2
        )
        min_cross = float(cross.min())
        med_cross = float(np.median(cross.min(axis=1)))
        score = (min_cross, med_cross)
        if best is None or score > best["score"]:
            best = {
                "anchor": int(anchor),
                "test_idx": test_idx,
                "dropped_idx": dropped_idx,
                "train_idx": train_idx,
                "score": score,
                "test_count": len(test_idx),
                "dropped_count": len(dropped_idx),
                "train_count": len(train_idx),
            }

    if best is None:
        raise RuntimeError("Failed to build split with non-empty train/test.")

    test_set = set(int(i) for i in best["test_idx"])
    dropped_set = set(int(i) for i in best["dropped_idx"])
    train_set = set(int(i) for i in best["train_idx"])
    assert len(test_set & dropped_set) == 0
    assert len(test_set & train_set) == 0
    assert len(dropped_set & train_set) == 0

    result = {
        "strategy": "spatial_ball_with_gap",
        "anchor_image": names[best["anchor"]],
        "counts": {
            "total": n,
            "train": best["train_count"],
            "test": best["test_count"],
            "dropped": best["dropped_count"],
        },
        "separation": {
            "min_test_to_train_distance": best["score"][0],
            "median_test_to_train_distance": best["score"][1],
        },
        "train": [Path(names[i]).stem for i in sorted(train_set)],
        "test": [Path(names[i]).stem for i in sorted(test_set)],
        "dropped": [Path(names[i]).stem for i in sorted(dropped_set)],
    }
    return result


def summarize_llff_separation(centers, llffhold=8):
    idx = np.arange(len(centers))
    test = idx[idx % llffhold == 0]
    train = idx[idx % llffhold != 0]
    if len(test) == 0 or len(train) == 0:
        return None
    cross = np.linalg.norm(centers[test][:, None, :] - centers[train][None, :, :], axis=2)
    return {
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "min_test_to_train_distance": float(cross.min()),
        "median_test_to_train_distance": float(np.median(cross.min(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description="Create out-of-train split from COLMAP camera poses.")
    parser.add_argument("-s", "--source_path", required=True, type=str, help="COLMAP scene root path")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output split json path")
    parser.add_argument("--test_ratio", type=float, default=0.12, help="Fraction of cameras used as test")
    parser.add_argument(
        "--gap_ratio",
        type=float,
        default=0.03,
        help="Fraction of cameras dropped as train-test spatial buffer",
    )
    args = parser.parse_args()

    scene_path = Path(args.source_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extrinsics = load_extrinsics(scene_path)
    cams = sorted(extrinsics.values(), key=lambda e: Path(e.name).stem)
    names = [c.name for c in cams]
    centers = np.stack([camera_center_from_extrinsic(c) for c in cams], axis=0)

    split = build_spatial_ball_split(
        names=names,
        centers=centers,
        test_ratio=args.test_ratio,
        gap_ratio=args.gap_ratio,
    )
    llff = summarize_llff_separation(centers, llffhold=8)
    split["baseline_llff_hold8"] = llff

    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"[Split] Wrote split json: {output_path}")
    print(f"[Split] Counts: {split['counts']}")
    print(f"[Split] Separation: {split['separation']}")
    if llff is not None:
        print(
            "[Split] LLFF hold8 separation (for reference): "
            f"min={llff['min_test_to_train_distance']:.4f}, "
            f"median={llff['median_test_to_train_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
