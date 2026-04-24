"""Merge two MeshFleet-style patch caches into a combined one.

The combined cache is an index-only directory:
  - patch_index.jsonl  (concatenated, absolute patch_file paths stay valid)
  - source_mesh_manifest.json  (merged records + metadata)
  - split_meshfleet_car.yaml  (combined town buckets)

No .npz files are copied — the trainer reads patch_file absolute paths
directly from the source caches, so disk usage stays near-zero.

Usage:
    python -m ss3dm_prior.tools.merge_patch_caches \
        --caches cacheA cacheB \
        --out_dir meshfleet_car_cache_v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rename_bucket(town_id: str, suffix: str) -> str:
    # Append a suffix so the combined split has disjoint town buckets per
    # source cache.
    return f"{town_id}__{suffix}" if suffix else town_id


def _apply_bucket_rename(rows: list[dict], suffix: str) -> list[dict]:
    if not suffix:
        return rows
    renamed = []
    for row in rows:
        row = dict(row)
        row["town_id"] = _rename_bucket(str(row.get("town_id", "")), suffix)
        renamed.append(row)
    return renamed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True, help="Source cache directories to merge, in order.")
    ap.add_argument("--suffixes", nargs="+", default=None, help="Optional per-cache town-id suffix; defaults to '' for first, 'ext{idx}' for rest.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    caches = [Path(c).expanduser().resolve() for c in args.caches]
    if args.suffixes is None:
        suffixes = [""] + [f"ext{i}" for i in range(1, len(caches))]
    else:
        assert len(args.suffixes) == len(caches), "suffixes must match caches count"
        suffixes = list(args.suffixes)

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. patch_index.jsonl
    all_rows: list[dict] = []
    for cache, sfx in zip(caches, suffixes):
        rows = _read_jsonl(cache / "patch_index.jsonl")
        rows = _apply_bucket_rename(rows, sfx)
        all_rows.extend(rows)
        print(f"[merge] {cache.name}: {len(rows)} rows (bucket_suffix={sfx or '(none)'})")
    _write_jsonl(out / "patch_index.jsonl", all_rows)
    print(f"[merge] combined: {len(all_rows)} rows -> {out / 'patch_index.jsonl'}")

    # 2. source_mesh_manifest.json
    merged_manifest: dict = {
        "dataset_name": "meshfleet_car_whole_mesh_combined",
        "dataset_root": str(caches[0]),
        "mesh_root": str(caches[0]),
        "generated_patch_index": str(out / "patch_index.jsonl"),
        "generated_split_config": str(out / "split_meshfleet_car.yaml"),
        "missing_mesh_count": 0,
        "records": [],
        "source_caches": [str(c) for c in caches],
    }
    for cache, sfx in zip(caches, suffixes):
        m = json.loads((cache / "source_mesh_manifest.json").read_text(encoding="utf-8"))
        merged_manifest["missing_mesh_count"] += int(m.get("missing_mesh_count", 0))
        for rec in m.get("records", []):
            rec = dict(rec)
            rec["source_cache"] = str(cache)
            rec["bucket_suffix"] = sfx
            merged_manifest["records"].append(rec)
    (out / "source_mesh_manifest.json").write_text(
        json.dumps(merged_manifest, indent=2), encoding="utf-8"
    )
    print(f"[merge] manifest records: {len(merged_manifest['records'])}")

    # 3. Combined split yaml — enumerate unique buckets in merged rows and
    #    classify by original name prefix (Train/Val/Test).
    import yaml  # type: ignore

    buckets_seen = set()
    for r in all_rows:
        buckets_seen.add(str(r["town_id"]))
    train_towns: list[str] = []
    val_towns: list[str] = []
    test_towns: list[str] = []
    for b in sorted(buckets_seen):
        head = b.split("__")[0]
        if "Test" in head:
            test_towns.append(b)
        elif "Val" in head:
            val_towns.append(b)
        else:
            train_towns.append(b)
    split_cfg = {
        "split_name": "meshfleet_car_whole_mesh_combined",
        "strategy": "preassigned_car_split",
        "unit_of_split": "car_mesh",
        "forbid_random_patch_split": True,
        "forbid_random_frame_split": True,
        "train_towns": train_towns,
        "val_towns": val_towns,
        "test_towns": test_towns,
        "notes": [
            "Combined MeshFleet v4 + Objaverse 1.0 vehicle extension (v0.2).",
            "town_id is a split bucket; each source cache contributes its own suffixed bucket set.",
        ],
    }
    (out / "split_meshfleet_car.yaml").write_text(
        yaml.safe_dump(split_cfg, sort_keys=False), encoding="utf-8"
    )
    print(f"[merge] split: train={len(train_towns)} val={len(val_towns)} test={len(test_towns)}")
    print(f"[done] merged cache at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
