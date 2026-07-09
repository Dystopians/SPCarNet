#!/usr/bin/env python
"""GEMS Stage-4 L5 report: quality-vs-TOTAL-artifact-MB Pareto (GOAL #E-06).

Reads the banked l5_<scene>_<variant>_v1 rows plus the uncompressed
final-stack row per scene and emits the curve table (markdown + json) under
analysis/final_stack/.
"""
import json
import os
import sys

G1 = "/data/peilincai/gems_stage1"
SCENES = ["garden", "bicycle", "kitchen"]
VARIANTS = ["jpeg95", "jpeg85", "jpeg70", "halfres", "ksubset50"]


def load_row(name):
    path = os.path.join(G1, "eval", name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    sys.path.insert(0, "/data/peilincai/mesh-splatting")
    out_dir = os.path.join(G1, "analysis", "final_stack")
    os.makedirs(out_dir, exist_ok=True)
    md = ["# L5 cache Pareto — quality vs TOTAL artifact MB (final stack)",
          "", "| scene | point | PSNR | LPIPS | cache MB raw/comp | ckpt MB |"
          " TOTAL raw MB | dPSNR vs uncompressed |", "|---|---|---|---|---|---|---|---|"]
    report = {}
    for scene in SCENES:
        base_row = load_row(f"l4_{scene}_cleanfixed30k_routed_v1")
        rows = {"uncompressed": base_row}
        for v in VARIANTS:
            rows[v] = load_row(f"l5_{scene}_{v}_v1")
        base_psnr = (base_row["rendering"]["mean"]["psnr"]
                     if base_row else None)
        report[scene] = {}
        for label, row in rows.items():
            if row is None:
                md.append(f"| {scene} | {label} | PENDING | | | | | |")
                continue
            c = row["cost"]
            p = row["rendering"]["mean"]["psnr"]
            entry = {
                "psnr": p,
                "lpips": row["rendering"]["mean"]["lpips"],
                "cache_mb_raw": c.get("cache_mb_raw"),
                "cache_mb_compressed": c.get("cache_mb_compressed"),
                "ckpt_mb": c.get("disk_mb"),
                "total_mb_raw": c.get("total_artifact_mb"),
                "dpsnr_vs_uncompressed": (p - base_psnr) if base_psnr else None,
            }
            report[scene][label] = entry
            d = entry["dpsnr_vs_uncompressed"]
            md.append(
                f"| {scene} | {label} | {p:.3f} "
                f"| {entry['lpips']:.4f} "
                f"| {entry['cache_mb_raw']:.0f}/{entry['cache_mb_compressed']:.0f} "
                f"| {entry['ckpt_mb']:.0f} | {entry['total_mb_raw']:.0f} "
                f"| {'—' if d is None or label == 'uncompressed' else f'{d:+.3f}'} |")
    md += ["", "TOTAL = checkpoint + raw cache (the on-disk-usable artifact);"
           " lossless-compressed cache size also listed (shippable form).",
           "CIs for any headline point: rerun tools/ecr/rung_gate.py on the"
           " chosen pair."]
    path = os.path.join(out_dir, "l5_pareto.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "l5_pareto.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n".join(md))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
