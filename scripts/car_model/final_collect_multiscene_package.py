#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/carnet/meshsplatopt/final_multiscene_package"
DOC = ROOT / "docs/car_model/final_stageF12_multiscene_package_report.md"


SCENES = [
    {
        "scene": "parking_phone_tiny",
        "clean_method": "clean-long 22k",
        "clean_triangles": 8548242,
        "clean_psnr": 18.480,
        "clean_ssim": 0.635,
        "clean_lpips": 0.347,
        "clean_absrel": 0.082,
        "clean_depth_mae": 1.868,
        "clean_normal": 45.108,
        "best_method": "R53.01 area70 strict recovery 26k",
        "best_triangles": 2564473,
        "best_psnr": 18.706,
        "best_ssim": 0.648,
        "best_lpips": 0.338,
        "best_absrel": 0.080,
        "best_depth_mae": 1.854,
        "best_normal": 44.261,
        "wandb": "q15qg2b8",
        "evidence": "docs/car_model/parking_clean_to_compact_repair_report.md",
        "decision": "PASS",
    },
    {
        "scene": "bonsai",
        "clean_method": "clean-long 22k",
        "clean_triangles": 88460,
        "clean_psnr": 10.944348,
        "clean_ssim": 0.222848,
        "clean_lpips": 0.586158,
        "clean_absrel": 0.194249,
        "clean_depth_mae": 1.816410,
        "clean_normal": 45.358356,
        "best_method": "CSEF50 strict recovery 26k",
        "best_triangles": 44230,
        "best_psnr": 10.957497,
        "best_ssim": 0.224758,
        "best_lpips": 0.586415,
        "best_absrel": 0.185180,
        "best_depth_mae": 1.737815,
        "best_normal": 43.493975,
        "wandb": "irdsa4c8",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "decision": "PASS",
    },
    {
        "scene": "courtyard",
        "clean_method": "clean-long 22k",
        "clean_triangles": 1677484,
        "clean_psnr": 12.103508,
        "clean_ssim": 0.296648,
        "clean_lpips": 0.569308,
        "clean_absrel": 0.354648,
        "clean_depth_mae": 3.829044,
        "clean_normal": 40.821649,
        "best_method": "CSEF50 strict recovery 26k",
        "best_triangles": 838742,
        "best_psnr": 12.555809,
        "best_ssim": 0.338273,
        "best_lpips": 0.545077,
        "best_absrel": 0.322233,
        "best_depth_mae": 3.608432,
        "best_normal": 40.830157,
        "wandb": "jz93wrbc",
        "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
        "decision": "PASS",
    },
    {
        "scene": "room",
        "clean_method": "clean-long 22k",
        "clean_triangles": 84506,
        "clean_psnr": 14.258379,
        "clean_ssim": 0.400864,
        "clean_lpips": 0.578919,
        "clean_absrel": 0.206282,
        "clean_depth_mae": 1.480230,
        "clean_normal": 55.442653,
        "best_method": "CSEF50 strict recovery 26k",
        "best_triangles": 42253,
        "best_psnr": 14.387163,
        "best_ssim": 0.414954,
        "best_lpips": 0.568281,
        "best_absrel": 0.225027,
        "best_depth_mae": 1.603030,
        "best_normal": 54.642793,
        "wandb": "pb1tg4p2",
        "evidence": "docs/car_model/final_stageF9_third_scene_room_and_qualitative_report.md",
        "decision": "PASS",
    },
    {
        "scene": "counter",
        "clean_method": "clean-long 22k",
        "clean_triangles": 83834,
        "clean_psnr": 14.136182,
        "clean_ssim": 0.512802,
        "clean_lpips": 0.452049,
        "clean_absrel": 0.076996,
        "clean_depth_mae": 0.369973,
        "clean_normal": 44.287035,
        "best_method": "CSEF40 strict recovery 26k",
        "best_triangles": 50300,
        "best_psnr": 14.212033,
        "best_ssim": 0.518401,
        "best_lpips": 0.450481,
        "best_absrel": 0.085542,
        "best_depth_mae": 0.406373,
        "best_normal": 43.476972,
        "wandb": "glzzth4b",
        "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md",
        "decision": "PASS_PARETO",
    },
]

NEGATIVE_ROWS = [
    {"scene": "bonsai", "row": "CSEF70", "finding": "70 percent compaction fails SSIM gate", "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md"},
    {"scene": "counter", "row": "CSEF50", "finding": "50 percent compaction is a boundary case and misses SSIM by 0.003827", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "counter", "row": "CSEF50 30k", "finding": "extended recovery worsens SSIM and LPIPS", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "parking_phone_tiny", "row": "grid fill full-budget", "finding": "fill branch does not beat matched sparse-depth control", "evidence": "docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md"},
]


def enrich(row: dict) -> dict:
    out = dict(row)
    out["reduction"] = 1.0 - row["best_triangles"] / row["clean_triangles"]
    out["d_psnr"] = row["best_psnr"] - row["clean_psnr"]
    out["d_ssim"] = row["best_ssim"] - row["clean_ssim"]
    out["d_lpips"] = row["best_lpips"] - row["clean_lpips"]
    out["d_absrel"] = row["best_absrel"] - row["clean_absrel"]
    out["d_depth_mae"] = row["best_depth_mae"] - row["clean_depth_mae"]
    out["d_normal"] = row["best_normal"] - row["clean_normal"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [enrich(row) for row in SCENES]
    write_csv(OUT / "main_quantitative_table.csv", rows)
    write_csv(OUT / "negative_result_table.csv", NEGATIVE_ROWS)
    (OUT / "main_quantitative_table.json").write_text(json.dumps(rows, indent=2))

    pass_count = sum(1 for row in rows if row["decision"].startswith("PASS"))
    render_win_count = sum(1 for row in rows if row["d_psnr"] > 0 and row["d_ssim"] > 0 and row["d_lpips"] <= 0.002)
    lines = [
        "# Final Stage F12 - Multi-Scene Package Report",
        "",
        "Decision: `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS`.",
        "",
        f"Scenes with compact-recovery pass decisions: `{pass_count}/{len(rows)}`.",
        f"Scenes with PSNR+SSIM improvement and LPIPS non-regression tolerance: `{render_win_count}/{len(rows)}`.",
        "",
        "## Main Quantitative Table",
        "",
        "| scene | clean triangles | ours triangles | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['clean_triangles']:,} | {row['best_triangles']:,} | {row['reduction']*100:.1f}% | {fmt(row['d_psnr'])} | {fmt(row['d_ssim'])} | {fmt(row['d_lpips'])} | {fmt(row['d_absrel'])} | {fmt(row['d_depth_mae'])} | {fmt(row['d_normal'])} | {row['decision']} |"
        )
    lines += [
        "",
        "## Per-Scene Evidence",
        "",
        "| scene | clean row | best row | W&B | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['scene']} | {row['clean_method']} | {row['best_method']} | `{row['wandb']}` | `{row['evidence']}` |")
    lines += [
        "",
        "## Negative Result Table",
        "",
        "| scene | row | finding | evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in NEGATIVE_ROWS:
        lines.append(f"| {row['scene']} | {row['row']} | {row['finding']} | `{row['evidence']}` |")
    lines += [
        "",
        "## Gate",
        "",
        "PASS with ablation gaps. At least two scenes show meaningful compact-recovery benefit over fair clean-long baselines; five scenes now have auditable long-baseline comparisons. The remaining NeurIPS risk is not scene count, but missing matched ablations against area-only, random same-count compaction, no-sparse-depth, no-freeze, and posthoc simplification controls.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "main_quantitative_table.csv")
    print(OUT / "negative_result_table.csv")
    print(DOC)


if __name__ == "__main__":
    main()

