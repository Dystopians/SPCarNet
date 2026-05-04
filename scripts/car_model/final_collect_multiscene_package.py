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
        "best_method": "Open3D QEM50 strict recovery 26k",
        "best_triangles": 44230,
        "best_psnr": 11.082405,
        "best_ssim": 0.243249,
        "best_lpips": 0.570177,
        "best_absrel": 0.182966,
        "best_depth_mae": 1.793852,
        "best_normal": 42.889339,
        "wandb": "bsed9ik1",
        "evidence": "docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md",
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
        "best_method": "Open3D QEM50 strict recovery 26k",
        "best_triangles": 42253,
        "best_psnr": 15.061190,
        "best_ssim": 0.481082,
        "best_lpips": 0.516805,
        "best_absrel": 0.181129,
        "best_depth_mae": 1.345221,
        "best_normal": 54.900779,
        "wandb": "9wri3owt",
        "evidence": "docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md",
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
        "best_method": "Open3D QEM40 strict recovery 26k",
        "best_triangles": 50300,
        "best_psnr": 14.409434,
        "best_ssim": 0.547456,
        "best_lpips": 0.420855,
        "best_absrel": 0.068076,
        "best_depth_mae": 0.338664,
        "best_normal": 43.716007,
        "wandb": "kr8565st",
        "evidence": "docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md",
        "decision": "PASS_PARETO",
    },
]

NEGATIVE_ROWS = [
    {"scene": "bonsai", "row": "CSEF70", "finding": "70 percent compaction fails SSIM gate", "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md"},
    {"scene": "bonsai", "row": "CSEF50", "finding": "passes clean-long but is superseded by Open3D QEM50 on render, AbsRel, and normal", "evidence": "docs/car_model/final_stageF22_bonsai_posthoc_qem_baseline_report.md"},
    {"scene": "courtyard", "row": "Open3D QEM50", "finding": "improves SSIM, LPIPS, and normal but is weaker than CSEF50 on PSNR, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF23_courtyard_posthoc_qem_baseline_report.md"},
    {"scene": "counter", "row": "CSEF50", "finding": "50 percent compaction is a boundary case and misses SSIM by 0.003827", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "counter", "row": "CSEF50 30k", "finding": "extended recovery worsens SSIM and LPIPS", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "counter", "row": "random40", "finding": "same-count random compaction loses badly to area40 and CSEF40", "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md"},
    {"scene": "counter", "row": "CSEF40", "finding": "passes clean-long but is not the strongest selector on counter; area40 and QEM40 are better", "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md"},
    {"scene": "counter", "row": "area40", "finding": "strong structured selector row but superseded by Open3D QEM40 on render, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md"},
    {"scene": "counter", "row": "area40 no-freeze", "finding": "omitting strict topology freeze collapses topology to 18,693 triangles and loses badly to frozen area40", "evidence": "docs/car_model/final_stageF18_counter_no_freeze_control_report.md"},
    {"scene": "room", "row": "QEM50 no-freeze", "finding": "omitting strict topology freeze collapses topology to 20,742 triangles and loses badly to frozen QEM50", "evidence": "docs/car_model/final_stageF24_room_qem_no_freeze_control_report.md"},
    {"scene": "room", "row": "random50", "finding": "same-count random compaction loses badly to area50 and clean-long", "evidence": "docs/car_model/final_stageF19_room_selector_ablation_report.md"},
    {"scene": "room", "row": "CSEF50", "finding": "passes clean-long but is superseded by area50 on all tracked independent metrics", "evidence": "docs/car_model/final_stageF19_room_selector_ablation_report.md"},
    {"scene": "room", "row": "area50", "finding": "strong structured selector row but superseded by Open3D QEM50 on render, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md"},
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
        "PASS with ablation gaps. At least two scenes show meaningful compact-recovery benefit over fair clean-long baselines; five scenes now have auditable long-baseline comparisons. The remaining NeurIPS risk is not scene count, but missing matched ablations against area-only, random same-count compaction beyond the completed controls, further replicated no-freeze controls beyond the completed counter/room controls, explicit sparse-depth-loss variants if claimed, and posthoc simplification controls beyond the completed bonsai/courtyard/room/counter QEM rows.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "main_quantitative_table.csv")
    print(OUT / "negative_result_table.csv")
    print(DOC)


if __name__ == "__main__":
    main()
