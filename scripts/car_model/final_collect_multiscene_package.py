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
        "best_method": "CSEF70 + sparse-depth strict recovery 26k",
        "best_triangles": 2564473,
        "best_psnr": 18.712330,
        "best_ssim": 0.647730,
        "best_lpips": 0.338259,
        "best_absrel": 0.079071,
        "best_depth_mae": 1.854015,
        "best_normal": 44.035708,
        "wandb": "x6rmhhlp",
        "evidence": "docs/car_model/final_stageF33_parking_csef_sparse_depth_report.md",
        "decision": "PASS_PARETO",
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
        "best_method": "Open3D QEM50 + sparse-depth strict recovery 26k",
        "best_triangles": 44230,
        "best_psnr": 11.081614,
        "best_ssim": 0.243248,
        "best_lpips": 0.569658,
        "best_absrel": 0.181698,
        "best_depth_mae": 1.779783,
        "best_normal": 42.425734,
        "wandb": "07k1ii1d",
        "evidence": "docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md",
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
        "best_method": "Open3D QEM40 + sparse-depth strict recovery 26k",
        "best_triangles": 50300,
        "best_psnr": 14.408769,
        "best_ssim": 0.547570,
        "best_lpips": 0.420202,
        "best_absrel": 0.068014,
        "best_depth_mae": 0.339115,
        "best_normal": 43.585215,
        "wandb": "x9b89ssf",
        "evidence": "docs/car_model/final_stageF32_counter_qem_sparse_depth_report.md",
        "decision": "PASS_PARETO",
    },
]

NEGATIVE_ROWS = [
    {"scene": "bonsai", "row": "CSEF70", "finding": "70 percent compaction fails SSIM gate", "evidence": "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md"},
    {"scene": "bonsai", "row": "CSEF50", "finding": "passes clean-long but is superseded by QEM/sparse-depth on render and by area50 or sparse-depth QEM on geometry/perceptual metrics", "evidence": "docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md"},
    {"scene": "bonsai", "row": "QEM50 frozen", "finding": "strong render row, but QEM50 + sparse-depth is better on LPIPS, AbsRel, Depth MAE, and normal at the same topology with negligible PSNR/SSIM cost", "evidence": "docs/car_model/final_stageF28_bonsai_qem_sparse_depth_report.md"},
    {"scene": "bonsai", "row": "QEM50 no-freeze", "finding": "omitting strict topology freeze collapses topology to 17,962 triangles and loses badly to frozen QEM50", "evidence": "docs/car_model/final_stageF27_bonsai_qem_no_freeze_control_report.md"},
    {"scene": "bonsai", "row": "area50", "finding": "strong structured Pareto row: slightly behind QEM50 on PSNR/SSIM, but better on LPIPS, AbsRel, Depth MAE, and normal", "evidence": "docs/car_model/final_stageF26_bonsai_selector_ablation_report.md"},
    {"scene": "bonsai", "row": "random50", "finding": "same-count random compaction loses badly to clean-long, area50, CSEF50, and QEM50 on render and AbsRel", "evidence": "docs/car_model/final_stageF26_bonsai_selector_ablation_report.md"},
    {"scene": "courtyard", "row": "Open3D QEM50", "finding": "improves SSIM, LPIPS, and normal but is weaker than CSEF50 on PSNR, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF23_courtyard_posthoc_qem_baseline_report.md"},
    {"scene": "courtyard", "row": "CSEF50 + sparse-depth", "finding": "fixes the courtyard normal regression and slightly improves AbsRel, but gives back small PSNR, LPIPS, and Depth MAE margins, so CSEF50 remains the balanced headline", "evidence": "docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md"},
    {"scene": "courtyard", "row": "QEM50 + sparse-depth", "finding": "improves QEM50 on PSNR, SSIM, AbsRel, and Depth MAE, but remains weaker than CSEF50 on PSNR and sparse depth", "evidence": "docs/car_model/final_stageF30_F31_courtyard_sparse_depth_controls_report.md"},
    {"scene": "courtyard", "row": "CSEF50 no-freeze", "finding": "omitting strict topology freeze makes CSEF50 drift to 1,317,435 triangles and lose badly on render and sparse depth, supporting the frozen recovery contract", "evidence": "docs/car_model/final_stageF35_courtyard_csef_no_freeze_control_report.md"},
    {"scene": "counter", "row": "CSEF50", "finding": "50 percent compaction is a boundary case and misses SSIM by 0.003827", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "counter", "row": "CSEF50 30k", "finding": "extended recovery worsens SSIM and LPIPS", "evidence": "docs/car_model/final_stageF10_fourth_scene_counter_report.md"},
    {"scene": "counter", "row": "random40", "finding": "same-count random compaction loses badly to area40 and CSEF40", "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md"},
    {"scene": "counter", "row": "CSEF40", "finding": "passes clean-long but is not the strongest selector on counter; area40 and QEM40 are better", "evidence": "docs/car_model/final_stageF16_counter_random_same_count_ablation_report.md"},
    {"scene": "counter", "row": "area40", "finding": "strong structured selector row but superseded by Open3D QEM40 on render, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF21_counter_posthoc_qem_baseline_report.md"},
    {"scene": "counter", "row": "QEM40 frozen", "finding": "strong PSNR/Depth row, but QEM40 + sparse-depth improves SSIM, LPIPS, AbsRel, and normal at the same topology with negligible PSNR/Depth cost", "evidence": "docs/car_model/final_stageF32_counter_qem_sparse_depth_report.md"},
    {"scene": "counter", "row": "area40 no-freeze", "finding": "omitting strict topology freeze collapses topology to 18,693 triangles and loses badly to frozen area40", "evidence": "docs/car_model/final_stageF18_counter_no_freeze_control_report.md"},
    {"scene": "parking_phone_tiny", "row": "R53.01 area70", "finding": "strong area-only control at the same triangle count; superseded by CSEF70 on PSNR, LPIPS, AbsRel, Depth MAE, and normal with negligible SSIM loss", "evidence": "docs/car_model/final_stageF7_parking_pareto_report.md"},
    {"scene": "parking_phone_tiny", "row": "CSEF70", "finding": "strong all-metric clean-long win, but CSEF70 + sparse-depth improves PSNR, LPIPS, AbsRel, and normal at identical topology with negligible SSIM cost and a small Depth MAE tradeoff", "evidence": "docs/car_model/final_stageF33_parking_csef_sparse_depth_report.md"},
    {"scene": "parking_phone_tiny", "row": "F34 CSEF70 sparse-depth 30k continuation", "finding": "longer fixed-topology sparse-depth continuation slightly improves sparse depth proxies but sharply worsens PSNR, SSIM, LPIPS, and normal, so F33 26k remains the validated parking row", "evidence": "docs/car_model/final_stageF34_parking_long_continuation_report.md"},
    {"scene": "parking_phone_tiny", "row": "CSEF70 no-freeze", "finding": "omitting strict topology freeze makes CSEF70 drift to 3,533,325 triangles and lose badly to frozen CSEF70/sparse-depth on render and sparse depth", "evidence": "docs/car_model/final_stageF36_parking_csef_no_freeze_control_report.md"},
    {"scene": "parking_phone_tiny", "row": "fast-QEM70 matched", "finding": "matched QEM reaches the F7/F33 topology within 9 triangles and improves sparse geometry proxies, but collapses PSNR, SSIM, and LPIPS relative to clean-long and F33", "evidence": "docs/car_model/final_stageF37_parking_fast_qem_matched_baseline_report.md"},
    {"scene": "parking_phone_tiny", "row": "Open3D QEM70", "finding": "Open3D QEM did not reach the matched 2,564,473-triangle target on the 8.55M-triangle parking mesh, stopping at 8,125,970 triangles, so it is rejected as an unmatched compression control", "evidence": "docs/car_model/final_stageF25_parking_posthoc_qem_baseline_report.md"},
    {"scene": "synthetic_edit_suite", "row": "no-gate/no-rollback counterfactual", "finding": "F38 applies identical bad edits with and without MeshSplatOpt gate/rollback; the gate rejects and exactly restores all unsafe edits, while the unsafe path commits an unobserved floater, a 5m free-space snap, and supported-surface deletion", "evidence": "docs/car_model/final_stageF38_counterfactual_gate_ablation_report.md"},
    {"scene": "parking_phone_tiny", "row": "real gate-removed ratio0.04", "finding": "F39 same-schedule 500-iteration ablation shows the gated run rolls back a 2579-triangle candidate set while the gate-removed run commits it and is slightly worse on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE", "evidence": "docs/car_model/final_stageF39_real_gate_removed_ablation_report.md"},
    {"scene": "parking_phone_tiny", "row": "real gate-removed ratio0.04 long", "finding": "F41 same-schedule 2000-iteration ablation again shows gate-on rolling back the no-accept 2579-triangle candidate set while gate-off commits it; final metrics are mixed, so this supports unsafe-edit rejection rather than monotonic final-metric dominance", "evidence": "docs/car_model/final_stageF41_gate_removed_ratio004_long_report.md"},
    {"scene": "parking_phone_tiny", "row": "real gate-removed ratio0.04 7000", "finding": "F42 same-schedule 7000-iteration ablation again shows gate-on rolling back the no-accept 2579-triangle candidate set while gate-off commits it; gated wins PSNR, SSIM, and LPIPS, while sparse geometry proxies are still slightly better for no-gate", "evidence": "docs/car_model/final_stageF42_gate_removed_ratio004_7000_report.md"},
    {"scene": "room", "row": "QEM50 no-freeze", "finding": "omitting strict topology freeze collapses topology to 20,742 triangles and loses badly to frozen QEM50", "evidence": "docs/car_model/final_stageF24_room_qem_no_freeze_control_report.md"},
    {"scene": "room", "row": "QEM50 + sparse-depth", "finding": "improves SSIM, LPIPS, AbsRel, Depth MAE, and normal at identical topology, but gives back 0.001 dB PSNR, so QEM50 remains the room PSNR headline", "evidence": "docs/car_model/final_stageF29_room_qem_sparse_depth_report.md"},
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
        "PASS with ablation gaps. Five scenes show compact-recovery benefit over fair clean-long baselines, and all five have auditable long-baseline comparisons. F34 adds a parking long-continuation rejection showing that simply training the best sparse-depth row from 26k to 30k hurts visual quality, so F33 is the validated stopping point. F36 extends no-freeze failure evidence to all five final-package scenes, including the largest parking scene. F37 closes the matched parking posthoc simplification gap: fast-QEM reaches the target topology and is strong on sparse geometry proxies, but collapses visual render quality. F38 closes the implementation-level no-gate/no-rollback counterfactual with identical-edit evidence. F39 adds a real-scene gate-removed same-schedule ablation where gate-on rolls back an aggressive candidate set and gate-off commits it with worse PSNR, SSIM, LPIPS, AbsRel, and Depth MAE. F41 adds the requested longer ratio0.04 counterpart with the same rollback/commit mechanism but mixed metrics. F42 extends the real gate-removed ablation to 7000 iterations: gate-on again blocks the no-accept candidate commit and now wins PSNR, SSIM, and LPIPS, while no-gate remains slightly better on sparse geometry proxies. The remaining NeurIPS risk is now mainly multi-scene long-budget gate-removed ablations and tightening the paper claim around render-quality Pareto improvement rather than universal dominance over every geometry proxy.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "main_quantitative_table.csv")
    print(OUT / "negative_result_table.csv")
    print(DOC)


if __name__ == "__main__":
    main()
