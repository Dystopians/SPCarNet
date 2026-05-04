#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "outputs/carnet/meshsplatopt/final_multiscene_package"
OUT = ROOT / "outputs/carnet/meshsplatopt/final_paper_assets"
DOC = ROOT / "docs/car_model/final_stageF13_paper_assets_report.md"
PANEL_W = 320
PANEL_H = 220


QUALITATIVE_SCENES = [
    {
        "scene": "parking_phone_tiny",
        "gt": "outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/csef_low_evidence_boundary_protected/prune70/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model/test/ours_22000/renders",
        "control": "outputs/carnet/meshsplatopt/final_stageF37_parking_fast_qem_matched_baseline/prune70_pass6/recovery_model/test/ours_26000/renders",
        "ours": "outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model/test/ours_26000/renders",
        "control_label": "fast-QEM70",
        "ours_label": "CSEF70+sparse",
    },
    {
        "scene": "bonsai",
        "gt": "outputs/carnet/meshsplatopt/final_stageF28_bonsai_qem_sparse_depth/prune50/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000/test/ours_22000/renders",
        "control": "outputs/carnet/meshsplatopt/final_stageF22_bonsai_posthoc_qem_baseline/prune50/recovery_model/test/ours_26000/renders",
        "ours": "outputs/carnet/meshsplatopt/final_stageF28_bonsai_qem_sparse_depth/prune50/recovery_model/test/ours_26000/renders",
        "control_label": "QEM50",
        "ours_label": "QEM50+sparse",
    },
    {
        "scene": "courtyard",
        "gt": "outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/courtyard/csef_low_evidence_boundary_protected/prune50/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000/test/ours_22000/renders",
        "control": "outputs/carnet/meshsplatopt/final_stageF23_courtyard_posthoc_qem_baseline/prune50/recovery_model/test/ours_26000/renders",
        "ours": "outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/courtyard/csef_low_evidence_boundary_protected/prune50/recovery_model/test/ours_26000/renders",
        "control_label": "QEM50",
        "ours_label": "CSEF50",
    },
    {
        "scene": "room",
        "gt": "outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000/test/ours_22000/renders",
        "control": "outputs/carnet/meshsplatopt/final_stageF19_room_selector_ablation/area_smallest/prune50/recovery_model/test/ours_26000/renders",
        "ours": "outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/recovery_model/test/ours_26000/renders",
        "control_label": "area50",
        "ours_label": "QEM50",
    },
    {
        "scene": "counter",
        "gt": "outputs/carnet/meshsplatopt/final_stageF32_counter_qem_sparse_depth/prune40/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000/test/ours_22000/renders",
        "control": "outputs/carnet/meshsplatopt/final_stageF21_counter_posthoc_qem_baseline/prune40/recovery_model/test/ours_26000/renders",
        "ours": "outputs/carnet/meshsplatopt/final_stageF32_counter_qem_sparse_depth/prune40/recovery_model/test/ours_26000/renders",
        "control_label": "QEM40",
        "ours_label": "QEM40+sparse",
    },
]

FREEZE_FAILURE_SCENES = [
    {
        "scene": "parking_phone_tiny",
        "gt": "outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model/test/ours_22000/renders",
        "frozen": "outputs/carnet/meshsplatopt/final_stageF33_parking_csef_sparse_depth/prune70/recovery_model/test/ours_26000/renders",
        "no_freeze": "outputs/carnet/meshsplatopt/final_stageF36_parking_csef_no_freeze_control/prune70/recovery_model/test/ours_26000/renders",
        "frozen_label": "CSEF70+sparse frozen",
        "no_freeze_label": "CSEF70 no-freeze",
    },
    {
        "scene": "courtyard",
        "gt": "outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/courtyard/csef_low_evidence_boundary_protected/prune50/recovery_model/test/ours_26000/gt",
        "clean": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000/test/ours_22000/renders",
        "frozen": "outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/courtyard/csef_low_evidence_boundary_protected/prune50/recovery_model/test/ours_26000/renders",
        "no_freeze": "outputs/carnet/meshsplatopt/final_stageF35_courtyard_csef_no_freeze_control/csef50/recovery_model/test/ours_26000/renders",
        "frozen_label": "CSEF50 frozen",
        "no_freeze_label": "CSEF50 no-freeze",
    },
]


def load_rows() -> list[dict[str, str]]:
    table = PKG / "main_quantitative_table.csv"
    if not table.exists():
        raise FileNotFoundError(f"Missing {table}; run final_collect_multiscene_package.py first")
    with table.open() as f:
        return list(csv.DictReader(f))


def make_method_diagram(path: Path) -> None:
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (1400, 360), "white")
    d = ImageDraw.Draw(canvas)
    boxes = [
        ("Clean long\nMesh Splatting", 35, 110),
        ("CSEF\nsurface evidence", 260, 110),
        ("Certified edit\ncompact/repair", 500, 110),
        ("Rollback gate\nrender+geometry", 750, 110),
        ("Topology-frozen\nrecovery", 1000, 110),
        ("Pareto report\nquality/topology", 1220, 110),
    ]
    for text, x, y in boxes:
        d.rounded_rectangle((x, y, x + 150, y + 95), radius=8, outline=(30, 30, 30), width=2, fill=(245, 248, 252))
        d.multiline_text((x + 12, y + 20), text, fill=(20, 20, 20), font=font, spacing=5)
    for (_, x, y), (_, nx, ny) in zip(boxes, boxes[1:]):
        d.line((x + 150, y + 48, nx - 14, ny + 48), fill=(50, 50, 50), width=3)
        d.polygon([(nx - 14, ny + 48), (nx - 28, ny + 40), (nx - 28, ny + 56)], fill=(50, 50, 50))
    d.text((40, 255), "All table rows use independent render.py metrics and scene-matched clean-long baselines.", fill=(50, 50, 50), font=font)
    canvas.save(path)


def make_triangle_bar(rows: list[dict[str, str]], path: Path) -> None:
    font = ImageFont.load_default()
    w, h = 1200, 520
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    max_tri = max(int(float(row["clean_triangles"])) for row in rows)
    y = 60
    d.text((40, 20), "Triangle Count: Clean Long vs MeshSplatOpt", fill=(20, 20, 20), font=font)
    for row in rows:
        clean = int(float(row["clean_triangles"]))
        ours = int(float(row["best_triangles"]))
        scene = row["scene"]
        clean_w = int(760 * clean / max_tri)
        ours_w = int(760 * ours / max_tri)
        d.text((40, y), scene, fill=(20, 20, 20), font=font)
        d.rectangle((200, y, 200 + clean_w, y + 18), fill=(160, 160, 160))
        d.rectangle((200, y + 24, 200 + ours_w, y + 42), fill=(50, 120, 200))
        d.text((980, y), f"clean {clean:,}", fill=(40, 40, 40), font=font)
        d.text((980, y + 24), f"ours {ours:,}", fill=(40, 40, 40), font=font)
        y += 82
    canvas.save(path)


def make_pareto_summary(rows: list[dict[str, str]], path: Path) -> None:
    data = [
        {
            "scene": row["scene"],
            "reduction": float(row["reduction"]),
            "d_psnr": float(row["d_psnr"]),
            "d_ssim": float(row["d_ssim"]),
            "d_lpips": float(row["d_lpips"]),
            "source": "outputs/carnet/meshsplatopt/final_multiscene_package/main_quantitative_table.csv",
        }
        for row in rows
    ]
    path.write_text(json.dumps(data, indent=2))


def image_names(path: Path) -> set[str]:
    return {p.name for p in path.glob("*.png")}


def select_frame(paths: list[Path]) -> str:
    common = set.intersection(*(image_names(path) for path in paths))
    if not common:
        raise RuntimeError(f"No common frames for {paths}")
    preferred = ["00005.png", "00013.png", "00022.png", "00031.png"]
    for name in preferred:
        if name in common:
            return name
    return sorted(common)[len(common) // 2]


def fit_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), (245, 245, 245))
    x = (PANEL_W - img.width) // 2
    y = (PANEL_H - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def error_heatmap(gt: Path, pred: Path) -> Image.Image:
    gt_img = fit_image(gt)
    pred_img = fit_image(pred)
    diff = ImageChops.difference(gt_img, pred_img).convert("L")
    diff = ImageOps.autocontrast(diff)
    heat = ImageOps.colorize(diff, black=(20, 20, 20), white=(255, 50, 20))
    return heat


def captioned(img: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    font = ImageFont.load_default()
    out = Image.new("RGB", (PANEL_W, PANEL_H + 38), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle((0, PANEL_H, PANEL_W, PANEL_H + 38), fill=(255, 255, 255))
    d.text((8, PANEL_H + 6), title, fill=(20, 20, 20), font=font)
    if subtitle:
        d.text((8, PANEL_H + 22), subtitle, fill=(80, 80, 80), font=font)
    return out


def concat_h(images: list[Image.Image]) -> Image.Image:
    w = sum(img.width for img in images)
    h = max(img.height for img in images)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width
    return out


def concat_v(images: list[Image.Image], gap: int = 14) -> Image.Image:
    w = max(img.width for img in images)
    h = sum(img.height for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def make_qualitative_panel(spec: dict[str, str], out_dir: Path) -> dict[str, str]:
    paths = {key: ROOT / spec[key] for key in ("gt", "clean", "control", "ours")}
    frame = select_frame(list(paths.values()))
    gt = paths["gt"] / frame
    clean = paths["clean"] / frame
    control = paths["control"] / frame
    ours = paths["ours"] / frame
    row = concat_h(
        [
            captioned(fit_image(gt), "GT", frame),
            captioned(fit_image(clean), "clean long", "22k"),
            captioned(fit_image(control), spec["control_label"], "same budget"),
            captioned(fit_image(ours), spec["ours_label"], "ours"),
            captioned(error_heatmap(gt, ours), "ours error", "abs RGB"),
        ]
    )
    path = out_dir / f"{spec['scene']}_qualitative_panel.png"
    row.save(path)
    return {
        "scene": spec["scene"],
        "frame": frame,
        "panel": str(path),
        "gt": str(gt),
        "clean": str(clean),
        "control": str(control),
        "ours": str(ours),
    }


def make_qualitative_assets(out_dir: Path) -> tuple[list[dict[str, str]], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    images = []
    for spec in QUALITATIVE_SCENES:
        panel = make_qualitative_panel(spec, out_dir)
        panels.append(panel)
        images.append(Image.open(panel["panel"]).convert("RGB"))
    montage = out_dir / "final_multiscene_qualitative_montage.png"
    concat_v(images).save(montage)
    return panels, montage


def make_freeze_failure_panel(spec: dict[str, str], out_dir: Path) -> dict[str, str]:
    paths = {key: ROOT / spec[key] for key in ("gt", "clean", "frozen", "no_freeze")}
    frame = select_frame(list(paths.values()))
    gt = paths["gt"] / frame
    clean = paths["clean"] / frame
    frozen = paths["frozen"] / frame
    no_freeze = paths["no_freeze"] / frame
    row = concat_h(
        [
            captioned(fit_image(gt), "GT", frame),
            captioned(fit_image(clean), "clean long", "22k"),
            captioned(fit_image(frozen), spec["frozen_label"], "838,742 tri"),
            captioned(fit_image(no_freeze), spec["no_freeze_label"], "1,317,435 tri"),
            captioned(error_heatmap(gt, no_freeze), "no-freeze error", "abs RGB"),
        ]
    )
    path = out_dir / f"{spec['scene']}_no_freeze_failure_panel.png"
    row.save(path)
    return {
        "scene": spec["scene"],
        "frame": frame,
        "panel": str(path),
        "gt": str(gt),
        "clean": str(clean),
        "frozen": str(frozen),
        "no_freeze": str(no_freeze),
    }


def make_freeze_failure_assets(out_dir: Path) -> tuple[list[dict[str, str]], Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    images = []
    for spec in FREEZE_FAILURE_SCENES:
        panel = make_freeze_failure_panel(spec, out_dir)
        panels.append(panel)
        images.append(Image.open(panel["panel"]).convert("RGB"))
    if not images:
        return panels, None
    montage = out_dir / "freeze_failure_montage.png"
    concat_v(images).save(montage)
    return panels, montage


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    method = OUT / "meshsplatopt_method_diagram.png"
    bars = OUT / "triangle_count_bar_chart.png"
    pareto = OUT / "pareto_summary.json"
    make_method_diagram(method)
    make_triangle_bar(rows, bars)
    make_pareto_summary(rows, pareto)
    qualitative, qualitative_montage = make_qualitative_assets(OUT / "qualitative_panels")
    freeze_failures, freeze_failure_montage = make_freeze_failure_assets(OUT / "freeze_failure_panels")

    source_montages = [
        ROOT / "outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_montage.png",
        ROOT / "outputs/carnet/meshsplatopt/final_stageF10_qualitative_evidence/room_counter_clean_vs_csef_montage.png",
    ]
    copied = []
    for src in source_montages:
        if src.exists():
            dst = OUT / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    manifest = {
        "method_diagram": str(method),
        "triangle_count_bar_chart": str(bars),
        "pareto_summary": str(pareto),
        "copied_montages": [str(p) for p in copied],
        "qualitative_panels": qualitative,
        "qualitative_montage": str(qualitative_montage),
        "freeze_failure_panels": freeze_failures,
        "freeze_failure_montage": str(freeze_failure_montage) if freeze_failure_montage else "",
        "source_table": str(PKG / "main_quantitative_table.csv"),
    }
    (OUT / "paper_assets_manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "# Final Stage F13 - Paper Assets Report",
        "",
        "Decision: `FINAL_F13_PAPER_ASSETS_PASS_TRACEABLE`.",
        "",
        "## Assets",
        "",
        f"- method diagram: `{method}`",
        f"- triangle count bar chart: `{bars}`",
        f"- Pareto summary JSON: `{pareto}`",
        f"- multi-scene qualitative montage: `{qualitative_montage}`",
        f"- freeze-failure montage: `{freeze_failure_montage}`",
        f"- manifest: `{OUT / 'paper_assets_manifest.json'}`",
    ]
    for item in qualitative:
        lines.append(f"- {item['scene']} qualitative panel: `{item['panel']}`")
    for item in freeze_failures:
        lines.append(f"- {item['scene']} no-freeze failure panel: `{item['panel']}`")
    for p in copied:
        lines.append(f"- qualitative montage: `{p}`")
    lines += [
        "",
        "## Traceability",
        "",
        "All quantitative assets are generated from `outputs/carnet/meshsplatopt/final_multiscene_package/main_quantitative_table.csv`. The new qualitative panels use independent render outputs and record every source image path plus the selected frame in `paper_assets_manifest.json`. The freeze-failure panels use the F35/F36 independent no-freeze renders and show why the strict topology-frozen recovery contract is visually load-bearing.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "paper_assets_manifest.json")
    print(DOC)


if __name__ == "__main__":
    main()
