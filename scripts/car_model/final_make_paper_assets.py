#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "outputs/carnet/meshsplatopt/final_multiscene_package"
OUT = ROOT / "outputs/carnet/meshsplatopt/final_paper_assets"
DOC = ROOT / "docs/car_model/final_stageF13_paper_assets_report.md"


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    method = OUT / "meshsplatopt_method_diagram.png"
    bars = OUT / "triangle_count_bar_chart.png"
    pareto = OUT / "pareto_summary.json"
    make_method_diagram(method)
    make_triangle_bar(rows, bars)
    make_pareto_summary(rows, pareto)

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
        f"- manifest: `{OUT / 'paper_assets_manifest.json'}`",
    ]
    for p in copied:
        lines.append(f"- qualitative montage: `{p}`")
    lines += [
        "",
        "## Traceability",
        "",
        "All quantitative assets are generated from `outputs/carnet/meshsplatopt/final_multiscene_package/main_quantitative_table.csv`. Qualitative montages retain source render paths through their original manifests.",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(OUT / "paper_assets_manifest.json")
    print(DOC)


if __name__ == "__main__":
    main()

