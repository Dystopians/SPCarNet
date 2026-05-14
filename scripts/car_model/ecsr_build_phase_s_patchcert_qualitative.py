#!/usr/bin/env python3
"""Build qualitative panels for the Phase-S direct patch-cert carrier pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_method(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not payload:
        return "", {}
    key = next(iter(payload.keys()))
    row = payload.get(key)
    return str(key), row if isinstance(row, dict) else {}


def per_view(path: Path, preferred_method: str = "") -> tuple[str, dict[str, dict[str, float]]]:
    payload = read_json(path)
    if preferred_method and isinstance(payload.get(preferred_method), dict):
        method, row = preferred_method, payload[preferred_method]
    else:
        method, row = first_method(payload)
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = row.get(metric) if isinstance(row, dict) else None
        if not isinstance(values, dict):
            continue
        for view_name, value in values.items():
            try:
                out.setdefault(str(view_name), {})[metric] = float(value)
            except Exception:
                continue
    return method, {name: vals for name, vals in out.items() if all(key in vals for key in METRICS)}


def selected_model(policy_root: Path, scene: str) -> Path:
    summary = read_json(policy_root / scene / "summary.json")
    model_path = ((summary.get("selected") or {}).get("model_path") or "").strip()
    if not model_path:
        raise FileNotFoundError(policy_root / scene / "summary.json")
    path = ROOT / model_path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def thumb(image: Image.Image, width: int) -> Image.Image:
    img = image.copy()
    w, h = img.size
    if w == width:
        return img
    height = max(1, int(round(h * width / max(w, 1))))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def label_panel(image: Image.Image, text: str) -> Image.Image:
    pad = 8
    label_h = 34
    out = Image.new("RGB", (image.width, image.height + label_h), (255, 255, 255))
    out.paste(image, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, label_h), fill=(245, 245, 245))
    font = ImageFont.load_default()
    draw.text((pad, 10), text, fill=(20, 20, 20), font=font)
    return out


def abs_error(render: Image.Image, gt: Image.Image) -> np.ndarray:
    a = np.asarray(render, dtype=np.float32) / 255.0
    b = np.asarray(gt.resize(render.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return np.abs(a - b).mean(axis=2)


def heat_error(err: np.ndarray, boost: float) -> Image.Image:
    value = np.clip(err * float(boost), 0.0, 1.0)
    rgb = np.stack([value, value, value], axis=2)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB")


def heat_delta(base_err: np.ndarray, cand_err: np.ndarray, boost: float) -> Image.Image:
    delta = (base_err - cand_err) * float(boost)
    pos = np.clip(delta, 0.0, 1.0)
    neg = np.clip(-delta, 0.0, 1.0)
    rgb = np.zeros((*delta.shape, 3), dtype=np.float32)
    rgb[..., 1] = pos
    rgb[..., 0] = neg
    rgb[..., 2] = neg
    rgb += 0.08
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB")


def hcat(images: list[Image.Image], gap: int = 8) -> Image.Image:
    height = max(img.height for img in images)
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width + gap
    return out


def vcat(images: list[Image.Image], gap: int = 12) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def candidate_rows(args: argparse.Namespace, scene: str) -> list[dict[str, Any]]:
    scene_root = ROOT / str(args.root_template).format(scene=scene)
    decision_path = scene_root / "decisions" / f"{scene}_decision.json"
    decision = read_json(decision_path)
    if not decision:
        return []
    phasej_method = str(decision.get("base_test_method_report_only", ""))
    candidate_method = str(decision.get("candidate_test_method_report_only", ""))
    _, base = per_view(scene_root / scene / "phasej_test_per_view.json", phasej_method)
    _, cand = per_view(scene_root / scene / "model" / "test_per_view.json", candidate_method)
    model = selected_model(ROOT / args.policy_root, scene)
    phasej_dir = model / "test" / phasej_method
    cand_dir = scene_root / scene / "model" / "test" / candidate_method
    view_names = sorted(set(base) & set(cand))
    rows: list[dict[str, Any]] = []
    for view in view_names:
        delta = {key: cand[view][key] - base[view][key] for key in METRICS}
        score = delta["PSNR"] + 20.0 * delta["SSIM"] - 20.0 * delta["LPIPS"]
        paths = {
            "gt": cand_dir / "gt" / view,
            "phasej": phasej_dir / "renders" / view,
            "candidate": cand_dir / "renders" / view,
        }
        if all(path.is_file() for path in paths.values()):
            rows.append(
                {
                    "scene": scene,
                    "view": view,
                    "accepted": bool(decision.get("accepted", False)),
                    "selected": decision.get("selected_label", ""),
                    "decision_reasons": decision.get("decision_reasons", []),
                    "score": score,
                    "delta": delta,
                    "paths": {key: str(value) for key, value in paths.items()},
                }
            )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[: int(args.views_per_scene)]


def build_panel(row: dict[str, Any], out_dir: Path, *, image_width: int, diff_boost: float) -> Path:
    gt = load_rgb(Path(row["paths"]["gt"]))
    phasej = load_rgb(Path(row["paths"]["phasej"]))
    cand = load_rgb(Path(row["paths"]["candidate"]))
    gt_t = thumb(gt, image_width)
    phasej_t = thumb(phasej, image_width)
    cand_t = thumb(cand, image_width)
    phasej_err = abs_error(phasej_t, gt_t)
    cand_err = abs_error(cand_t, gt_t)
    err = heat_error(cand_err, diff_boost)
    delta = heat_delta(phasej_err, cand_err, diff_boost)
    d = row["delta"]
    title = (
        f"{row['scene']} {row['view']} "
        f"dP {d['PSNR']:+.6f} dS {d['SSIM']:+.6f} dL {d['LPIPS']:+.6f}"
    )
    panels = [
        label_panel(gt_t, "GT"),
        label_panel(phasej_t, "Phase-J"),
        label_panel(cand_t, "Direct PatchCert"),
        label_panel(err, f"PatchCert abs err x{diff_boost:g}"),
        label_panel(delta, f"green better / magenta worse x{diff_boost:g}"),
    ]
    row_img = hcat(panels)
    header = Image.new("RGB", (row_img.width, 32), (255, 255, 255))
    draw = ImageDraw.Draw(header)
    draw.text((8, 10), title, fill=(20, 20, 20), font=ImageFont.load_default())
    panel = vcat([header, row_img], gap=0)
    out_path = out_dir / f"{row['scene']}_{Path(row['view']).stem}_patchcert_panel.png"
    panel.save(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="garden,bicycle,counter,flowers,bonsai")
    parser.add_argument(
        "--root_template",
        default="outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_{scene}",
    )
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--out_dir", default="outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v5_patchcarrier_pilot_20260514_qualitative")
    parser.add_argument("--views_per_scene", type=int, default=1)
    parser.add_argument("--image_width", type=int, default=300)
    parser.add_argument("--diff_boost", type=float, default=80.0)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        rows.extend(candidate_rows(args, scene))
    panels: list[Path] = []
    for row in rows:
        path = build_panel(row, out_dir, image_width=int(args.image_width), diff_boost=float(args.diff_boost))
        row["panel"] = str(path)
        panels.append(path)
    if panels:
        contact = vcat([load_rgb(path) for path in panels], gap=16)
        contact_path = out_dir / "patchcert_qualitative_contact_sheet.png"
        contact.save(contact_path)
    else:
        contact_path = out_dir / "patchcert_qualitative_contact_sheet.png"
    (out_dir / "qualitative_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase-S Direct Patch-Cert Qualitative Panels",
        "",
        "Rows are selected by report-only held-out test balanced delta for visualization only.",
        "Selection/promotion remains controlled by train-val decision JSON files.",
        "",
        f"- contact sheet: `{contact_path}`",
        "",
        "| scene | view | accepted | dPSNR | dSSIM | dLPIPS | panel |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row["delta"]
        lines.append(
            f"| {row['scene']} | {row['view']} | {str(row['accepted']).lower()} | "
            f"{d['PSNR']:+.6f} | {d['SSIM']:+.6f} | {d['LPIPS']:+.6f} | `{row['panel']}` |"
        )
    (out_dir / "qualitative_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "contact_sheet": str(contact_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
