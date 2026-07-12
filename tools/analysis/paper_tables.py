#!/usr/bin/env python
"""Emit LaTeX table bodies from banked GEMS JSON artifacts.

The paper should include these generated bodies rather than copied numbers.
This script intentionally uses only Python stdlib modules.
"""

from __future__ import annotations

import json
import os


REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"
OUT_DIR = os.path.join(REPO, "RESULTS", "tables_tex")

FINAL_SUMMARY = os.path.join(G1, "analysis", "final_stack", "final_stack_summary.json")
HIER_CIS = os.path.join(G1, "analysis", "final_stack", "hierarchical_cis.json")
E07_3DGS = os.path.join(G1, "analysis", "final_stack", "e07_matched_total_3dgs.json")
TEMPORAL = os.path.join(G1, "analysis", "temporal", "temporal_summary.json")
L4_GATE = os.path.join(G1, "analysis", "e0_pj2026", "l4_gate.json")


NOTES = []


def note(message: str) -> None:
    NOTES.append(message)
    print("NOTE:", message)


def load_json(path: str, table: str, row: str | None = None, required: bool = True):
    if not os.path.exists(path):
        target = f"{table}" if row is None else f"{table} row {row}"
        if required:
            note(f"{target}: missing source {path}; skipping affected output")
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_by_basename(basename: str) -> list[str]:
    matches = []
    for root, _, files in os.walk(G1):
        if basename in files:
            matches.append(os.path.join(root, basename))
    return sorted(matches)


def tex_cell(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt_num(value, digits: int, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def fmt_ci(stat: dict, digits: int = 3) -> str:
    return (
        f"{fmt_num(stat.get('mean'), digits, signed=True)} "
        f"[{fmt_num(stat.get('ci_lo'), digits, signed=True)}, "
        f"{fmt_num(stat.get('ci_hi'), digits, signed=True)}]"
    )


def fmt_diff_ci(stat: dict, digits: int = 3) -> str:
    return (
        f"{fmt_num(stat.get('mean_diff'), digits, signed=True)} "
        f"[{fmt_num(stat.get('ci_lo'), digits, signed=True)}, "
        f"{fmt_num(stat.get('ci_hi'), digits, signed=True)}]"
    )


def row(cells: list[object]) -> str:
    return " & ".join(tex_cell(c) for c in cells) + r" \\"


def write_table(filename: str, sources: list[str], lines: list[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    unique_sources = []
    seen = set()
    for src in sources:
        if src and src not in seen:
            unique_sources.append(src)
            seen.add(src)
    header = "% Sources: " + "; ".join(unique_sources)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for line in lines:
            fh.write(line + "\n")
    print(f"wrote {path}")


def nested(obj, keys: list[str]):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def required_stat(obj: dict, keys: list[str], table: str, row_label: str):
    stat = nested(obj, keys)
    if not isinstance(stat, dict):
        note(f"{table} row {row_label}: missing value {'.'.join(keys)}; skipping row")
        return None
    return stat


def build_t1() -> None:
    table = "T1_main"
    summary = load_json(FINAL_SUMMARY, table)
    hier = load_json(HIER_CIS, table)
    if summary is None or hier is None:
        write_table("T1_main.tex", [p for p in (FINAL_SUMMARY, HIER_CIS) if os.path.exists(p)], [])
        return

    lines = []
    per_scene = summary.get("per_scene", {})
    # final_stack_summary.json stores a canonical paper order: full9 first,
    # then the extra external/temporal scenes. Preserve that source order.
    for scene, data in per_scene.items():
        pj = data.get("vs", {}).get("pj", {})
        psnr_stat = pj.get("psnr")
        lpips_stat = pj.get("lpips")
        if not isinstance(psnr_stat, dict) or not isinstance(lpips_stat, dict):
            note(f"{table} row {scene}: missing vs.pj psnr/lpips stats; skipping row")
            continue
        lines.append(
            row(
                [
                    scene,
                    fmt_num(data.get("psnr"), 2),
                    fmt_num(data.get("ssim"), 4),
                    fmt_num(data.get("lpips"), 4),
                    fmt_diff_ci(psnr_stat, 3),
                    fmt_diff_ci(lpips_stat, 3),
                ]
            )
        )

    lines.append(r"\midrule")
    pj_full9 = summary.get("full9", {}).get("pj", {})
    dpsnr = pj_full9.get("dpsnr")
    dlpips = pj_full9.get("dlpips")
    if isinstance(dpsnr, dict) and isinstance(dlpips, dict):
        lines.append(row(["full9 mean (stratified)", "--", "--", "--", fmt_ci(dpsnr, 3), fmt_ci(dlpips, 3)]))
    else:
        note(f"{table} row full9 mean (stratified): missing full9.pj dpsnr/dlpips; skipping row")

    h1 = hier.get("H1_final_vs_pj2026", {})
    hpsnr = required_stat(h1, ["psnr", "hierarchical"], table, "full9 mean (scene-cluster)")
    hlpips = required_stat(h1, ["lpips", "hierarchical"], table, "full9 mean (scene-cluster)")
    if hpsnr is not None and hlpips is not None:
        lines.append(row(["full9 mean (scene-cluster)", "--", "--", "--", fmt_ci(hpsnr, 3), fmt_ci(hlpips, 3)]))

    write_table("T1_main.tex", [FINAL_SUMMARY, HIER_CIS], lines)


def build_t2() -> None:
    table = "T2_ladder"
    sources = []
    lines = []
    gate_files = [
        ("l1", os.path.join(G1, "analysis", "e0_pj2026", "l1_gate.json")),
        ("l2", os.path.join(G1, "analysis", "e0_pj2026", "l2_gate.json")),
        ("l3", os.path.join(G1, "analysis", "e0_pj2026", "l3_gate.json")),
        ("l4", os.path.join(G1, "analysis", "e0_pj2026", "l4_gate.json")),
        ("l4_vs_floor", os.path.join(G1, "analysis", "e0_pj2026", "l4_vs_floor.json")),
    ]
    for default_label, path in gate_files:
        data = load_json(path, table, default_label)
        if data is None:
            continue
        sources.append(path)
        dpsnr = data.get("full9_mean_dpsnr")
        dlpips = data.get("full9_mean_dlpips")
        if not isinstance(dpsnr, dict) or not isinstance(dlpips, dict):
            note(f"{table} row {default_label}: missing full9 mean stats; skipping row")
            continue
        lines.append(row([data.get("label", default_label), fmt_ci(dpsnr, 3), fmt_ci(dlpips, 3), data.get("verdict", "--")]))

    for scene in ("bonsai", "garden"):
        label = f"conf-inputs-off ({scene})"
        abl = os.path.join(G1, "eval", f"abl_{scene}_confoff_v1", "metrics.json")
        base = os.path.join(G1, "eval", f"l4_{scene}_cleanfixed30k_routed_v1", "metrics.json")
        diff = paired_mean_psnr(abl, base, table, label)
        if diff is None:
            continue
        sources.extend([abl, base])
        lines.append(row([label, fmt_num(diff, 3, signed=True), "--", "--"]))

    write_table("T2_ladder.tex", sources, lines)


def per_view_metric(metrics: dict, metric: str):
    values = nested(metrics, ["rendering", "per_view", metric])
    return values if isinstance(values, list) else None


def paired_mean_psnr(abl_path: str, base_path: str, table: str, label: str):
    abl = load_json(abl_path, table, label)
    base = load_json(base_path, table, label)
    if abl is None or base is None:
        return None
    abl_names = nested(abl, ["rendering", "per_view", "image_names"])
    base_names = nested(base, ["rendering", "per_view", "image_names"])
    if isinstance(abl_names, list) and isinstance(base_names, list) and abl_names != base_names:
        note(f"{table} row {label}: per-view image_names differ; skipping row")
        return None
    abl_psnr = per_view_metric(abl, "psnr")
    base_psnr = per_view_metric(base, "psnr")
    if abl_psnr is None or base_psnr is None or len(abl_psnr) != len(base_psnr) or not abl_psnr:
        note(f"{table} row {label}: missing or mismatched rendering.per_view.psnr arrays; skipping row")
        return None
    diffs = [float(a) - float(b) for a, b in zip(abl_psnr, base_psnr)]
    return sum(diffs) / len(diffs)


def build_t3() -> None:
    table = "T3_compact"
    summary = load_json(FINAL_SUMMARY, table)
    l4_gate = load_json(L4_GATE, table)
    sources = [FINAL_SUMMARY]
    if l4_gate is not None:
        sources.append(L4_GATE)
    if summary is None:
        write_table("T3_compact.tex", [p for p in sources if os.path.exists(p)], [])
        return

    lines = []
    l6 = summary.get("l6", {})
    dpsnr = l6.get("dpsnr_vs_primary_anchor")
    dlpips = l6.get("dlpips_vs_primary_anchor")
    if isinstance(dpsnr, dict):
        lines.append(row(["L6 vs primary anchor", "dPSNR", fmt_ci(dpsnr, 3)]))
    else:
        note(f"{table} row L6 dpsnr_vs_primary_anchor: missing stat; skipping row")
    if isinstance(dlpips, dict):
        lines.append(row(["L6 vs primary anchor", "dLPIPS", fmt_ci(dlpips, 3)]))
    else:
        note(f"{table} row L6 dlpips_vs_primary_anchor: missing stat; skipping row")

    # Use final_stack_summary.json for per-scene final PSNR. The eval/final_* rows
    # currently differ slightly and are not the final-stack summary values.
    if isinstance(l4_gate, dict) and isinstance(l4_gate.get("per_scene"), dict):
        full9_scenes = list(l4_gate["per_scene"].keys())
    else:
        full9_scenes = sorted(summary.get("per_scene", {}).keys())
        note(f"{table}: l4_gate full9 scene list unavailable; using sorted final_stack per_scene keys")
    per_scene = summary.get("per_scene", {})
    for scene in full9_scenes:
        data = per_scene.get(scene)
        if not isinstance(data, dict) or "psnr" not in data:
            note(f"{table} row final_{scene}_B50_v1: missing per_scene.{scene}.psnr; skipping row")
            continue
        lines.append(row([f"final_{scene}_B50_v1", "PSNR", fmt_num(data.get("psnr"), 2)]))

    write_table("T3_compact.tex", sources, lines)


def extract_metric(obj: dict, metric: str):
    paths = [
        [metric],
        ["metrics", metric],
        ["mean", metric],
        ["rendering", "mean", metric],
        ["ibr", metric],
        ["difix", metric],
    ]
    for keys in paths:
        value = nested(obj, keys)
        if isinstance(value, (int, float)):
            return value
    return None


def build_t4() -> None:
    table = "T4_cost_external"
    matched = load_json(E07_3DGS, table)
    sources = [E07_3DGS] if matched is not None else []
    lines = []
    if matched is None or not isinstance(matched.get("scenes"), dict):
        write_table("T4_cost_external.tex", sources, lines)
        return

    # The banked R1/external files define the real trio as bicycle/garden/kitchen.
    # That differs from the prompt's bonsai hint, so discover scenes from E07.
    scenes = sorted(matched["scenes"].keys())
    for scene in scenes:
        ecr_metrics_path = os.path.join(G1, "eval", f"l4_{scene}_cleanfixed30k_routed_v1", "metrics.json")
        ecr = load_json(ecr_metrics_path, table, f"ECR cost ({scene})")
        if ecr is not None:
            cost = ecr.get("cost", {})
            if all(k in cost for k in ("total_artifact_mb", "transport_ms_per_frame", "end_to_end_fps")):
                sources.append(ecr_metrics_path)
                lines.append(
                    row(
                        [
                            scene,
                            "ECR final stack",
                            "--",
                            "--",
                            fmt_num(cost.get("total_artifact_mb"), 2),
                            fmt_num(cost.get("transport_ms_per_frame"), 2),
                            fmt_num(cost.get("end_to_end_fps"), 2),
                        ]
                    )
                )
            else:
                note(f"{table} row ECR cost ({scene}): missing cost total/transport/e2e values; skipping row")

        m3d = matched["scenes"][scene].get("3dgs_matched_total", {})
        if "psnr" in m3d and "disk_mb" in m3d:
            lines.append(row([scene, "3DGS matched point", fmt_num(m3d.get("psnr"), 2), "--", fmt_num(m3d.get("disk_mb"), 2), "--", "--"]))
        else:
            note(f"{table} row 3DGS matched point ({scene}): missing psnr/disk_mb; skipping row")

        difix_path = os.path.join(G1, "analysis", "difix_cell", f"difix_{scene}.json")
        difix = load_json(difix_path, table, f"Difix ({scene})")
        if difix is not None:
            psnr = extract_metric(difix, "psnr")
            lpips = extract_metric(difix, "lpips")
            if psnr is None or lpips is None:
                note(f"{table} row Difix ({scene}): missing psnr/lpips; skipping row")
            else:
                sources.append(difix_path)
                lines.append(row([scene, "Difix3D+", fmt_num(psnr, 2), fmt_num(lpips, 4), "--", "--", "--"]))

        ibr_path = os.path.join(G1, "analysis", "ibr_cell", f"ibr_{scene}.json")
        if not os.path.exists(ibr_path):
            alt = find_by_basename(f"ibr_{scene}.json")
            if alt:
                ibr_path = alt[0] if len(alt) == 1 else ""
        if not ibr_path or not os.path.exists(ibr_path):
            note(f"{table} row IBRNet ({scene}): missing source {os.path.join(G1, 'analysis', 'ibr_cell', f'ibr_{scene}.json')}; writing pending cells")
            lines.append(row([scene, "IBRNet", "pending", "pending", "pending", "pending", "pending"]))
            continue
        ibr = load_json(ibr_path, table, f"IBRNet ({scene})", required=False)
        psnr = extract_metric(ibr, "psnr") if isinstance(ibr, dict) else None
        lpips = extract_metric(ibr, "lpips") if isinstance(ibr, dict) else None
        if psnr is None or lpips is None:
            note(f"{table} row IBRNet ({scene}): missing psnr/lpips in {ibr_path}; writing pending cells")
            lines.append(row([scene, "IBRNet", "pending", "pending", "pending", "pending", "pending"]))
        else:
            sources.append(ibr_path)
            lines.append(row([scene, "IBRNet", fmt_num(psnr, 2), fmt_num(lpips, 4), "--", "--", "--"]))

    write_table("T4_cost_external.tex", sources, lines)


def build_t5() -> None:
    table = "T5_temporal"
    temporal = load_json(TEMPORAL, table)
    lines = []
    if temporal is None:
        write_table("T5_temporal.tex", [TEMPORAL] if os.path.exists(TEMPORAL) else [], lines)
        return
    for scene in sorted(temporal.keys()):
        data = temporal[scene]
        needed = [
            "roughness_base_mean",
            "roughness_final_mean",
            "roughness_ratio_mean",
            "support_switches_per_step_mean",
        ]
        if not isinstance(data, dict) or any(k not in data for k in needed):
            note(f"{table} row {scene}: missing temporal roughness/switch values; skipping row")
            continue
        lines.append(
            row(
                [
                    scene,
                    fmt_num(data.get("roughness_base_mean"), 4),
                    fmt_num(data.get("roughness_final_mean"), 4),
                    fmt_num(data.get("roughness_ratio_mean"), 3),
                    fmt_num(data.get("support_switches_per_step_mean"), 3),
                ]
            )
        )
    write_table("T5_temporal.tex", [TEMPORAL], lines)


def main() -> None:
    build_t1()
    build_t2()
    build_t3()
    build_t4()
    build_t5()
    if NOTES:
        print(f"completed with {len(NOTES)} note(s)")
    else:
        print("completed with no notes")


if __name__ == "__main__":
    main()
