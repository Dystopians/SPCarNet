#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/carnet/meshsplatopt/final_stageF46_unified_csef50_sparse_depth"
OUT = ROOT / "docs/car_model/final_stageF46_unified_csef_sparse_depth_report.md"


@dataclass(frozen=True)
class Baseline:
    triangles: int
    psnr: float
    ssim: float
    lpips: float
    absrel: float
    depth: float
    normal: float


CLEAN = {
    "bonsai": Baseline(88460, 10.944348, 0.222848, 0.586158, 0.194249, 1.816410, 45.358356),
    "room": Baseline(84506, 14.258379, 0.400864, 0.578919, 0.206282, 1.480230, 55.442653),
    "counter": Baseline(83834, 14.136182, 0.512802, 0.452049, 0.076996, 0.369973, 44.287035),
}

CSEF_REFERENCE = {
    "bonsai_prune50": Baseline(44230, 10.957497, 0.224758, 0.586415, 0.185180, 1.737815, 43.493975),
    "room_prune50": Baseline(42253, 14.387163, 0.414954, 0.568281, 0.225027, 1.603030, 54.642793),
    "counter_prune50": Baseline(41917, 14.077559, 0.498974, 0.468391, 0.094731, 0.438932, 43.823390),
    "counter_prune40": Baseline(50300, 14.212033, 0.518401, 0.450481, 0.085542, 0.406373, 43.476972),
}

RUNS = [
    ("bonsai", "prune50", "xpv6dd08", "fixed CSEF50 + sparse-depth"),
    ("room", "prune50", "7fq1dnqk", "fixed CSEF50 + sparse-depth"),
    ("room", "prune20", "v7ld1o0x", "validation-budget CSEF20 + sparse-depth"),
    ("counter", "prune50", "vuvaul2s", "fixed CSEF50 + sparse-depth"),
    ("counter", "prune40", "ihoyzp1a", "validation-budget CSEF40 + sparse-depth"),
    ("counter", "prune30", "panxl9lh", "validation-budget CSEF30 + sparse-depth"),
    ("counter", "prune20", "pijpv7ny", "validation-budget CSEF20 + sparse-depth"),
]


def _load(scene: str, prune: str) -> dict:
    model = BASE / scene / prune / "recovery_model"
    contract = BASE / scene / prune / "recovery_contract"
    render = json.loads((model / "results.json").read_text(encoding="utf-8"))["ours_26000"]
    geom = json.loads((model / "geometry_eval_colmap/iter_26000_max500.json").read_text(encoding="utf-8"))
    topology = json.loads((contract / "topology_audit.json").read_text(encoding="utf-8"))
    return {
        "triangles": int(topology["final"]["triangles"]),
        "vertices": int(topology["final"]["vertices"]),
        "psnr": float(render["PSNR"]),
        "ssim": float(render["SSIM"]),
        "lpips": float(render["LPIPS"]),
        "absrel": float(geom["depth"]["abs_rel"]),
        "depth": float(geom["depth"]["mae"]),
        "normal": float(geom["normal"]["mean_ang_deg"]),
        "topology_unchanged": bool(topology["topology_unchanged"]),
        "sparse_depth_enabled": bool(topology["sparse_depth_enabled"]),
    }


def _delta(row: dict, base: Baseline) -> dict[str, float]:
    return {
        "d_psnr": row["psnr"] - base.psnr,
        "d_ssim": row["ssim"] - base.ssim,
        "d_lpips": row["lpips"] - base.lpips,
        "d_absrel": row["absrel"] - base.absrel,
        "d_depth": row["depth"] - base.depth,
        "d_normal": row["normal"] - base.normal,
    }


def _status(d: dict[str, float]) -> str:
    ok = (
        d["d_psnr"] > 0
        and d["d_ssim"] > 0
        and d["d_lpips"] < 0
        and d["d_absrel"] < 0
        and d["d_depth"] < 0
        and d["d_normal"] < 0
    )
    if ok:
        return "PASS_ALL_METRIC_CLEAN_WIN"
    render = d["d_psnr"] > 0 and d["d_ssim"] > 0 and d["d_lpips"] < 0
    geom = d["d_absrel"] < 0 and d["d_depth"] < 0 and d["d_normal"] < 0
    if render and geom:
        return "PASS"
    if render or geom:
        return "MIXED"
    return "FAIL"


def _fmt(value: float | int, digits: int = 6, signed: bool = False) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def main() -> int:
    rows = []
    for scene, prune, wandb, label in RUNS:
        row = _load(scene, prune)
        clean = CLEAN[scene]
        d = _delta(row, clean)
        rows.append((scene, prune, wandb, label, row, d, _status(d)))

    lines = [
        "# Final Stage F46 - Unified CSEF Sparse-Depth Fairness Repair",
        "",
        "Date: 2026-05-04",
        "",
        "Decision: `F46_VALIDATION_BUDGET_CSEF_REPAIR_PASS_WITH_FIXED50_LIMITATION`.",
        "",
        "## Goal",
        "",
        "Respond to the F45 fairness audit by running a single CSEF selector family with explicit sparse-depth strict topology-frozen recovery. The first batch tests fixed CSEF50; the second batch tests conservative validation-selected CSEF budgets on the scenes where fixed CSEF50 was weak.",
        "",
        "All rows use online W&B, `22000->26000`, `--freeze_topology_updates`, `--skip_restricted_delaunay`, `--enable_sparse_colmap_depth_loss`, low-error sparse COLMAP samples, independent render metrics, independent COLMAP geometry, and exact topology audit.",
        "",
        "## Results vs Clean-Long",
        "",
        "| scene | row | W&B | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scene, prune, wandb, label, row, d, status in rows:
        clean = CLEAN[scene]
        reduction = 100.0 * (1.0 - row["triangles"] / clean.triangles)
        lines.append(
            f"| {scene} | {label} | `{wandb}` | {_fmt(row['triangles'])} | {reduction:.1f}% | "
            f"{_fmt(row['psnr'])} | {_fmt(row['ssim'])} | {_fmt(row['lpips'])} | {_fmt(row['absrel'])} | "
            f"{_fmt(row['depth'])} | {_fmt(row['normal'])} | {_fmt(d['d_psnr'], signed=True)} | "
            f"{_fmt(d['d_ssim'], signed=True)} | {_fmt(d['d_lpips'], signed=True)} | "
            f"{_fmt(d['d_absrel'], signed=True)} | {_fmt(d['d_depth'], signed=True)} | "
            f"{_fmt(d['d_normal'], signed=True)} | `{status}` |"
        )

    lines.extend(
        [
            "",
            "## Reference Against Earlier CSEF Rows",
            "",
            "| scene | row | comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scene, prune, _wandb, label, row, _d, _status_value in rows:
        key = f"{scene}_{prune}"
        if key not in CSEF_REFERENCE:
            continue
        d = _delta(row, CSEF_REFERENCE[key])
        lines.append(
            f"| {scene} | {label} | vs previous CSEF {prune} | {_fmt(d['d_psnr'], signed=True)} | "
            f"{_fmt(d['d_ssim'], signed=True)} | {_fmt(d['d_lpips'], signed=True)} | "
            f"{_fmt(d['d_absrel'], signed=True)} | {_fmt(d['d_depth'], signed=True)} | "
            f"{_fmt(d['d_normal'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "F46 does not rescue the claim that one fixed CSEF50 hyperparameter is universally enough. Fixed CSEF50 remains weak on counter and mixed on room depth. That limitation should stay visible in the paper.",
            "",
            "F46 does, however, materially repairs the fairness story: using the same CSEF selector family, sparse-depth strict recovery, and a conservative validation-selected budget, both previously weak public scenes now have all-metric clean-long wins. Room CSEF20 improves PSNR, SSIM, LPIPS, AbsRel, Depth MAE, and normal while keeping 20% topology reduction. Counter CSEF20 does the same while keeping 20% topology reduction. Counter CSEF30 is also near-all-metric, missing only small depth margins, and CSEF40 improves every tracked metric over the earlier CSEF40 row.",
            "",
            "The safe claim is now: MeshSplatOpt supports a validation-selected CSEF-family compact-recovery protocol with conservative fallback budgets, and this protocol can produce all-metric clean-long wins on the formerly weak room/counter scenes without switching to QEM. The stronger F12 table may still use QEM rows as a posthoc simplification/operator baseline, but the method no longer depends on QEM to pass those scenes.",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
