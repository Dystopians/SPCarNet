#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _ratio_tag(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 10000.0)):04d}"


def _selected_model(policy_root: Path, scene: str, force_ratio: float | None = None) -> Path:
    if force_ratio is not None:
        model = policy_root / scene / _ratio_tag(force_ratio) / "compact_model"
        if not model.is_dir():
            raise FileNotFoundError(model)
        return model
    summary = _read_json(policy_root / scene / "summary.json")
    selected = summary.get("selected") or {}
    model_path = selected.get("model_path")
    if not model_path:
        raise RuntimeError(f"missing selected model in {policy_root / scene / 'summary.json'}")
    model = ROOT / model_path
    if not model.is_dir():
        raise FileNotFoundError(model)
    return model


def _count_bins(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_bins(item) for item in value)
    return 1


def _accepted_bin_fraction(report: dict[str, Any]) -> tuple[int, int, float]:
    calibrator = report.get("alpha_calibrator") or {}
    accepted = int(calibrator.get("accepted_bins") or 0)
    total = _count_bins(calibrator.get("accept_table") or [])
    fraction = float(accepted) / float(total) if total > 0 else 0.0
    return accepted, total, fraction


def _choose_source(args: argparse.Namespace, model: Path) -> tuple[str, dict[str, Any]]:
    adaptive_report = _read_json(model / "test" / args.adaptive_method / "ela_report.json")
    accepted, total, accepted_fraction = _accepted_bin_fraction(adaptive_report)
    mean_alpha = _num(adaptive_report.get("mean_alpha"), 0.0)
    active_fraction = _num(adaptive_report.get("mean_alpha_active_fraction"), 0.0)
    covered_fraction = _num(adaptive_report.get("mean_covered_fraction"), 0.0)
    stable = (
        accepted_fraction >= float(args.min_accepted_bin_fraction)
        and mean_alpha >= float(args.min_mean_alpha)
        and active_fraction >= float(args.min_active_fraction)
    )
    decision = {
        "policy": "Phase-H guarded adaptive alpha",
        "uses_test_gt": False,
        "adaptive_method": args.adaptive_method,
        "fallback_method": args.fallback_method,
        "min_accepted_bin_fraction": float(args.min_accepted_bin_fraction),
        "min_mean_alpha": float(args.min_mean_alpha),
        "min_active_fraction": float(args.min_active_fraction),
        "accepted_bins": accepted,
        "total_bins": total,
        "accepted_bin_fraction": accepted_fraction,
        "mean_alpha": mean_alpha,
        "mean_alpha_active_fraction": active_fraction,
        "mean_covered_fraction": covered_fraction,
        "stable_adaptive": bool(stable),
        "selected_method": args.adaptive_method if stable else args.fallback_method,
    }
    return str(decision["selected_method"]), decision


def _copy_method(src: Path, dst: Path, *, force: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} already exists; pass --force to refresh it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the Phase-H no-GT guarded adaptive-alpha ELA method.")
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--adaptive_method", default="ours_26000_phaseh_adaptalpha_full_ela")
    parser.add_argument("--fallback_method", default="ours_26000_phasef_alpha0875grid_ela")
    parser.add_argument("--output_method", default="ours_26000_phaseh_guarded_adaptalpha_ela")
    parser.add_argument("--min_accepted_bin_fraction", type=float, default=0.65)
    parser.add_argument("--min_mean_alpha", type=float, default=0.30)
    parser.add_argument("--min_active_fraction", type=float, default=0.55)
    parser.add_argument("--force_ratio", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    policy_root = ROOT / args.policy_root
    rows = []
    for scene in [item.strip() for item in str(args.scenes).replace(" ", ",").split(",") if item.strip()]:
        model = _selected_model(policy_root, scene, args.force_ratio)
        selected_method, decision = _choose_source(args, model)
        source_dir = model / "test" / selected_method
        out_dir = model / "test" / args.output_method
        row = {
            "scene": scene,
            "model": str(model.relative_to(ROOT)),
            **decision,
            "output_method": args.output_method,
        }
        rows.append(row)
        if not args.dry_run:
            _copy_method(source_dir, out_dir, force=bool(args.force))
            report_path = out_dir / "ela_report.json"
            report = _read_json(report_path)
            report["guarded_policy"] = row
            report["method_name"] = args.output_method
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {"args": vars(args), "rows": rows}
    summary_path = policy_root / f"{args.output_method}_guarded_decisions.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
