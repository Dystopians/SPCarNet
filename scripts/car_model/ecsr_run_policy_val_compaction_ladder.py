#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}


@dataclass(frozen=True)
class Metrics:
    psnr: float
    ssim: float
    lpips: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Metrics":
        return cls(
            psnr=float(payload["PSNR"]),
            ssim=float(payload["SSIM"]),
            lpips=float(payload["LPIPS"]),
        )

    def to_dict(self) -> dict[str, float]:
        return {"PSNR": self.psnr, "SSIM": self.ssim, "LPIPS": self.lpips}


def _run(cmd: list[str], *, gpu: int | None, log_path: Path, cwd: Path = ROOT) -> None:
    env = os.environ.copy()
    if gpu is not None and int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _scene_source_model(source_root: Path, scene: str, policy_tag: str) -> Path:
    model = source_root / scene / policy_tag / "compact_model"
    if not (model / "point_cloud").is_dir():
        raise FileNotFoundError(f"missing source compact model for {scene}: {model}")
    return model


def _image_set(scene: str, outdoor_images: str, indoor_images: str) -> str:
    return outdoor_images if scene in OUTDOOR_SCENES else indoor_images


def _method_name(prefix: str, ratio: float) -> str:
    return f"{prefix}_r{int(round(ratio * 10000)):04d}"


def _load_eval_metrics(path: Path, method: str) -> Metrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method not in payload:
        raise KeyError(f"method {method!r} missing from {path}")
    return Metrics.from_payload(payload[method])


def _read_audit(model: Path) -> dict[str, Any]:
    audit_path = model / "topology_audit.json"
    if not audit_path.is_file():
        return {}
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _checkpoint_file(model: Path, iteration: int) -> Path:
    return model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"


def _render_and_eval(
    *,
    scene: str,
    model_path: Path,
    source_path: Path,
    split_file: Path,
    iteration: int,
    images: str,
    method_name: str,
    output_json: Path,
    per_view_json: Path,
    gpu: int,
    log_path: Path,
    no_depth: bool,
    skip_failed_views: bool,
) -> Metrics:
    if output_json.is_file():
        try:
            return _load_eval_metrics(output_json, method_name)
        except Exception:
            pass
    render_cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "-i",
        images,
        "--resolution",
        "-1",
        "--eval",
        "--split_strategy",
        "file",
        "--split_file",
        str(split_file),
        "--iteration",
        str(iteration),
        "--method_name",
        method_name,
        "--skip_train",
        "--quiet",
    ]
    if no_depth:
        render_cmd.append("--no_depth")
    if skip_failed_views:
        render_cmd.append("--skip_failed_views")
    _run(render_cmd, gpu=gpu, log_path=log_path)

    eval_cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model_path),
        "--split",
        "test",
        "--methods",
        method_name,
        "--output",
        str(output_json),
        "--per_view_output",
        str(per_view_json),
    ]
    _run(eval_cmd, gpu=gpu, log_path=log_path)
    return _load_eval_metrics(output_json, method_name)


def _apply_compaction(
    *,
    source_model: Path,
    output_model: Path,
    iteration: int,
    selector_mode: str,
    ratio: float,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    existing_audit = _read_audit(output_model)
    if existing_audit and _checkpoint_file(output_model, iteration).is_file():
        return existing_audit
    selector_dir = output_model / "selector"
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py",
        "--source_model",
        str(source_model),
        "--iteration",
        str(iteration),
        "--output_model",
        str(output_model),
        "--selector_mode",
        selector_mode,
        "--target_prune_fraction",
        f"{ratio:.8f}",
        "--selector_out_dir",
        str(selector_dir),
        "--seed",
        str(seed),
    ]
    _run(cmd, gpu=None, log_path=log_path)
    return _read_audit(output_model)


def _accept_candidate(
    baseline: Metrics,
    candidate: Metrics,
    *,
    min_delta_psnr: float,
    min_delta_ssim: float,
    max_delta_lpips: float,
) -> tuple[bool, dict[str, float]]:
    delta = {
        "dPSNR": candidate.psnr - baseline.psnr,
        "dSSIM": candidate.ssim - baseline.ssim,
        "dLPIPS": candidate.lpips - baseline.lpips,
    }
    accepted = (
        delta["dPSNR"] >= min_delta_psnr
        and delta["dSSIM"] >= min_delta_ssim
        and delta["dLPIPS"] <= max_delta_lpips
    )
    return accepted, delta


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['scene']} | {row['ratio']:.4f} | {row['accepted']} | "
        f"{row['metrics']['PSNR']:.4f} | {row['metrics']['SSIM']:.5f} | {row['metrics']['LPIPS']:.5f} | "
        f"{row['delta']['dPSNR']:+.4f} | {row['delta']['dSSIM']:+.5f} | {row['delta']['dLPIPS']:+.5f} | "
        f"{row.get('additional_removed_fraction', 0.0):.4f} | {row.get('model_path', '')} |"
    )


def _write_scene_markdown(scene_dir: Path, scene: str, rows: list[dict[str, Any]], selected: dict[str, Any] | None) -> None:
    lines = [
        f"# ECSR Policy-Val Compaction Ladder: {scene}",
        "",
        "This run fixes the selector and ratio grid before looking at held-out paper-test images.",
        "The policy-val split is carved from the training COLMAP cameras, so the final test set is not used for policy selection.",
        "",
        "| Scene | Ratio | Accepted | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | Add. Removed | Model |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_format_row(row) for row in rows)
    lines.extend(["", "## Selected", ""])
    if selected is None:
        lines.append("No candidate passed the fixed policy-val guardrail; keep the source Compact-ELA/SOR model.")
    else:
        lines.append(
            f"Selected ratio `{selected['ratio']:.4f}` with policy-val deltas "
            f"dPSNR `{selected['delta']['dPSNR']:+.4f}`, "
            f"dSSIM `{selected['delta']['dSSIM']:+.5f}`, "
            f"dLPIPS `{selected['delta']['dLPIPS']:+.5f}`."
        )
    scene_dir.joinpath("policy_val_ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wandb_init(args: argparse.Namespace):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        print(f"[wandb] disabled: {exc}", file=sys.stderr)
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        config=vars(args),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fixed policy-val compaction ladder on Compact-ELA/SOR models."
    )
    parser.add_argument("--scenes", required=True, help="Comma-separated Mip-NeRF360 scene list.")
    parser.add_argument("--data_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument(
        "--source_root",
        default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k",
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument(
        "--split_root",
        default="outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file",
    )
    parser.add_argument("--out_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v1")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--ratios", default="0.005,0.010,0.020")
    parser.add_argument("--selector_mode", default="csef_low_evidence_boundary_protected")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help=(
            "Physical GPU id to set inside subprocesses. Use -1 to preserve the caller's "
            "CUDA_VISIBLE_DEVICES, which is safer for parallel launchers."
        ),
    )
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--method_prefix", default="ours_26000_policyval_ladder")
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--min_delta_psnr", type=float, default=-0.03)
    parser.add_argument("--min_delta_ssim", type=float, default=-0.0015)
    parser.add_argument("--max_delta_lpips", type=float, default=0.0020)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="phase_f_policy_val_ladder")
    parser.add_argument("--wandb_name", default="phase_f_policy_val_ladder")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    if not scenes:
        raise ValueError("no scenes provided")
    if not ratios:
        raise ValueError("no ratios provided")

    source_root = Path(args.source_root)
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run = _wandb_init(args)
    all_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any] | None] = {}
    try:
        for scene in scenes:
            scene_dir = out_root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            log_path = scene_dir / "policy_val_ladder.log"
            source_model = _scene_source_model(source_root, scene, args.policy_tag)
            source_path = data_root / scene
            split_file = split_root / scene / "split_file.json"
            if not split_file.is_file():
                raise FileNotFoundError(f"missing policy-val split file: {split_file}")
            images = _image_set(scene, args.outdoor_images, args.indoor_images)

            baseline_method = _method_name(f"{args.method_prefix}_source", 0.0)
            baseline_json = scene_dir / "source_policy_val_results.json"
            baseline_per_view = scene_dir / "source_policy_val_per_view.json"
            baseline_metrics = _render_and_eval(
                scene=scene,
                model_path=source_model,
                source_path=source_path,
                split_file=split_file,
                iteration=args.iteration,
                images=images,
                method_name=baseline_method,
                output_json=baseline_json,
                per_view_json=baseline_per_view,
                gpu=args.gpu,
                log_path=log_path,
                no_depth=args.no_depth,
                skip_failed_views=args.skip_failed_views,
            )

            rows: list[dict[str, Any]] = []
            for ratio in ratios:
                method = _method_name(args.method_prefix, ratio)
                candidate_dir = scene_dir / f"ratio_{int(round(ratio * 10000)):04d}" / "compact_model"
                audit = _apply_compaction(
                    source_model=source_model,
                    output_model=candidate_dir,
                    iteration=args.iteration,
                    selector_mode=args.selector_mode,
                    ratio=ratio,
                    seed=args.seed,
                    log_path=log_path,
                )
                metrics_json = candidate_dir / "policy_val_results.json"
                per_view_json = candidate_dir / "policy_val_per_view.json"
                candidate_metrics = _render_and_eval(
                    scene=scene,
                    model_path=candidate_dir,
                    source_path=source_path,
                    split_file=split_file,
                    iteration=args.iteration,
                    images=images,
                    method_name=method,
                    output_json=metrics_json,
                    per_view_json=per_view_json,
                    gpu=args.gpu,
                    log_path=log_path,
                    no_depth=args.no_depth,
                    skip_failed_views=args.skip_failed_views,
                )
                topo_ok = int(audit.get("invalid_index_count", 0)) == 0 and int(audit.get("degenerate_face_count", 0)) == 0
                accepted, delta = _accept_candidate(
                    baseline_metrics,
                    candidate_metrics,
                    min_delta_psnr=args.min_delta_psnr,
                    min_delta_ssim=args.min_delta_ssim,
                    max_delta_lpips=args.max_delta_lpips,
                )
                accepted = bool(accepted and topo_ok)
                row = {
                    "scene": scene,
                    "ratio": float(ratio),
                    "accepted": accepted,
                    "topology_ok": topo_ok,
                    "metrics": candidate_metrics.to_dict(),
                    "baseline_metrics": baseline_metrics.to_dict(),
                    "delta": delta,
                    "audit": audit,
                    "additional_removed_fraction": float(audit.get("removed_fraction", 0.0)),
                    "model_path": str(candidate_dir),
                    "method": method,
                }
                rows.append(row)
                all_rows.append(row)
                if run is not None:
                    run.log(
                        {
                            f"{scene}/ratio": float(ratio),
                            f"{scene}/accepted": int(accepted),
                            f"{scene}/policy_val_psnr": candidate_metrics.psnr,
                            f"{scene}/policy_val_ssim": candidate_metrics.ssim,
                            f"{scene}/policy_val_lpips": candidate_metrics.lpips,
                            f"{scene}/dpsnr": delta["dPSNR"],
                            f"{scene}/dssim": delta["dSSIM"],
                            f"{scene}/dlpips": delta["dLPIPS"],
                            f"{scene}/additional_removed_fraction": float(audit.get("removed_fraction", 0.0)),
                        }
                    )

            accepted_rows = [row for row in rows if row["accepted"]]
            selected = None
            if accepted_rows:
                selected = max(
                    accepted_rows,
                    key=lambda row: (
                        float(row.get("additional_removed_fraction", 0.0)),
                        float(row["delta"]["dPSNR"]),
                        -float(row["delta"]["dLPIPS"]),
                    ),
                )
            selections[scene] = selected
            scene_dir.joinpath("summary.json").write_text(
                json.dumps(
                    {
                        "scene": scene,
                        "source_model": str(source_model),
                        "policy_val_baseline": baseline_metrics.to_dict(),
                        "rows": rows,
                        "selected": selected,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_scene_markdown(scene_dir, scene, rows, selected)

        global_payload = {"rows": all_rows, "selections": selections}
        summary_stem = "summary_" + "_".join(scenes)
        out_root.joinpath(f"{summary_stem}.json").write_text(
            json.dumps(global_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        out_root.joinpath("summary.json").write_text(
            json.dumps(global_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# ECSR Phase-F Policy-Val Compaction Ladder",
            "",
            "Fixed-policy internal-validation compaction sweep. Acceptance uses only the training-derived policy-val split.",
            "",
            "| Scene | Ratio | Accepted | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | Add. Removed | Model |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        lines.extend(_format_row(row) for row in all_rows)
        lines.extend(["", "## Selected Models", ""])
        for scene in scenes:
            selected = selections.get(scene)
            if selected is None:
                lines.append(f"- `{scene}`: keep source Compact-ELA/SOR; no extra ratio passed policy-val guardrails.")
            else:
                lines.append(
                    f"- `{scene}`: `{selected['model_path']}` at ratio `{selected['ratio']:.4f}` "
                    f"(additional removed `{selected.get('additional_removed_fraction', 0.0):.4f}`)."
                )
        summary_md = "\n".join(lines) + "\n"
        out_root.joinpath(f"{summary_stem}.md").write_text(summary_md, encoding="utf-8")
        out_root.joinpath("summary.md").write_text(summary_md, encoding="utf-8")
    finally:
        if run is not None:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
