"""Run a paper-style SS3DM prior ablation suite and aggregate the results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.tools.aggregate_ablation_results import aggregate_suite_results
from ss3dm_prior.utils.io import dump_json, dump_yaml, load_json, load_yaml


ROOT = Path("/data2/peilincai/mesh-splatting")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    model_base_path: Path
    train_base_path: Path
    model_overrides: dict[str, Any]
    train_overrides: dict[str, Any]
    debug_train_overrides: dict[str, Any]
    checkpoint_candidates: tuple[str, ...]
    patch_cache_role: str = "main"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _run_command(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT), env=env)
    lines = [
        f"command: {' '.join(command)}",
        f"returncode: {result.returncode}",
        "",
        "stdout:",
        result.stdout,
        "",
        "stderr:",
        result.stderr,
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed, see log: {log_path}")


def _resolve_checkpoint(checkpoint_dir: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = checkpoint_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} matching {candidates}")


def _write_patch(
    patch_path: Path,
    *,
    patch_id: str,
    town_id: str,
    sequence_id: str,
    tile_id: int,
    center_x: float,
    center_y: float,
    intrinsic_target: float,
    camera_support_count: int,
    lidar_support_count: int,
    rng_seed: int,
) -> PatchIndexRecord:
    rng = np.random.default_rng(rng_seed)
    clean_points = (rng.normal(size=(64, 3)) * 0.08).astype(np.float32)
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (64, 1))
    observed_points = clean_points[:32]
    surface_query_points = clean_points[:16]
    free_query_points = np.clip(clean_points[16:32] + np.asarray([0.0, 0.0, 0.2], dtype=np.float32), -1.0, 1.0)
    unknown_query_points = rng.uniform(-0.4, 0.4, size=(12, 3)).astype(np.float32)
    query_points_all = np.concatenate([surface_query_points, free_query_points, unknown_query_points], axis=0)
    query_labels_all = np.concatenate(
        [
            np.ones((len(surface_query_points),), dtype=np.int8),
            np.zeros((len(free_query_points),), dtype=np.int8),
            np.zeros((len(unknown_query_points),), dtype=np.int8),
        ]
    )
    query_ignore_mask = np.concatenate(
        [
            np.zeros((len(surface_query_points),), dtype=bool),
            np.zeros((len(free_query_points),), dtype=bool),
            np.ones((len(unknown_query_points),), dtype=bool),
        ]
    )
    visible_surface_fraction = 0.25 + 0.05 * float(rng_seed % 5)
    free_space_fraction = float(len(free_query_points) / len(query_points_all))
    unknown_fraction = float(len(unknown_query_points) / len(query_points_all))
    patch_center = np.asarray([center_x, center_y, 0.0], dtype=np.float32)
    difficulty_components = {"observed_to_clean_nn_error": round(0.18 + 0.03 * rng_seed, 4)}

    sample = TeacherPatchSample(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        patch_center_world=patch_center,
        patch_radius_m=3.0,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_id=patch_id,
        num_local_faces=32,
        num_observed_points_raw=32,
        teacher_area_local=1.0,
        source_town_mesh_cache_dir="synthetic_mesh_cache",
        source_sequence_observed_cache="synthetic_observed_cache",
        patch_cache_format_version=2,
        surface_query_points=surface_query_points,
        surface_query_labels=np.ones((len(surface_query_points),), dtype=np.int8),
        free_query_points=free_query_points,
        free_query_labels=np.zeros((len(free_query_points),), dtype=np.int8),
        unknown_query_points=unknown_query_points,
        query_points_all=query_points_all,
        query_labels_all=query_labels_all,
        query_ignore_mask=query_ignore_mask,
        camera_support_count=camera_support_count,
        lidar_support_count=lidar_support_count,
        visible_surface_fraction=visible_surface_fraction,
        free_space_fraction=free_space_fraction,
        unknown_fraction=unknown_fraction,
        intrinsic_patch_difficulty_target=intrinsic_target,
        difficulty_components_json=difficulty_components,
        metadata={"planarity_hint": 0.5},
    )
    sample.save(patch_path)
    return PatchIndexRecord(
        patch_id=patch_id,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_file=str(patch_path),
        num_local_faces=32,
        num_observed_points_raw=32,
        num_clean_points=64,
        num_observed_points=32,
        teacher_area_local=1.0,
        planarity_hint=0.5,
        patch_cache_format_version=2,
        num_surface_query_points=len(surface_query_points),
        num_free_query_points=len(free_query_points),
        num_unknown_query_points=len(unknown_query_points),
        camera_support_count=camera_support_count,
        lidar_support_count=lidar_support_count,
        visible_surface_fraction=visible_surface_fraction,
        free_space_fraction=free_space_fraction,
        unknown_fraction=unknown_fraction,
        intrinsic_patch_difficulty_target=intrinsic_target,
        difficulty_components_json=difficulty_components,
    )


def _build_debug_inputs(root_dir: Path) -> dict[str, str]:
    inputs_dir = root_dir / "debug_inputs"
    patch_cache_main = inputs_dir / "patch_cache_v2"
    patch_cache_lidar = inputs_dir / "patch_cache_v2_lidar_only"
    manifest_path = inputs_dir / "synthetic_manifest.json"
    data_config_path = inputs_dir / "data.yaml"
    split_config_path = inputs_dir / "split.yaml"

    patch_specs = [
        ("TownTrain", "TownTrain__seq0", 0.0, 0.0, 0.20, 2, 3),
        ("TownTrain", "TownTrain__seq0", 1.0, 0.0, 0.28, 2, 3),
        ("TownVal", "TownVal__seq0", 0.0, 1.0, 0.34, 2, 3),
        ("TownVal", "TownVal__seq0", 1.0, 1.0, 0.42, 2, 4),
        ("TownEval", "TownEval__seqA", 0.0, 2.0, 0.50, 3, 4),
        ("TownEval", "TownEval__seqA", 1.0, 2.0, 0.56, 3, 4),
        ("TownEval", "TownEval__seqB", 0.0, 3.0, 0.62, 3, 5),
        ("TownEval", "TownEval__seqB", 1.0, 3.0, 0.68, 3, 5),
        ("TownEval", "TownEval__seqC", 0.0, 4.0, 0.74, 4, 5),
        ("TownEval", "TownEval__seqC", 1.0, 4.0, 0.82, 4, 5),
    ]
    for cache_dir, force_camera_zero in ((patch_cache_main, False), (patch_cache_lidar, True)):
        records: list[PatchIndexRecord] = []
        for idx, (town_id, sequence_id, cx, cy, intrinsic_target, camera_support, lidar_support) in enumerate(patch_specs):
            patch_dir = cache_dir / town_id / sequence_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            records.append(
                _write_patch(
                    patch_dir / f"patch_{idx:03d}.npz",
                    patch_id=f"{sequence_id}__patch_{idx:03d}",
                    town_id=town_id,
                    sequence_id=sequence_id,
                    tile_id=idx,
                    center_x=cx,
                    center_y=cy,
                    intrinsic_target=intrinsic_target,
                    camera_support_count=0 if force_camera_zero else camera_support,
                    lidar_support_count=lidar_support,
                    rng_seed=idx,
                )
            )
        write_patch_index_jsonl(cache_dir / "patch_index.jsonl", records)

    dump_json(manifest_path, {"dataset": "synthetic_ablation_debug", "created_at_utc": datetime.now(timezone.utc).isoformat()})
    dump_yaml(data_config_path, {"dataset": {"name": "synthetic_ablation_debug"}})
    dump_yaml(
        split_config_path,
        {
            "split_name": "synthetic_ablation_debug",
            "strategy": "town_holdout",
            "unit_of_split": "sequence",
            "forbid_random_patch_split": True,
            "forbid_random_frame_split": True,
            "train_towns": ["TownTrain"],
            "val_towns": ["TownVal"],
            "test_towns": ["TownEval"],
        },
    )
    return {
        "data_config": str(data_config_path),
        "split_config": str(split_config_path),
        "patch_cache_dir": str(patch_cache_main),
        "lidar_only_patch_cache_dir": str(patch_cache_lidar),
        "manifest_path": str(manifest_path),
    }


def _build_variants(*, include_optional_lidar_only: bool, lidar_model_base: Path, hybrid_model_base: Path, strict_train_base: Path, hybrid_train_base: Path) -> list[AblationVariant]:
    variants = [
        AblationVariant(
            name="legacy_v1_strict",
            description="Strict legacy v1 baseline without visibility, VQ, or intrinsic difficulty supervision.",
            model_base_path=lidar_model_base,
            train_base_path=strict_train_base,
            model_overrides={},
            train_overrides={},
            debug_train_overrides={},
            checkpoint_candidates=("best_gain.pt", "best_recon.pt", "last.pt"),
        ),
        AblationVariant(
            name="v2_no_visibility",
            description="Hybrid v2 without visibility, free-space, intrinsic difficulty, or VQ supervision.",
            model_base_path=hybrid_model_base,
            train_base_path=hybrid_train_base,
            model_overrides={
                "model": {
                    "use_vector_quantization": False,
                },
                "loss_weights": {
                    "intrinsic_difficulty_loss": 0.0,
                    "intrinsic_difficulty_pairwise_weight": 0.0,
                    "occupancy_bce_loss": 0.0,
                    "free_space_violation_loss": 0.0,
                    "vq_commitment_loss": 0.0,
                    "prototype_diversity_loss": 0.0,
                },
            },
            train_overrides={
                "train": {
                    "curriculum": {
                        "occupancy_start_epoch": 999,
                        "intrinsic_start_epoch": 999,
                        "vq_start_epoch": 999,
                        "prototype_start_epoch": 999,
                    }
                }
            },
            debug_train_overrides={},
            checkpoint_candidates=("best_composite.pt", "best_gain.pt", "best_recon.pt", "last.pt"),
        ),
        AblationVariant(
            name="v2_visibility_no_vq",
            description="Hybrid v2 with visibility and free-space supervision but vector quantization disabled.",
            model_base_path=hybrid_model_base,
            train_base_path=hybrid_train_base,
            model_overrides={
                "model": {
                    "use_vector_quantization": False,
                },
                "loss_weights": {
                    "intrinsic_difficulty_loss": 0.0,
                    "intrinsic_difficulty_pairwise_weight": 0.0,
                    "vq_commitment_loss": 0.0,
                    "prototype_diversity_loss": 0.0,
                },
            },
            train_overrides={
                "train": {
                    "curriculum": {
                        "intrinsic_start_epoch": 999,
                        "vq_start_epoch": 999,
                        "prototype_start_epoch": 999,
                    }
                }
            },
            debug_train_overrides={
                "train": {
                    "curriculum": {
                        "occupancy_start_epoch": 0,
                    }
                }
            },
            checkpoint_candidates=("best_composite.pt", "best_visibility.pt", "best_gain.pt", "best_recon.pt", "last.pt"),
        ),
        AblationVariant(
            name="v2_visibility_plus_vq",
            description="Hybrid v2 with visibility and free-space supervision plus VQ/prototype memory, but no intrinsic difficulty loss.",
            model_base_path=hybrid_model_base,
            train_base_path=hybrid_train_base,
            model_overrides={
                "loss_weights": {
                    "intrinsic_difficulty_loss": 0.0,
                    "intrinsic_difficulty_pairwise_weight": 0.0,
                }
            },
            train_overrides={
                "train": {
                    "curriculum": {
                        "intrinsic_start_epoch": 999,
                    }
                }
            },
            debug_train_overrides={
                "train": {
                    "curriculum": {
                        "occupancy_start_epoch": 0,
                        "vq_start_epoch": 0,
                        "prototype_start_epoch": 0,
                    }
                }
            },
            checkpoint_candidates=("best_composite.pt", "best_visibility.pt", "best_gain.pt", "best_recon.pt", "last.pt"),
        ),
        AblationVariant(
            name="v2_full",
            description="Full hybrid v2 with visibility, intrinsic difficulty, VQ, and occupancy/free-space supervision.",
            model_base_path=hybrid_model_base,
            train_base_path=hybrid_train_base,
            model_overrides={},
            train_overrides={},
            debug_train_overrides={
                "train": {
                    "curriculum": {
                        "occupancy_start_epoch": 0,
                        "intrinsic_start_epoch": 0,
                        "vq_start_epoch": 0,
                        "prototype_start_epoch": 0,
                    }
                }
            },
            checkpoint_candidates=("best_composite.pt", "best_visibility.pt", "best_gain.pt", "best_recon.pt", "last.pt"),
        ),
    ]
    if include_optional_lidar_only:
        variants.append(
            AblationVariant(
                name="v2_no_camera_visibility",
                description="Optional lidar-only visibility ablation using a patch cache with camera support counts removed.",
                model_base_path=hybrid_model_base,
                train_base_path=hybrid_train_base,
                model_overrides={},
                train_overrides={},
                debug_train_overrides={
                    "train": {
                        "curriculum": {
                            "occupancy_start_epoch": 0,
                            "intrinsic_start_epoch": 0,
                            "vq_start_epoch": 0,
                            "prototype_start_epoch": 0,
                        }
                    }
                },
                checkpoint_candidates=("best_composite.pt", "best_visibility.pt", "best_gain.pt", "best_recon.pt", "last.pt"),
                patch_cache_role="lidar_only",
            )
        )
    return variants


def _write_variant_configs(
    variant: AblationVariant,
    *,
    config_root: Path,
    debug_mode: bool,
) -> tuple[Path, Path]:
    model_config = load_yaml(variant.model_base_path)
    train_config = load_yaml(variant.train_base_path)
    train_config = _deep_update(
        train_config,
        {
            "train": {
                "wandb_enable": False,
                "wandb_mode": "disabled",
            }
        },
    )
    if debug_mode:
        train_config = _deep_update(
            train_config,
            {
                "train": {
                    "seed": 0,
                    "epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "amp": False,
                    "log_interval": 1,
                    "val_interval": 1,
                    "save_interval": 1,
                    "max_visualization_examples": 1,
                    "step_visualization_interval_steps": 1,
                    "step_visualization_num_examples": 1,
                    "debug_use_all_patches_for_train_val": False,
                    "allow_debug_split_override": False,
                    "allow_split_fallback": False,
                    "debug_val_fraction": 0.5,
                    "hard_example_sampling": {"enable": False},
                    "curriculum": {
                        "warmup_epochs": 0,
                        "main_start_epoch": 0,
                    },
                }
            },
        )
    model_config = _deep_update(model_config, variant.model_overrides)
    train_config = _deep_update(train_config, variant.train_overrides)
    if debug_mode:
        train_config = _deep_update(train_config, variant.debug_train_overrides)
    config_dir = config_root / variant.name
    config_dir.mkdir(parents=True, exist_ok=True)
    model_path = config_dir / "model.yaml"
    train_path = config_dir / "train.yaml"
    dump_yaml(model_path, model_config)
    dump_yaml(train_path, train_config)
    return model_path, train_path


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a paper-style SS3DM prior ablation suite.")
    parser.add_argument("--output_dir", required=True, help="Root output directory for the suite.")
    parser.add_argument("--suite_name", default="v2_ablation_suite", help="Suite subdirectory name.")
    parser.add_argument("--data_config", default=str(ROOT / "configs/ss3dm_prior/data_default.yaml"), help="Base data config.")
    parser.add_argument("--split_config", default=str(ROOT / "configs/ss3dm_prior/splits/default_town_split.yaml"), help="Train/eval split config with train, val, and test towns.")
    parser.add_argument("--patch_cache_dir", default=None, help="Patch cache root for the main suite.")
    parser.add_argument("--lidar_only_patch_cache_dir", default=None, help="Optional lidar-only patch cache root for `v2_no_camera_visibility`.")
    parser.add_argument("--manifest_path", default=str(ROOT / "outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json"), help="Manifest path metadata.")
    parser.add_argument("--observed_cache_dir", default=str(ROOT / "outputs/ss3dm_prior/observed_cache"), help="Observed cache root metadata.")
    parser.add_argument("--town_mesh_cache_dir", default=str(ROOT / "outputs/ss3dm_prior/town_mesh_cache"), help="Town mesh cache root metadata.")
    parser.add_argument("--legacy_model_config", default=str(ROOT / "configs/ss3dm_prior/model_v7_gain.yaml"), help="Base model config for legacy ablations.")
    parser.add_argument("--hybrid_model_config", default=str(ROOT / "configs/ss3dm_prior/model_v8_hybrid.yaml"), help="Base model config for hybrid v2 ablations.")
    parser.add_argument("--strict_train_config", default=str(ROOT / "configs/ss3dm_prior/train_v8_strict.yaml"), help="Base train config for strict legacy ablations.")
    parser.add_argument("--hybrid_train_config", default=str(ROOT / "configs/ss3dm_prior/train_v8_hybrid.yaml"), help="Base train config for hybrid v2 ablations.")
    parser.add_argument("--variants", default=None, help="Comma-separated subset of variant names to run.")
    parser.add_argument("--wandb_mode", default="disabled", help="wandb mode passed to train/eval.")
    parser.add_argument("--debug_synthetic", action="store_true", help="Create a tiny synthetic patch cache and run a 1-epoch smoke ablation suite.")
    parser.add_argument("--include_optional_lidar_only", action="store_true", help="Also run `v2_no_camera_visibility` if a lidar-only patch cache is available.")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing train/eval outputs when present.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on the first variant failure instead of continuing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)
    suite_dir = Path(args.output_dir).expanduser().resolve() / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "data_config": args.data_config,
        "split_config": args.split_config,
        "patch_cache_dir": args.patch_cache_dir,
        "lidar_only_patch_cache_dir": args.lidar_only_patch_cache_dir,
        "manifest_path": args.manifest_path,
        "observed_cache_dir": args.observed_cache_dir,
        "town_mesh_cache_dir": args.town_mesh_cache_dir,
    }
    if args.debug_synthetic:
        input_paths = _deep_update(input_paths, _build_debug_inputs(suite_dir))
    if input_paths["patch_cache_dir"] is None:
        raise ValueError("`--patch_cache_dir` is required unless `--debug_synthetic` is used.")

    variants = _build_variants(
        include_optional_lidar_only=args.include_optional_lidar_only,
        lidar_model_base=Path(args.legacy_model_config).expanduser().resolve(),
        hybrid_model_base=Path(args.hybrid_model_config).expanduser().resolve(),
        strict_train_base=Path(args.strict_train_config).expanduser().resolve(),
        hybrid_train_base=Path(args.hybrid_train_config).expanduser().resolve(),
    )
    if args.variants:
        wanted = {item.strip() for item in args.variants.split(",") if item.strip()}
        variants = [variant for variant in variants if variant.name in wanted]
    if not variants:
        raise ValueError("No ablation variants selected.")

    manifest: dict[str, Any] = {
        "suite_name": args.suite_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "debug_synthetic": bool(args.debug_synthetic),
        "input_paths": input_paths,
        "variants": [],
    }

    config_root = suite_dir / "generated_configs"
    logs_root = suite_dir / "logs"
    runs_root = suite_dir / "runs"
    logs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        variant_entry: dict[str, Any] = {
            "variant": variant.name,
            "description": variant.description,
            "status": "pending",
            "train_output_dir": str((runs_root / variant.name / "train").resolve()),
            "eval_output_dir": str((runs_root / variant.name / "eval").resolve()),
        }
        manifest["variants"].append(variant_entry)
        try:
            if variant.patch_cache_role == "lidar_only":
                patch_cache_dir = input_paths.get("lidar_only_patch_cache_dir")
                if patch_cache_dir is None:
                    variant_entry["status"] = "skipped"
                    variant_entry["skip_reason"] = "No lidar-only patch cache was provided."
                    continue
            else:
                patch_cache_dir = input_paths["patch_cache_dir"]

            model_config_path, train_config_path = _write_variant_configs(
                variant,
                config_root=config_root,
                debug_mode=bool(args.debug_synthetic),
            )
            variant_entry["model_config_path"] = str(model_config_path)
            variant_entry["train_config_path"] = str(train_config_path)
            variant_entry["patch_cache_dir"] = str(Path(patch_cache_dir).expanduser().resolve())

            train_output_dir = Path(variant_entry["train_output_dir"])
            eval_output_root = Path(variant_entry["eval_output_dir"])
            checkpoint_dir = train_output_dir / "checkpoints"
            summary_path = eval_output_root / variant.name / "metrics_summary.json"

            if not (args.skip_existing and summary_path.exists() and checkpoint_dir.exists()):
                train_cmd = [
                    sys.executable,
                    "-m",
                    "ss3dm_prior.train",
                    "--data_config",
                    str(input_paths["data_config"]),
                    "--model_config",
                    str(model_config_path),
                    "--train_config",
                    str(train_config_path),
                    "--manifest_path",
                    str(input_paths["manifest_path"]),
                    "--observed_cache_dir",
                    str(input_paths["observed_cache_dir"]),
                    "--town_mesh_cache_dir",
                    str(input_paths["town_mesh_cache_dir"]),
                    "--patch_cache_dir",
                    str(patch_cache_dir),
                    "--split_config",
                    str(input_paths["split_config"]),
                    "--output_dir",
                    str(train_output_dir),
                    "--run_name",
                    variant.name,
                    "--wandb_mode",
                    args.wandb_mode,
                ]
                _run_command(train_cmd, log_path=logs_root / f"{variant.name}__train.log")

                selected_checkpoint = _resolve_checkpoint(checkpoint_dir, variant.checkpoint_candidates)
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "ss3dm_prior.eval",
                    "--checkpoint",
                    str(selected_checkpoint),
                    "--manifest_path",
                    str(input_paths["manifest_path"]),
                    "--patch_cache_dir",
                    str(patch_cache_dir),
                    "--split_config",
                    str(input_paths["split_config"]),
                    "--output_dir",
                    str(eval_output_root),
                    "--eval_name",
                    variant.name,
                    "--wandb_mode",
                    args.wandb_mode,
                ]
                _run_command(eval_cmd, log_path=logs_root / f"{variant.name}__eval.log")
            selected_checkpoint = _resolve_checkpoint(checkpoint_dir, variant.checkpoint_candidates)
            variant_entry["selected_checkpoint"] = str(selected_checkpoint.resolve())
            variant_entry["metrics_summary_path"] = str(summary_path.resolve())
            variant_entry["summary_metrics"] = load_json(summary_path)
            variant_entry["status"] = "completed"
        except Exception as exc:  # pragma: no cover - failure reporting path
            variant_entry["status"] = "failed"
            variant_entry["error"] = str(exc)
            if args.fail_fast:
                break

    manifest_path = suite_dir / "suite_manifest.json"
    dump_json(manifest_path, manifest)
    csv_path, md_path = aggregate_suite_results(manifest_path)
    print(f"suite_manifest_json: {manifest_path}")
    print(f"ablation_summary_csv: {csv_path}")
    print(f"ablation_summary_md: {md_path}")
    failed = [entry for entry in manifest["variants"] if entry.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
