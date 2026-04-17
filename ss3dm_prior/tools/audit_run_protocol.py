"""Audit SS3DM prior run metadata for split-protocol issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ss3dm_prior.utils.io import load_yaml


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_yaml(path)
    return data if isinstance(data, dict) else None


def _load_checkpoint_run_config(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = payload.get("run_config")
    return run_config if isinstance(run_config, dict) else None


def _resolve_run_dir(target_path: Path) -> tuple[Path | None, Path | None]:
    if target_path.is_dir():
        return target_path, None
    if target_path.is_file():
        checkpoint_path = target_path
        if checkpoint_path.parent.name == "checkpoints":
            return checkpoint_path.parent.parent, checkpoint_path
        return None, checkpoint_path
    raise FileNotFoundError(f"Target does not exist: {target_path}")


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_flag(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _extract_protocol_state(
    *,
    resolved_train_config: dict[str, Any] | None,
    run_metadata_json: dict[str, Any] | None,
    resolved_run_metadata_json: dict[str, Any] | None,
    checkpoint_run_config: dict[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_train = checkpoint_run_config.get("train_config", {}) if checkpoint_run_config else {}
    checkpoint_cli = checkpoint_run_config.get("cli", {}) if checkpoint_run_config else {}
    run_metadata_cli = (
        run_metadata_json.get("cli", {})
        if run_metadata_json is not None and isinstance(run_metadata_json.get("cli"), dict)
        else {}
    )
    resolved_run_cli = resolved_run_metadata_json if resolved_run_metadata_json else {}

    split_config_path = _first_present(
        checkpoint_cli.get("split_config"),
        resolved_run_cli.get("split_config"),
        run_metadata_cli.get("split_config"),
        run_metadata_json.get("split_config") if run_metadata_json else None,
    )

    return {
        "debug_use_all_patches_for_train_val": _normalize_flag(
            _first_present(
                resolved_train_config.get("debug_use_all_patches_for_train_val") if resolved_train_config else None,
                checkpoint_train.get("debug_use_all_patches_for_train_val"),
            )
        ),
        "allow_debug_split_override": _normalize_flag(
            _first_present(
                resolved_train_config.get("allow_debug_split_override") if resolved_train_config else None,
                checkpoint_train.get("allow_debug_split_override"),
            )
        ),
        "allow_split_fallback": _normalize_flag(
            _first_present(
                resolved_train_config.get("allow_split_fallback") if resolved_train_config else None,
                checkpoint_train.get("allow_split_fallback"),
            )
        ),
        "split_config_path": str(split_config_path) if split_config_path else None,
    }


def _protocol_state_complete(protocol_state: dict[str, Any]) -> bool:
    return all(
        protocol_state.get(key) is not None
        for key in (
            "debug_use_all_patches_for_train_val",
            "allow_debug_split_override",
            "allow_split_fallback",
            "split_config_path",
        )
    )


def audit_run_protocol(target: str | Path) -> dict[str, Any]:
    target_path = Path(target).expanduser().resolve()
    run_dir, checkpoint_path = _resolve_run_dir(target_path)

    resolved_train_config = (
        _read_yaml_if_exists(run_dir / "configs" / "resolved_train_config.yaml") if run_dir is not None else None
    )
    run_metadata_json = _read_json_if_exists(run_dir / "configs" / "run_metadata.json") if run_dir is not None else None
    resolved_run_metadata_json = (
        _read_json_if_exists(run_dir / "configs" / "resolved_run_metadata.json") if run_dir is not None else None
    )

    protocol_state = _extract_protocol_state(
        resolved_train_config=resolved_train_config,
        run_metadata_json=run_metadata_json,
        resolved_run_metadata_json=resolved_run_metadata_json,
        checkpoint_run_config=None,
    )
    checkpoint_run_config = None
    if checkpoint_path is not None and not _protocol_state_complete(protocol_state):
        checkpoint_run_config = _load_checkpoint_run_config(checkpoint_path)
        protocol_state = _extract_protocol_state(
            resolved_train_config=resolved_train_config,
            run_metadata_json=run_metadata_json,
            resolved_run_metadata_json=resolved_run_metadata_json,
            checkpoint_run_config=checkpoint_run_config,
        )

    split_data = None
    split_config_path = protocol_state["split_config_path"]
    if split_config_path is not None:
        candidate_split_path = Path(split_config_path).expanduser()
        if not candidate_split_path.is_absolute() and run_dir is not None:
            candidate_split_path = (run_dir / candidate_split_path).resolve()
        else:
            candidate_split_path = candidate_split_path.resolve()
        if candidate_split_path.exists():
            split_data = _read_yaml_if_exists(candidate_split_path)
            protocol_state["split_config_path"] = str(candidate_split_path)

    protocol_warnings: list[str] = []
    if protocol_state["debug_use_all_patches_for_train_val"] is None:
        protocol_warnings.append("Could not verify `debug_use_all_patches_for_train_val` from saved metadata.")
    elif protocol_state["debug_use_all_patches_for_train_val"]:
        protocol_warnings.append(
            "debug leakage detected: `debug_use_all_patches_for_train_val=true` mixes train/val from the full patch bank."
        )

    if protocol_state["allow_debug_split_override"] is None:
        protocol_warnings.append("Could not verify `allow_debug_split_override` from saved metadata.")
    elif protocol_state["allow_debug_split_override"]:
        protocol_warnings.append(
            "`allow_debug_split_override=true` permits bypassing the declared split protocol for train/val."
        )

    if protocol_state["allow_split_fallback"] is None:
        protocol_warnings.append("Could not verify `allow_split_fallback` from saved metadata.")
    elif protocol_state["allow_split_fallback"]:
        protocol_warnings.append(
            "`allow_split_fallback=true` permits a debug all-patch fallback when the formal split is empty."
        )

    if split_config_path is None:
        protocol_warnings.append("Split config path is unavailable in saved metadata.")
    elif split_data is None:
        protocol_warnings.append(f"Split config could not be loaded from `{split_config_path}`.")

    expected_train_towns = list(split_data.get("train_towns", [])) if split_data else []
    expected_val_towns = list(split_data.get("val_towns", [])) if split_data else []
    expected_test_towns = list(split_data.get("test_towns", [])) if split_data else []

    overlaps = {
        "train/val": sorted(set(expected_train_towns) & set(expected_val_towns)),
        "train/test": sorted(set(expected_train_towns) & set(expected_test_towns)),
        "val/test": sorted(set(expected_val_towns) & set(expected_test_towns)),
    }
    for overlap_name, overlap_values in overlaps.items():
        if overlap_values:
            protocol_warnings.append(
                f"Split config has overlapping towns for {overlap_name}: {', '.join(overlap_values)}."
            )

    protocol_summary = {
        "target_path": str(target_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "debug_use_all_patches_for_train_val": protocol_state["debug_use_all_patches_for_train_val"],
        "allow_debug_split_override": protocol_state["allow_debug_split_override"],
        "allow_split_fallback": protocol_state["allow_split_fallback"],
        "split_config_path": protocol_state["split_config_path"],
        "expected_train_towns": expected_train_towns,
        "expected_val_towns": expected_val_towns,
        "expected_test_towns": expected_test_towns,
        "metadata_sources": {
            "resolved_train_config": str(run_dir / "configs" / "resolved_train_config.yaml")
            if run_dir is not None and (run_dir / "configs" / "resolved_train_config.yaml").exists()
            else None,
            "run_metadata_json": str(run_dir / "configs" / "run_metadata.json")
            if run_dir is not None and (run_dir / "configs" / "run_metadata.json").exists()
            else None,
            "resolved_run_metadata_json": str(run_dir / "configs" / "resolved_run_metadata.json")
            if run_dir is not None and (run_dir / "configs" / "resolved_run_metadata.json").exists()
            else None,
            "checkpoint_run_config": str(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else None,
        },
    }
    return {
        "protocol_valid": len(protocol_warnings) == 0,
        "protocol_warnings": protocol_warnings,
        "protocol_summary": protocol_summary,
    }


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a saved SS3DM prior run for split-protocol warnings.")
    parser.add_argument("target", help="Training output directory or checkpoint path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)
    result = audit_run_protocol(args.target)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
