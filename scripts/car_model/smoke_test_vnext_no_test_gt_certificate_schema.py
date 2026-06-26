#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_NAME = "vnext_no_test_gt_manifest_certificate"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_source_record(record: dict[str, Any]) -> None:
    _assert(record.get("split") in {"train_fit", "train_policy_val", "target_test"}, record)
    _assert(record.get("role") in {"parent_render", "train_gt_rgb", "candidate_render", "surface_evidence"}, record)
    _assert(record.get("path"), record)
    _assert(record.get("sha256"), record)
    if record["split"] == "target_test":
        _assert(record["role"] != "train_gt_rgb", f"target/test GT source leaked into manifest: {record}")


def _validate_gate_block(decision: dict[str, Any], thresholds: dict[str, float]) -> None:
    gates = decision["policy_val_gates"]
    _assert(decision["thresholds"]["source_split"] == "train_policy_val", decision)
    _assert(gates["mse_direction"] in {"improved", "regressed"}, gates)
    _assert(gates["positive_view_fraction"] >= 0.0 and gates["positive_view_fraction"] <= 1.0, gates)
    _assert(gates["support_count"] >= 0, gates)
    _assert(gates["residual_variance"] >= 0.0, gates)
    _assert(gates["target_camera_support"] in {"supported", "oot"}, gates)
    _assert(gates["parent_candidate_distance"] >= 0.0, gates)
    if decision["accepted"]:
        _assert(gates["mean_psnr_gain"] >= thresholds["min_mean_psnr_gain"], gates)
        _assert(gates["mean_ssim_gain"] >= thresholds["min_mean_ssim_gain"], gates)
        _assert(gates["mean_lpips_gain"] >= thresholds["min_mean_lpips_gain"], gates)
        _assert(gates["positive_view_fraction"] >= thresholds["min_positive_view_fraction"], gates)
        _assert(gates["cvar20_mse_delta"] <= thresholds["max_cvar20_mse_delta"], gates)
        _assert(gates["support_count"] >= thresholds["min_support_count"], gates)
        _assert(gates["residual_variance"] <= thresholds["max_residual_variance"], gates)


def validate_manifest_and_certificate(manifest: dict[str, Any], certificate: dict[str, Any]) -> None:
    _assert(manifest["schema_name"] == SCHEMA_NAME, manifest)
    _assert(certificate["schema_name"] == SCHEMA_NAME, certificate)
    _assert(manifest["schema_version"] == SCHEMA_VERSION, manifest)
    _assert(certificate["schema_version"] == SCHEMA_VERSION, certificate)
    _assert(certificate["manifest_payload_sha256"] == manifest["manifest_payload_sha256"], certificate)
    _assert(manifest["manifest_payload_sha256"] == _manifest_payload_sha256(manifest), manifest)

    audit = manifest["no_test_gt_audit"]
    _assert(audit["heldout_test_gt_used_for_selection"] is False, audit)
    _assert(audit["target_gt_used_for_branch_alpha_capacity_fallback"] is False, audit)
    _assert(audit["heldout_test_gt_paths"] == [], audit)
    _assert(audit["selection_allowed_splits"] == ["train_fit", "train_policy_val"], audit)
    _assert(audit["threshold_source_split"] == "train_policy_val", audit)

    sources_by_id = manifest["sources"]
    _assert(sources_by_id, "manifest must contain hashed source records")
    for source_id, source in sources_by_id.items():
        _assert(source["id"] == source_id, source)
        _validate_source_record(source)

    decisions = certificate["decisions"]
    accepted = [item for item in decisions if item["decision"] == "accepted"]
    fallback = [item for item in decisions if item["decision"] == "fallback_parent"]
    _assert(len(accepted) == 1, decisions)
    _assert(len(fallback) == 1, decisions)

    for decision in decisions:
        _assert(decision["scene"] == manifest["scene"], decision)
        _assert(decision["selection_evidence_splits"] == ["train_policy_val"], decision)
        for source_id in decision["selection_source_ids"]:
            source = sources_by_id[source_id]
            _assert(source["split"] in audit["selection_allowed_splits"], decision)
            _assert(source["split"] != "target_test", decision)
        for output in decision["target_outputs"]:
            _assert(output["split"] == "target_test", output)
            _assert(output["gt_used"] is False, output)
            _assert(output["path"], output)
            _assert(output["sha256"], output)
        _validate_gate_block(decision, certificate["thresholds"])

    accept = accepted[0]
    _assert(accept["accepted"] is True, accept)
    _assert(accept["selected_output"] == "candidate", accept)
    _assert(accept["rejection_reasons"] == [], accept)
    _assert(accept["policy_val_gates"]["mse_direction"] == "improved", accept)
    _assert(accept["artifact_contract"]["parent_preserving_formula"] == "parent_rgb + confidence * residual_rgb", accept)
    _assert(accept["artifact_contract"]["parent_preserving_default"] is True, accept)
    _assert(accept["artifact_contract"]["confidence_range"] == [0.0, 1.0], accept)

    reject = fallback[0]
    _assert(reject["accepted"] is False, reject)
    _assert(reject["selected_output"] == "parent", reject)
    _assert(reject["fallback"]["kind"] == "exact_parent_noop", reject)
    _assert(reject["fallback"]["output_parent_render_sha256"] == reject["fallback"]["source_parent_render_sha256"], reject)
    _assert(reject["policy_val_gates"]["mse_direction"] == "regressed", reject)
    _assert(
        reject["policy_val_gates"]["positive_view_fraction"] < certificate["thresholds"]["min_positive_view_fraction"],
        reject,
    )
    _assert(reject["policy_val_gates"]["cvar20_mse_delta"] > certificate["thresholds"]["max_cvar20_mse_delta"], reject)
    _assert(reject["rejection_reasons"], reject)
    for reason in reject["rejection_reasons"]:
        _assert(isinstance(reason["code"], str) and reason["code"], reason)
        _assert(reason["source_split"] == "train_policy_val", reason)
        _assert("threshold" in reason and "observed" in reason, reason)


def _source_record(source_id: str, split: str, role: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source_id,
        "split": split,
        "role": role,
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "bytes": artifact["bytes"],
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_parent = _write_text(root / "train_fit" / "parent_000.txt", "parent train fit\n")
        train_gt = _write_text(root / "policy_val" / "gt_001.txt", "train policy-val gt\n")
        train_candidate = _write_text(root / "policy_val" / "candidate_001.txt", "candidate policy-val render\n")
        target_parent = _write_text(root / "target_test" / "parent_010.txt", "parent target render\n")
        target_candidate = _write_text(root / "target_test" / "candidate_010.txt", "candidate target render\n")
        target_fallback = _write_text(root / "target_test" / "fallback_011.txt", "parent target render\n")

        manifest = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "scene": "synthetic_vnext_schema_scene",
            "method": "vNext_certified_residual_surface_texture",
            "splits": {
                "fit": {"split": "train", "view_subset": "fit"},
                "policy_val": {"split": "train", "view_subset": "policy_val"},
                "target": {"split": "test", "view_subset": "heldout", "gt_allowed_for_selection": False},
            },
            "sources": {
                "fit_parent_000": _source_record("fit_parent_000", "train_fit", "parent_render", train_parent),
                "policy_val_gt_001": _source_record("policy_val_gt_001", "train_policy_val", "train_gt_rgb", train_gt),
                "policy_val_candidate_001": _source_record(
                    "policy_val_candidate_001", "train_policy_val", "candidate_render", train_candidate
                ),
                "target_parent_010": _source_record("target_parent_010", "target_test", "parent_render", target_parent),
            },
            "no_test_gt_audit": {
                "heldout_test_gt_used_for_selection": False,
                "target_gt_used_for_branch_alpha_capacity_fallback": False,
                "heldout_test_gt_paths": [],
                "selection_allowed_splits": ["train_fit", "train_policy_val"],
                "threshold_source_split": "train_policy_val",
            },
        }
        manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
        manifest_path = root / "manifest.json"
        _json_dump(manifest_path, manifest)

        certificate = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "thresholds": {
                "min_mean_psnr_gain": 0.0,
                "min_mean_ssim_gain": 0.0,
                "min_mean_lpips_gain": -0.000001,
                "min_positive_view_fraction": 0.75,
                "max_cvar20_mse_delta": 0.0,
                "min_support_count": 4,
                "max_residual_variance": 0.05,
            },
            "decisions": [
                {
                    "scene": manifest["scene"],
                    "candidate_id": "synthetic_accept_surface_texture",
                    "decision": "accepted",
                    "accepted": True,
                    "selected_output": "candidate",
                    "selection_evidence_splits": ["train_policy_val"],
                    "selection_source_ids": ["policy_val_gt_001", "policy_val_candidate_001"],
                    "thresholds": {"source_split": "train_policy_val"},
                    "policy_val_gates": {
                        "mean_psnr_gain": 0.42,
                        "mean_ssim_gain": 0.003,
                        "mean_lpips_gain": 0.012,
                        "mse_direction": "improved",
                        "positive_view_fraction": 1.0,
                        "cvar20_mse_delta": -0.0004,
                        "support_count": 12,
                        "residual_variance": 0.018,
                        "target_camera_support": "supported",
                        "parent_candidate_distance": 0.07,
                    },
                    "artifact_contract": {
                        "parent_preserving_formula": "parent_rgb + confidence * residual_rgb",
                        "parent_preserving_default": True,
                        "confidence_range": [0.0, 1.0],
                    },
                    "target_outputs": [
                        {
                            "split": "target_test",
                            "path": target_candidate["path"],
                            "sha256": target_candidate["sha256"],
                            "gt_used": False,
                        }
                    ],
                    "rejection_reasons": [],
                },
                {
                    "scene": manifest["scene"],
                    "candidate_id": "synthetic_reject_surface_texture",
                    "decision": "fallback_parent",
                    "accepted": False,
                    "selected_output": "parent",
                    "selection_evidence_splits": ["train_policy_val"],
                    "selection_source_ids": ["policy_val_gt_001", "policy_val_candidate_001"],
                    "thresholds": {"source_split": "train_policy_val"},
                    "policy_val_gates": {
                        "mean_psnr_gain": 0.0,
                        "mean_ssim_gain": 0.0,
                        "mean_lpips_gain": 0.0,
                        "mse_direction": "regressed",
                        "positive_view_fraction": 0.25,
                        "cvar20_mse_delta": 0.003,
                        "support_count": 2,
                        "residual_variance": 0.091,
                        "target_camera_support": "oot",
                        "parent_candidate_distance": 0.31,
                    },
                    "fallback": {
                        "kind": "exact_parent_noop",
                        "reason_code": "policy_val_tail_risk_failed",
                        "source_parent_render_sha256": target_parent["sha256"],
                        "output_parent_render_sha256": target_fallback["sha256"],
                    },
                    "target_outputs": [
                        {
                            "split": "target_test",
                            "path": target_fallback["path"],
                            "sha256": target_fallback["sha256"],
                            "gt_used": False,
                        }
                    ],
                    "rejection_reasons": [
                        {
                            "code": "policy_val_positive_view_fraction_failed",
                            "source_split": "train_policy_val",
                            "threshold": 0.75,
                            "observed": 0.25,
                        },
                        {
                            "code": "target_camera_support_oot",
                            "source_split": "train_policy_val",
                            "threshold": "supported",
                            "observed": "oot",
                        },
                    ],
                },
            ],
        }
        certificate_path = root / "certificate.json"
        _json_dump(certificate_path, certificate)

        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded_certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        validate_manifest_and_certificate(loaded_manifest, loaded_certificate)

    print("vNext no-test-GT manifest/certificate schema smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
