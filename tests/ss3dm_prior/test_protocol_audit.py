from __future__ import annotations

import json
from pathlib import Path

import torch

from ss3dm_prior.metrics import (
    retrieval_top1_cross_sequence,
    retrieval_top1_nonself,
    retrieval_top5_nonself,
)
from ss3dm_prior.tools.audit_run_protocol import audit_run_protocol
from ss3dm_prior.utils.io import dump_yaml


def test_audit_run_protocol_detects_debug_leakage(tmp_path: Path) -> None:
    run_dir = tmp_path / "train_output"
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    split_path = tmp_path / "split.yaml"
    dump_yaml(
        split_path,
        {
            "train_towns": ["Town01", "Town02"],
            "val_towns": ["Town07"],
            "test_towns": ["Town10"],
        },
    )
    dump_yaml(
        config_dir / "resolved_train_config.yaml",
        {
            "debug_use_all_patches_for_train_val": True,
            "allow_debug_split_override": True,
            "allow_split_fallback": True,
        },
    )
    (config_dir / "resolved_run_metadata.json").write_text(
        json.dumps({"split_config": str(split_path)}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = audit_run_protocol(run_dir)

    assert result["protocol_valid"] is False
    assert any("debug leakage detected" in warning for warning in result["protocol_warnings"])
    assert any("allow_debug_split_override=true" in warning for warning in result["protocol_warnings"])
    assert result["protocol_summary"]["expected_train_towns"] == ["Town01", "Town02"]
    assert result["protocol_summary"]["expected_val_towns"] == ["Town07"]
    assert result["protocol_summary"]["expected_test_towns"] == ["Town10"]


def test_filtered_retrieval_metrics_match_clean_neighbor_reference() -> None:
    clean = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.7, -0.7],
        ],
        dtype=torch.float32,
    )
    corrupted = torch.tensor(
        [
            [0.78, 0.62],
            [0.98, 0.02],
            [0.25, 0.97],
            [0.72, -0.68],
        ],
        dtype=torch.float32,
    )
    patch_ids = ["p0", "p1", "p2", "p3"]
    sequence_ids = ["seq_a", "seq_a", "seq_b", "seq_c"]

    assert retrieval_top1_nonself(
        corrupted,
        clean,
        query_patch_ids=patch_ids,
        target_patch_ids=patch_ids,
        query_sequence_ids=sequence_ids,
        target_sequence_ids=sequence_ids,
    ) == 1.0
    assert retrieval_top5_nonself(
        corrupted,
        clean,
        query_patch_ids=patch_ids,
        target_patch_ids=patch_ids,
        query_sequence_ids=sequence_ids,
        target_sequence_ids=sequence_ids,
    ) == 1.0
    assert retrieval_top1_cross_sequence(
        corrupted,
        clean,
        query_patch_ids=patch_ids,
        target_patch_ids=patch_ids,
        query_sequence_ids=sequence_ids,
        target_sequence_ids=sequence_ids,
    ) == 0.5
