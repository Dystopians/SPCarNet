#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.compact_selector import (  # noqa: E402
    CompactionSignals,
    select_faces,
    write_selector_outputs,
)


def synthetic_signals() -> CompactionSignals:
    vertices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [1.02, 0, 0],
            [1.02, 0.05, 0],
            [1.00, 0.05, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [0.5, 2.01, 0],
            [4, 4, 3],
            [4.4, 4, 3],
            [4, 4.4, 3],
            [0, -2, 0],
            [4, -2, 0],
            [4, -1, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],  # supported
            [0, 2, 3],  # supported
            [1, 4, 5],  # redundant small
            [1, 5, 6],  # redundant small
            [7, 8, 9],  # hole rim / boundary
            [7, 9, 10],  # hole rim / boundary
            [11, 12, 13],  # high debt repair region
            [14, 15, 16],  # floater
            [17, 18, 19],  # large ground patch
            [17, 19, 20],  # large ground patch
        ],
        dtype=np.int64,
    )
    labels = np.asarray(
        [
            "supported",
            "supported",
            "redundant_small",
            "redundant_small",
            "hole_rim",
            "hole_rim",
            "repair_debt",
            "floater",
            "large_ground",
            "large_ground",
        ]
    )
    positive = np.asarray([0.95, 0.95, 0.15, 0.15, 0.90, 0.90, 0.80, 0.02, 0.80, 0.80])
    negative = np.asarray([0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.95, 0.00, 0.00])
    debt = np.asarray([0.05, 0.05, 0.05, 0.05, 0.85, 0.85, 1.00, 0.00, 0.75, 0.75])
    uncertainty = np.asarray([0.05, 0.05, 0.20, 0.20, 0.50, 0.50, 0.30, 0.80, 0.20, 0.20])
    protected = np.isin(labels, ["hole_rim", "repair_debt", "large_ground"])
    return CompactionSignals(
        vertices=vertices,
        faces=faces,
        positive_surface_evidence=positive,
        negative_free_space=negative,
        explanation_debt=debt,
        uncertainty=uncertainty,
        protected_faces=protected,
        labels=labels,
    )


def main() -> int:
    out_root = ROOT / "outputs/carnet/meshsplatopt/final_stageF4_selector_smoke"
    signals = synthetic_signals()
    labels = signals.labels
    area_selected, area_table = select_faces(signals, "area_smallest", 0.3, seed=7)
    csef_selected, csef_table = select_faces(signals, "csef_low_evidence_boundary_protected", 0.3, seed=7)
    pareto_selected, pareto_table = select_faces(signals, "pareto_area_csef", 0.3, seed=7)
    random_selected, random_table = select_faces(signals, "random_same_count", 0.3, seed=7)

    write_selector_outputs(out_root / "area_smallest", area_selected, area_table, "area_smallest", 0.3, labels)
    write_selector_outputs(
        out_root / "csef_low_evidence_boundary_protected",
        csef_selected,
        csef_table,
        "csef_low_evidence_boundary_protected",
        0.3,
        labels,
    )
    write_selector_outputs(out_root / "pareto_area_csef", pareto_selected, pareto_table, "pareto_area_csef", 0.3, labels)
    write_selector_outputs(out_root / "random_same_count", random_selected, random_table, "random_same_count", 0.3, labels)

    csef_labels = set(labels[csef_selected].tolist())
    assert len(csef_selected) == len(area_selected) == len(random_selected)
    assert set(csef_selected.tolist()) != set(area_selected.tolist())
    assert "redundant_small" in csef_labels
    assert "floater" in csef_labels
    assert "hole_rim" not in csef_labels
    assert "repair_debt" not in csef_labels
    assert "large_ground" not in csef_labels
    assert len(pareto_selected) == len(random_selected)
    print(f"F4 selector smoke PASS: area={area_selected.tolist()} csef={csef_selected.tolist()} random={random_selected.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
