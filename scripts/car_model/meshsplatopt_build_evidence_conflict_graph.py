#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.evidence_conflict_graph import (  # noqa: E402
    ECGConfig,
    build_evidence_conflict_graph,
    load_correspondence_npz,
    write_ecg_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an Evidence Conflict Graph from SCE sparse correspondence diagnostics.")
    parser.add_argument("--correspondence_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--margin_abs", type=float, default=0.0)
    parser.add_argument("--margin_rel", type=float, default=0.0)
    args = parser.parse_args()
    graph = build_evidence_conflict_graph(
        load_correspondence_npz(args.correspondence_npz),
        cfg=ECGConfig(margin_abs=float(args.margin_abs), margin_rel=float(args.margin_rel)),
        source=args.source,
        split=args.split,
    )
    write_ecg_outputs(graph, args.output_dir)
    print(
        {
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "clusters": len(graph["cluster_summary"]),
            "top_cluster": graph["cluster_summary"][0] if graph["cluster_summary"] else None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

