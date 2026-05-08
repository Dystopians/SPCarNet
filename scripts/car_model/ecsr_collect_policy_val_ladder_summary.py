#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_audit(scene_payload: dict[str, Any]) -> dict[str, Any]:
    source_model = Path(scene_payload["source_model"])
    audit_path = source_model / "topology_audit.json"
    if not audit_path.is_file():
        return {}
    return _load_json(audit_path)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _scene_row(scene_dir: Path) -> dict[str, Any] | None:
    summary_path = scene_dir / "summary.json"
    if not summary_path.is_file():
        return None
    payload = _load_json(summary_path)
    selected = payload.get("selected")
    if selected is None:
        return {
            "scene": payload.get("scene", scene_dir.name),
            "selected": False,
            "ratio": 0.0,
            "model_path": "",
            "dPSNR": 0.0,
            "dSSIM": 0.0,
            "dLPIPS": 0.0,
            "additional_removed_fraction": 0.0,
            "source_removed_fraction": None,
            "total_removed_fraction": None,
            "post_triangles": None,
            "post_vertices": None,
        }
    audit = selected.get("audit", {})
    source = _source_audit(payload)
    source_removed = source.get("removed_fraction")
    total_removed = None
    if source and audit:
        clean_triangles = int(source.get("pre_triangles", 0))
        post_triangles = int(audit.get("post_triangles", 0))
        if clean_triangles > 0 and post_triangles > 0:
            total_removed = 1.0 - float(post_triangles / clean_triangles)
    delta = selected.get("delta", {})
    return {
        "scene": payload.get("scene", scene_dir.name),
        "selected": True,
        "ratio": float(selected.get("ratio", 0.0)),
        "model_path": str(selected.get("model_path", "")),
        "dPSNR": float(delta.get("dPSNR", 0.0)),
        "dSSIM": float(delta.get("dSSIM", 0.0)),
        "dLPIPS": float(delta.get("dLPIPS", 0.0)),
        "additional_removed_fraction": float(selected.get("additional_removed_fraction", 0.0)),
        "source_removed_fraction": float(source_removed) if source_removed is not None else None,
        "total_removed_fraction": total_removed,
        "post_triangles": int(audit.get("post_triangles", 0)) if audit else None,
        "post_vertices": int(audit.get("post_vertices", 0)) if audit else None,
    }


def _mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate ECSR policy-val compaction ladder scene summaries.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", default="bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    rows = []
    missing = []
    for scene in scenes:
        row = _scene_row(root / scene)
        if row is None:
            missing.append(scene)
        else:
            rows.append(row)

    aggregate = {
        "root": str(root),
        "rows": rows,
        "missing": missing,
        "complete_count": len(rows),
        "scene_count": len(scenes),
        "accepted_count": sum(1 for row in rows if row["selected"]),
        "mean_selected_ratio": _mean([row["ratio"] for row in rows if row["selected"]]),
        "mean_additional_removed_fraction": _mean([row["additional_removed_fraction"] for row in rows if row["selected"]]),
        "mean_source_removed_fraction": _mean([row["source_removed_fraction"] for row in rows if row["selected"]]),
        "mean_total_removed_fraction": _mean([row["total_removed_fraction"] for row in rows if row["selected"]]),
    }
    output_json = Path(args.output_json) if args.output_json else root / "aggregate_policy_val_ladder.json"
    output_md = Path(args.output_md) if args.output_md else root / "aggregate_policy_val_ladder.md"
    output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Aggregate ECSR Policy-Val Compaction Ladder",
        "",
        f"- root: `{root}`",
        f"- complete scenes: `{len(rows)}` / `{len(scenes)}`",
        f"- accepted scenes: `{aggregate['accepted_count']}` / `{len(rows)}`",
        f"- mean selected ratio: `{_fmt(aggregate['mean_selected_ratio'] or 0.0)}`",
        f"- mean additional removed fraction: `{_fmt(aggregate['mean_additional_removed_fraction'] or 0.0)}`",
        f"- mean source removed fraction: `{_fmt(aggregate['mean_source_removed_fraction'] or 0.0)}`",
        f"- mean total removed fraction: `{_fmt(aggregate['mean_total_removed_fraction'] or 0.0)}`",
        "",
        "| Scene | Selected | Ratio | dPSNR | dSSIM | dLPIPS | Add. Removed | Source Removed | Total Removed | Post Tris | Model |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scene"],
                    str(row["selected"]),
                    _fmt(row["ratio"]),
                    f"{row['dPSNR']:+.5f}",
                    f"{row['dSSIM']:+.6f}",
                    f"{row['dLPIPS']:+.6f}",
                    _fmt(row["additional_removed_fraction"]),
                    _fmt(row["source_removed_fraction"] or 0.0),
                    _fmt(row["total_removed_fraction"] or 0.0),
                    str(row["post_triangles"] or ""),
                    row["model_path"],
                ]
            )
            + " |"
        )
    if missing:
        lines.extend(["", "## Missing", ""])
        lines.extend(f"- `{scene}`" for scene in missing)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
