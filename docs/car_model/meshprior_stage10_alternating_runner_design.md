# MeshPrior Stage 10 Design — Alternating Pipeline Runner

| Field | Value |
|---|---|
| Stage | M10 / alternating runner |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M9 scene gates and rollback |

## 1. Goal

M10 adds an orchestration runner that connects region mining, posterior inference, proposal generation, scene gates, accepted proposal export, and reporting.

The first runner is dry-run first. It must not modify scene geometry unless an explicit apply mode is added and enabled.

## 2. Pipeline Sequence

Default sequence:

1. region mining;
2. region posterior inference;
3. protect/prune proposal generation;
4. optional snap proposal generation;
5. optional fill proposal generation;
6. scene gate evaluation;
7. accepted proposal export;
8. report generation.

## 3. Optional Stages

Optional or resumable:

- skip region mining with `--skip_region_mining`;
- reuse `--regions_json`;
- reuse `--posterior_dir`;
- reuse `--proposals_json`;
- run gate/report only with `--eval_only`;
- include or exclude proposal types through `--proposal_types`.

## 4. Artifacts

The runner writes:

```text
run_config.json
regions.json
posterior/posterior_summary.json
proposals/proposals.json
scene_gate/gate_report.json
accepted_proposals.json
pipeline_report.md
```

Dry-run proposal meshes are stored as NPZ before/after pairs under `proposals/`.

## 5. Recovery and Scene Optimizer Integration

If scene training or optimizer integration exists, M10 only prepares accepted proposal artifacts for a later apply/recovery stage.

Recovery behavior:

- every gate evaluation writes rollback snapshots;
- failed substages write `pipeline_status.json`;
- accepted proposals are exported separately from all proposals;
- no scene model is modified in dry-run mode.

## 6. Resume Behavior

Resume flags let the runner skip expensive upstream work. When a supplied artifact is missing or invalid, the runner stops with a nonzero error and writes a failed status file.

## 7. Safety Rules

Safety flags:

- `--dry_run`;
- `--no_geometry_write`;
- `--max_regions`;
- `--max_proposals`;
- `--require_gate_pass`.

Default behavior:

- dry-run mode;
- no geometry write;
- synthetic fallback only when scene source is explicitly `synthetic`;
- stop safely on unknown proposal types or failed gate when required.

## 8. Stage Gate

M10 passes if:

- synthetic dry-run pipeline completes end-to-end;
- accepted and rejected proposal artifacts are written;
- no scene geometry is modified;
- report clearly states dry-run mode and gate results.
