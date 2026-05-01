# MeshPrior Stage 12 Design — Prior Calibration for Proposal Reliability

| Field | Value |
|---|---|
| Stage | M12 / prior calibration |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M11 scene experiment |

## 1. Observed Failure

Evidence from prior stages:

- M7 showed that snap with `max_disp=0.02` improved synthetic surface distance but reduced valid-surface protect recall from `0.9167` to `0.8333`.
- Tightening snap to `0.005` preserved valid-surface protect recall while still improving surface distance.
- M11 only validated dry-run topology behavior; render/COLMAP improvements remain unproven.

Primary failure class:

```text
score thresholds / movement thresholds too aggressive
```

Secondary risks:

- free-space calibration is still synthetic-only;
- posterior uncertainty is not scene-calibrated;
- shape-field gradients are usable for small corrections but unsafe for larger snap steps.

Not selected for M12:

- SDF/UDF training;
- retrieval-deformation prior;
- symmetry prior;
- full posterior uncertainty training.

Reason: current evidence supports a post-hoc proposal calibration first, not a new model training run.

## 2. Chosen Upgrade

M12 implements a lightweight surface-support calibration profile:

```text
surface_support_v1
```

It maps prior-derived proposal actions to conservative scene-safe parameters:

- snap max displacement: `0.005` canonical units;
- maximum allowed visible protect recall drop: `0.05`;
- require non-increasing free-space violation;
- require positive surface-distance improvement for snap.

## 3. Metrics

Primary proposal metrics:

- valid surface protect recall;
- snap surface-distance delta;
- snap max displacement;
- free-space violation delta;
- scene gate acceptance rate.

Diagnostic metrics:

- triangle count delta;
- boundary edge delta;
- component/floater count delta.

## 4. Synthetic Benchmark

Benchmark case:

```text
vertex_noise synthetic box
```

Compare:

- uncalibrated snap profile: `max_disp=0.02`;
- calibrated snap profile: `max_disp=0.005`.

Expected result:

- calibrated profile keeps valid-surface protect recall at or above baseline;
- calibrated profile still improves surface distance;
- free-space violation remains unchanged.

## 5. Pipeline Integration

M10 pipeline receives:

```text
--calibration_profile none|surface_support_v1
```

When `surface_support_v1` is enabled, snap proposals use the calibrated snap displacement.

## 6. Stage Gate

M12 passes if:

- calibration improves at least one proposal-relevant metric;
- valid-surface protect recall is not harmed versus damaged baseline;
- free-space violation does not increase;
- smoke test passes;
- pipeline accepts the calibration flag.
