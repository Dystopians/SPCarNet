# 6-23 Teacher-Bake v34 And Region Atlas v35 Log

Date: 2026-06-23

Status: `NOT PROMOTED`

This log records two follow-up attempts after the v33 higher-capacity region
basis bottleneck:

1. v34 Bonsai long teacher-bake recovery with W&B online logging;
2. v35 teacher-region surface residual texture/atlas adapter.

Both are real pipeline changes or evaluations, but neither is strong enough to
replace the Phase-J guarded adaptive ELA endpoint.

## Baselines For Bonsai

| method | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---|
| selected clean MeshSplatting `ours_26000` | 28.895233 | 0.896400 | 0.259493 | local fair clean baseline |
| compact parent | 28.864340 | 0.896012 | 0.259340 | Phase-F compact base |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | current paper-facing endpoint |

## v34: Long Teacher-Bake Recovery

### Command And Artifacts

Root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v34_teacher_bake_long_bonsai_20260623
```

Main log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v34_teacher_bake_long_bonsai_20260623/logs/run_bonsai_phaseg_v34_long_gpu5.log
```

Summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v34_teacher_bake_long_bonsai_20260623/phaseg_teacher_bake_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v34_teacher_bake_long_bonsai_20260623/phaseg_teacher_bake_summary.json
```

W&B:

```text
teacher render run: zrrst5kk
training run: na4d59me
project: mesh-splatting-ecsr
group: phaseg_v34_teacher_bake_long_bonsai
```

The run used GPU5 with W&B online and trained from `26000` to `27000`, saving
milestones at `26200`, `26500`, and `27000`.

### Critical Teacher Audit

The teacher policy loaded from the Phase-J report resolved to:

```json
{
  "source": "phasef_report",
  "alpha": 0.0,
  "mode": "residual",
  "k": 4
}
```

The train teacher ELA report also records:

```text
alpha = 0.0
alpha_source = cli
calibration reason = fixed_alpha_calibration_skipped
```

This means the supposed teacher renders were effectively no-op / compact-parent
teacher renders, not a useful Phase-J residual target. The result should
therefore be read as a negative control for the teacher-bake runner, not as a
valid test of strong teacher residual baking.

### Metrics

| method | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR Phase-J | dSSIM Phase-J | dLPIPS Phase-J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v34 `ours_27000` | 28.256157 | 0.874371 | 0.299884 | -0.639076 | -0.022029 | +0.040392 | -3.605848 | -0.055909 | +0.127329 |

Topology stayed frozen:

```text
triangles: 9,555,533 -> 9,555,533
vertices: 3,295,557 -> 3,295,557
```

Conclusion: `NOT PROMOTED`. The run is worse than selected clean, compact
parent, and Phase-J. The immediate cause is likely the invalid no-op teacher
source plus continued appearance training under strong rollback constraints.

## v35: Teacher-Region Surface Residual Texture Atlas

### Code Change

Implemented:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

The script fits a per-face barycentric residual texture atlas from train-only
teacher surface evidence, selects alpha on train-only policy-val views, and
applies the atlas to target evidence NPZs. It copies the source model shell and
writes final renders under:

```text
<output_model>/test/<method_name>/renders
<output_model>/test/<method_name>/gt
```

This turn also repaired a robustness bug: target NPZs without `barycentric`
now safely get zero delta / unchanged base render instead of aborting the
entire held-out evaluation.

### Target Evidence

Built Bonsai full-resolution test target surface evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_target_surface_evidence_images2/bonsai/views
```

Evidence status:

```text
37 / 37 target NPZs written
15 / 37 target NPZs lack barycentric and therefore receive no atlas delta
```

This is already a bottleneck: the atlas cannot affect those target views.

### v35 Attempts

| variant | gate status | key result |
|---|---|---|
| v1 default stride4 | rejected | policy-val relative gain `0.424816`, but only `933` policy-val samples, below `1024` threshold |
| v2 coverage-expanded | rejected | lowering train sampling thresholds did not produce an accepted policy |
| v3 stride3 | accepted | policy-val samples `1090`, relative gain `0.521415`, selected alpha `0.75` |

v3 artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_teacher_region_texture_adapter_v3_stride3
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/logs/apply_bonsai_teacher_region_texture_adapter_v3_stride3_retry.log
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/logs/metrics_bonsai_teacher_region_texture_adapter_v3_stride3_gpu4.log
```

v3 audit:

```text
accepted = true
atlas faces = 65
fit samples = 1436
policy-val samples = 1090
selected alpha = 0.75
policy-val relative gain = 0.521415
target written views = 37
target changed fraction = 0.000006657
```

The extremely low changed fraction is the decisive limitation. The train-only
gate sees a real residual fit on a small support, but the fitted support covers
almost none of the held-out image area.

### v35 Metrics

| method | PSNR | SSIM | LPIPS | dPSNR compact | dSSIM compact | dLPIPS compact | dPSNR clean | dSSIM clean | dLPIPS clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact parent | 28.864340 | 0.896012 | 0.259340 | 0.000000 | 0.000000 | 0.000000 | -0.030893 | -0.000388 | -0.000153 |
| v35 stride3 atlas | 28.864641 | 0.896011 | 0.259334 | +0.000301 | -0.000001 | -0.000005 | -0.030592 | -0.000389 | -0.000158 |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 | +2.966772 | +0.033879 | -0.086937 |

Conclusion: `NOT PROMOTED`. v35 is a real accepted train-only atlas candidate,
but its held-out effect is too sparse to matter. It is not competitive with
clean MeshSplatting and is nowhere near Phase-J.

## Lessons

1. v34 confirms that teacher-bake experiments must audit the teacher source
   before training. A Phase-J report row with `alpha=0.0` silently turns the
   teacher into a no-op target.
2. v35 confirms that a surface atlas can be wired into the pipeline and can
   pass train-only residual fitting, but the current carrier support is far too
   small on held-out views.
3. The next credible representation-level experiment should not continue
   tweaking atlas alpha. It should first build matched-resolution train evidence
   and ensure target barycentric coverage, then fit a larger support carrier.

## Next Step

Do not promote v34 or v35. The next experiment should be:

```text
matched-res train teacher surface evidence at images_2
  -> region carrier discovery at images_2
  -> surface residual atlas with higher held-out target coverage
  -> metrics against compact, selected clean, and Phase-J
```

