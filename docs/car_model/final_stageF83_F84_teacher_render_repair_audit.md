# Final Stage F83 / F84 Teacher-Render Repair Audit

Decision: `REJECT_NOT_PARENT_PARETO`.

F82 remains the accepted fixed-policy v5 checkpoint. F83 and F84 tested whether a masked clean-render teacher could repair the few held-out views where F82 still trails clean in per-view PSNR.

## What Changed

- Added strict-recovery support for optional teacher-render loss.
- Generated clean-long train-view renders for bonsai and courtyard.
- Ran online W&B continuation from F82 checkpoints with topology frozen:
  - F83 bonsai: `n6nx5w2z`, `26000 -> 28000`, teacher lambda `0.005`
  - F83 courtyard: `0p95l810`, `26000 -> 28000`, teacher lambda `0.005`
  - F84 bonsai: `acraejgs`, `26000 -> 27000`, teacher lambda `0.001`
- Added a parent-Pareto gate script so continuation branches cannot replace F82 unless they improve or tie parent render, geometry, and per-view PSNR stability.

## Results Against Clean

F83 and F84 still beat clean-long on the tested rows, but that is not enough. The relevant acceptance test is whether they strictly improve F82.

| stage | scene | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | clean status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F83 | bonsai | 11.075393 | 0.232822 | 0.579448 | 0.180689 | 1.797371 | 42.089326 | `PASS_ALL_METRIC_CLEAN_WIN` |
| F83 | courtyard | 12.300953 | 0.317779 | 0.565530 | 0.305840 | 3.406779 | 40.020689 | `PASS_ALL_METRIC_CLEAN_WIN` |
| F84 | bonsai | 11.073513 | 0.236555 | 0.576637 | 0.180572 | 1.795967 | 42.104761 | `PASS_ALL_METRIC_CLEAN_WIN` |

## Parent-Pareto Gate Against F82

| candidate | scene | render vs F82 | geometry vs F82 | per-view PSNR vs F82 | decision |
|---|---|---|---|---|---|
| F83 | bonsai | PSNR +0.0062, but SSIM -0.0083 and LPIPS +0.0065 | improves AbsRel / Depth / Normal | 14 / 37 views worse; min -0.0489 | `REJECT_NOT_PARENT_PARETO` |
| F83 | courtyard | PSNR +0.1023, SSIM +0.0091, LPIPS -0.0012 | Normal improves, but AbsRel +0.0040 and Depth +0.0669 worsen | 0 / 5 views worse; min +0.0159 | `REJECT_NOT_PARENT_PARETO` |
| F84 | bonsai | PSNR +0.0043, but SSIM -0.0046 and LPIPS +0.0037 | improves AbsRel / Depth / Normal | 11 / 37 views worse; min -0.0247 | `REJECT_NOT_PARENT_PARETO` |

## Interpretation

Teacher-render continuation is useful as a diagnostic and produces a real courtyard visual improvement, but it is not a clean replacement for F82 because it introduces metric tradeoffs. The accepted policy remains F82 fixed adaptive policy v5. The next repair should not use clean-render distillation as a blunt post-hoc tool; it should target the underlying uncertainty signal before recovery or use a stricter multi-objective continuation gate.
