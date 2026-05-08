# ECSR Phase-E Policy-Validated ELA Probes

Date: 2026-05-08

This note records the policy-level visual-repair probes run after the Phase-D
representation-level recovery failures. These probes are not promoted as the
final ECSR method. Their purpose is to test whether the current visual bottleneck
comes from an insufficiently intelligent train-only ELA policy.

## Code Changes

- `utils/evidence_lumigraph_adapter.py`
  - adds `confidence_magnitude_edge` benefit-gate features;
  - keeps the legacy `confidence_magnitude` gate backward-compatible;
  - adds optional calibration target frames so policy selection can be split
    into train-fit and train-policy-val views.
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - adds `--benefit_feature_mode auto`;
  - adds `--policy_holdout_fraction`;
  - reports selected benefit feature mode, fit views, and policy-val views.
- `scripts/car_model/evaluate_render_split_metrics.py`
  - adds `--merge_model_results` so a single newly rendered method can be
    evaluated and merged into `results.json` without recomputing all methods.

All modified Python files pass `py_compile`, and the ELA smoke test passes.

## Probe 1: Texture-Aware Benefit Gate

Fixed policy:

- scenes: `flowers`, `garden`, `bicycle`, `treehill`;
- base: compact checkpoint render `ours_26000`;
- candidate method: `ours_26000_sor_adaptive_geo_compact_ela_tebg_edge`;
- train-only auto-policy over residual mode, `k in {4,8}`, depth rel in
  `{0.06,0.12}`, residual clip in `{0.20,0.25}`, direction weight in
  `{0.20,0.35}`;
- benefit feature: `confidence_magnitude_edge`;
- W&B group: `ecsr_tebg_edge_outdoor_v1`.

| scene | W&B | current PSNR | edge PSNR | dPSNR | current SSIM | edge SSIM | dSSIM | current LPIPS | edge LPIPS | dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | `97a7ujbq` | 20.1828 | 20.1112 | -0.0716 | 0.5473 | 0.5411 | -0.0062 | 0.3510 | 0.3577 | +0.0067 |
| garden | `nu05al0u` | 26.0348 | 25.9276 | -0.1072 | 0.8171 | 0.8128 | -0.0043 | 0.1523 | 0.1572 | +0.0048 |
| bicycle | `9ozp5lhn` | 23.9127 | 23.8899 | -0.0228 | 0.6937 | 0.6913 | -0.0025 | 0.2803 | 0.2847 | +0.0045 |
| treehill | `y1c42c9l` | 21.1984 | 21.1740 | -0.0243 | 0.5882 | 0.5851 | -0.0031 | 0.3581 | 0.3651 | +0.0069 |

Decision: reject. Adding an edge-strength axis over-constrains the benefit gate
and consistently loses against the current Compact-ELA/SOR policy.

## Probe 2: Train-Fit / Train-Policy-Val Holdout

Fixed policy:

- scenes: `flowers`, `garden`;
- candidate method: `ours_26000_sor_adaptive_geo_compact_ela_holdout_auto`;
- `--policy_holdout_fraction 0.2`;
- `--benefit_feature_mode auto`, selecting between `confidence_magnitude` and
  `confidence_magnitude_edge` using policy-val views only;
- W&B group: `ecsr_ela_holdout_auto_v1`.

| scene | W&B | selected feature | current PSNR | holdout PSNR | dPSNR | current SSIM | holdout SSIM | dSSIM | current LPIPS | holdout LPIPS | dLPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | `qfr8g9vo` | `confidence_magnitude` | 20.1828 | 20.1869 | +0.0042 | 0.5473 | 0.5465 | -0.0008 | 0.3510 | 0.3509 | -0.0001 |
| garden | `gkyi0cl4` | `confidence_magnitude_edge` | 26.0348 | 25.9518 | -0.0831 | 0.8171 | 0.8138 | -0.0033 | 0.1523 | 0.1560 | +0.0037 |

Decision: reject as a promoted method. The holdout split is a useful guardrail,
but it does not yet predict held-out-test dominance reliably enough. `flowers`
has a tiny PSNR/LPIPS gain but loses SSIM; `garden` regresses on all RGB axes.

## Probe 3: ELA Alpha Ceiling

Fixed policy:

- scenes: `flowers`, `garden`;
- candidate method: `ours_26000_sor_adaptive_geo_compact_ela_alpha150`;
- same current auto-policy space, but alpha grid extended to
  `0,0.125,0.25,0.5,0.75,1.0,1.25,1.5`;
- W&B group: `ecsr_ela_alpha150_probe_v1`.

| scene | W&B | selected alpha | result |
|---|---|---:|---|
| flowers | `it2d2xvg` | 1.0 | identical to current Compact-ELA/SOR within metric precision |
| garden | `ahidixp2` | 1.0 | identical to current Compact-ELA/SOR within metric precision |

Decision: reject as an improvement path. The current method is not simply capped
by the alpha upper bound; train-only calibration still selects `alpha=1.0` when
larger strengths are available.

## Phase-E Conclusion

The current Compact-ELA/SOR visual adapter remains stronger than the newly
tested policy variants. The negative evidence is useful:

1. Texture-aware edge gating is not automatically better; it removes too much
   residual support in high-frequency outdoor scenes.
2. A train-fit / policy-val split is the right protocol direction, but the
   current policy-val objective is still too weak to guarantee held-out-test
   dominance.
3. More ELA strength is not the missing factor.

The next method should therefore not be another ELA-policy tweak. The next
credible path is representation-side: improve certificate-carrying contraction
and surface-attached recovery, with cached policy-val evaluation so candidates
can be rejected before expensive full held-out tests.
