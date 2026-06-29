# v169 Teacher Residual Projection and True Low-Rank Texture Progress Draft

Date: 2026-06-28

Status: progress draft only. Do not claim v169 completion or paper readiness from this file.

## Prompt Objective

v169 asks a narrow question: can the Phase-J teacher-parent residual be baked into a MeshSplatting-compatible surface representation that beats the Phase-J flowers reference on all axes without target/test RGB GT leakage?

Hard flowers gate from the v169 prompt:

- Phase-J flowers reference: PSNR `20.304358`, SSIM `0.557770`, LPIPS `0.329222`.
- A candidate must have higher PSNR, higher SSIM, and lower LPIPS.
- PSNR-only improvement is a failure; no full9 run should start before flowers exact clears all three metrics.

The current work is still diagnostic/progress state. The completed clean K4 exact
runs below do not clear the Phase-J flowers gate.

## Storage Preflight

Latest preflight snapshot:

- `/data`: `28T` size, `27T` used, `7.5M` available, `100%` used.
- `/dev/shm`: `252G` size, `250G` used, `2.5G` available, `100%` used.
- `/tmp` on `/`: `14T` size, `7.1T` used, `6.1T` available, `54%` used.
- user quota on `/dev/nvme0n1p4`: `102028M` used against `100G` limit.

Relevant artifact footprint:

- exact run root: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact`, about `4.1G`.
- v169 diagnostic reports: `/dev/shm/peilincai_spcarnet_v169_diagnostics`, about `568K`.
- shared vNext full9 inputs: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626`, about `20G`.

Implication: do not launch duplicate exact runs. Keep using `auto_link`/low-copy modes, and free or move artifacts before any larger run.

## Teacher-Signal Diagnostics Paths

Primary diagnostic outputs:

- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_teacher_signal_stride8_all.md`
- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_teacher_signal_stride8_all.json`
- smoke subset: `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_teacher_signal_stride16_max8.md`
- teacher evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- teacher evidence report: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence/teacher_surface_evidence_report.md`
- top supports CSV: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence/top_residual_supports.csv`

Key stride8/all findings:

- selected files: `46 / 46`, sample stride `8`.
- teacher-parent mean L1: `0.015249`; fit-signal mean L1: `0.008571`.
- RGB-domain clip pixel fraction: `0.000456`.
- raw-to-fit changed pixel fraction: `0.455082`, so masking/clipping removes a large fraction of raw teacher signal.
- valid face coverage: `0.999298`; active fit-signal coverage: `0.340522`.
- unique observed faces: `648825`; unique active fit-signal faces: `287955`.
- weighted sign consistency: `0.758650`.
- face+barycentric-bin8 same-cache projection proxy: projected signal L2 fraction `0.945259`, projected PSNR proxy `21.068553`, proxy margin vs Phase-J flowers PSNR `+0.764195`.

These are useful sanity checks, but the projection proxy is same-cache and PSNR-only; it is not held-out SSIM/LPIPS proof.

## Base Carrier Upper-Bound Results

Primary report:

- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_upper_bound_lpips.md`
- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_upper_bound_lpips.json`

Setup:

- fit evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json`
- candidate faces: `342`; view count: `46`.
- target/test GT usage: `none`; test usage: `none`.

Best all-axis policy-val row:

- texture size `16`, alpha `0.03125`.
- relative gain `+0.01808713`.
- PSNR gain `+0.07927048`.
- SSIM gain `+0.00000033`.
- LPIPS gain `+0.00000187`.
- relative positive-view fraction `0.916667`; PSNR positive-view fraction `0.916667`.

Caveat: the all-axis mean is positive, but tail behavior is weak. The selected row has SSIM positive-view fraction `0.5`, LPIPS positive-view fraction `0.5`, SSIM min-view gain `-0.00000620`, LPIPS min-view gain `-0.00001054`, and image-L1 mean gain `-0.00000007`.

## True Low-Rank K4 Policy-Val Results

Primary report:

- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_true_lowrank_k4_lpips.md`
- `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_true_lowrank_k4_lpips.json`

Setup:

- `teacher_distilled_basis_mode=low_rank_view_texture_k4`.
- `teacher_distilled_basis_apply_mode=blend`.
- `teacher_distilled_basis_blend=0.5`.
- adaptive low-support teacher basis enabled; min face samples `128`.
- texture size `16`; candidate faces `342`; view count `46`.
- target/test GT usage: `none`; test usage: `none`.

Best all-axis policy-val row:

- alpha `0.0625`.
- relative gain `+0.03954590`.
- PSNR gain `+0.17523385`.
- SSIM gain `+0.00000170`.
- LPIPS gain `+0.00000271`.
- image-L1 gain `+0.00000038`.
- relative and PSNR positive-view fraction `1.0`.
- relative min-view gain `+0.00277367`; relative CVaR20 gain `+0.01132603`.

Caveat: the mean row is better than base-carrier policy-val, but SSIM/LPIPS tails are still fragile: SSIM positive-view fraction `0.5`, LPIPS positive-view fraction `0.5`, SSIM min-view gain `-0.00000530`, LPIPS min-view gain `-0.00002567`, LPIPS CVaR20 gain `-0.00002247`.

## Clean True-LowRank K4 Exact Results

Two cleaner flowers exact runs were completed after the first draft. Both reused
the same train-fit Phase-J teacher evidence and stripped target evidence, and
both disabled the previously noisy policy machinery that was not part of the
true low-rank representation test.

Common setup:

- teacher evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- no-GT target evidence: `/dev/shm/peilincai_spcarnet_20260629_v169_true_lowrank_k4_reuse_exact/flowers/target_evidence_no_gt`
- `teacher_distilled_basis_mode=low_rank_view_texture_k4`
- texture size `16`, support expansion `none`, multiscale prior `none`, view-conditioned basis `none`
- alpha grid `0,0.0625`
- target/test RGB GT was stripped before apply.

Strict policy run:

- root: `/dev/shm/peilincai_spcarnet_20260629_v169_true_lowrank_k4_clean_strict_exact/flowers`
- metrics: PSNR `19.832010`, SSIM `0.505779`, LPIPS `0.405904`
- gate delta vs Phase-J: PSNR `-0.472348`, SSIM `-0.051991`, LPIPS worse by `+0.076682`
- selected alpha `0.0`, `effective_policy=fallback_noop`
- reject reason: L1 min-view gain was below `-2e-6`, and effective SSIM gain was below `1e-6`
- target changed fraction: `0.0`

Relaxed diagnostic run:

- root: `/dev/shm/peilincai_spcarnet_20260629_v169_true_lowrank_k4_clean_relaxed_exact/flowers`
- metrics: PSNR `19.832146`, SSIM `0.505778`, LPIPS `0.405913`
- gate delta vs Phase-J: PSNR `-0.472212`, SSIM `-0.051992`, LPIPS worse by `+0.076691`
- selected alpha `0.0625`, `effective_policy=accepted_atlas`
- policy-val selected row: relative gain `+0.03954590`, SSIM gain `+0.00000056`, image-L1 gain `+0.00000035`, LPIPS gain `+0.00000271`
- target changed pixels: `40513 / 37100800`
- target changed fraction: `0.00109197`; PNG-quantized changed fraction: `0.00050576`

Verdict: the K4 low-rank texture is a real representation change and can be made
slightly positive on train-policy-val, but it transfers almost no visible target
signal and fails all three Phase-J flowers gates. The bottleneck is no longer
"missing exact evidence"; it is inadequate target-visible teacher residual
coverage and too weak SSIM/LPIPS transfer.

## Rich Low-Rank K4 Feature Upgrade

A follow-up representation upgrade was implemented after the K4 exact failure:

- mode: `low_rank_view_texture_rich_k4`
- files:
  - `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
  - `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
  - `scripts/car_model/analyze_v169_policy_val_upper_bound.py`
- motivation: the original K4 mixture-weight feature vector was only
  `[1, camera_x, camera_y, camera_z, normal_dot_camera]`. The rich version keeps
  the same rank-4 surface factorization but predicts mixture weights from
  camera direction, normal, normal-dot-camera, UV polynomial terms, parent RGB,
  inverse depth, and alpha.
- leakage contract: fitting uses train-fit teacher residuals; certification uses
  train-policy-val GT; target/test apply uses prestripped target evidence with no
  RGB GT or residual keys.

Rich policy-val upper-bound:

- report: `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_rich_lowrank_k4_lpips.md`
- JSON: `/dev/shm/peilincai_spcarnet_v169_diagnostics/flowers_policy_val_rich_lowrank_k4_lpips.json`
- texture size `16`, alpha `0.0625`
- relative gain `+0.03797914`
- PSNR gain `+0.16815513`
- SSIM gain `+0.00000097`
- image-L1 gain `+0.00000041`
- LPIPS gain `+0.00000118`
- SSIM positive-view fraction `0.5`; LPIPS positive-view fraction `0.5`
- SSIM min-view gain `-0.00000727`
- LPIPS min-view gain `-0.00003444`
- LPIPS CVaR20 gain `-0.00002666`

Coverage comparison against original K4:

| variant | feature dim | supported faces | supported bins | supported bin fraction | mean retained energy |
|---|---:|---:|---:|---:|---:|
| K4 original | 5 | 224 | 7568 | 0.131975 | 0.997015 |
| rich-K4 | 18 | 255 | 10633 | 0.162883 | 0.872292 |

Interpretation: rich-K4 increases face/bin support, but the higher-dimensional
feature field has lower retained low-rank energy and worse perceptual tail risk.
It improves mean policy-val PSNR/SSIM/L1/LPIPS slightly, but is weaker than the
original K4 on relative gain, PSNR gain, LPIPS gain, and LPIPS tail.

Rich flowers exact:

- root: `/dev/shm/peilincai_spcarnet_20260629_v169_rich_lowrank_k4_clean_exact/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_v169_rich_exact/wandb/offline-run-20260628_205353-el9o9n93`
- manifest: `/dev/shm/peilincai_spcarnet_20260629_v169_rich_lowrank_k4_clean_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- manifest status: `COMPLETE`, all commands return code `0`
- protocol audit: passed; `target_gt_visible_to_apply=false`, `target_gt_visible_to_selection=false`
- policy: `fallback_noop`
- selected alpha: `0.0`
- reject reason: `lpips_min_view_gain -0.000034437 < min_policy_val_lpips_min_view_gain -0.000030000`
- target changed fraction: `0.0`
- final metrics: PSNR `19.832010`, SSIM `0.505779`, LPIPS `0.405904`
- gate delta vs Phase-J: PSNR `-0.472348`, SSIM `-0.051991`, LPIPS worse by `+0.076682`

Verdict: rich-K4 is a genuine representation-level attempt, but it is not a
successful v169 method. It supports more surface bins, yet the added context
does not robustly improve SSIM/LPIPS tails and therefore cannot be promoted.

## v168 Direct-Teacher Exact-Run Status

Run root:

- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers`

Manifest/report state:

- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- manifest status: `RUNNING`.
- completed manifest steps: reparent target evidence, build teacher surface evidence, strip target evidence no-GT, verify stripped target evidence no-GT.
- pending/incomplete manifest steps: apply certified residual texture, populate eval GT from target evidence, evaluate vNext target.

No-GT verifier:

- `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/reports/flowers_ours_26000_v168_direct_teacher_lowcopy_flowers_test_target_apply_no_gt_verify.json`
- passed: `true`.
- bad view count: `0`.
- view count: `22`.
- target GT visible to apply: `false`.
- target residual visible to apply: `false`.

Active processes observed for the exact run:

- parent runner PID `2504131`: `scripts/car_model/run_vnext_certified_residual_texture_scene.py`.
- adapter PID `2519284`: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`.

Latest adapter log snapshot:

- log: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/logs/02_certified_texture.log`
- latest clean checkpoint seen in the recent log snapshot: policy candidate `25 / 48` done, and candidate `26 / 48` started.
- no final flowers test metrics were present at this snapshot.

Do not claim exact-run completion until the manifest leaves `RUNNING`, the apply/eval commands have return code `0`, and final flowers metrics are parsed against the Phase-J all-axis gate.

## Earlier v169 True-LowRank Exact Attempt

After this draft was created, a storage-aware v169 exact attempt was launched without rebuilding the 2.5G teacher cache:

- run root: `/dev/shm/peilincai_spcarnet_20260629_v169_true_lowrank_k4_reuse_exact/flowers`
- method: `ours_26000_v169_true_lowrank_k4_flowers`
- teacher evidence reused from: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target evidence reused from: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`
- W&B mode: offline, `WANDB_DIR=/dev/shm/peilincai_wandb_v169_reuse_exact`
- GPU: `2`
- no-GT verifier: passed, `22` views, bad view count `0`, target GT visible to apply `false`, target residual visible to apply `false`.

Core flags:

- `--skip_teacher_cache`
- `--strict_no_target_gt_apply`
- `--teacher_distilled_basis_mode low_rank_view_texture_k4`
- `--texture_size 16 --texture_size_candidates 16`
- `--support_expansion_mode none --support_expansion_max_extra_faces_candidates 0`
- `--surface_multiscale_prior_mode none --surface_multiscale_prior_blend_candidates 0`
- `--view_conditioned_basis_mode none`
- `--no_policy_val_prior_bin_gain_hybrid`
- `--no_policy_val_bin_uncertainty_guard`
- `--no_policy_val_bin_uncertainty_shrink`
- `--no_target_visible_energy_score`
- `--enable_policy_val_image_lpips_gate`
- `--enable_policy_val_effective_margin_gate`

Outcome: this first reuse exact attempt failed before final evaluation because
mixed-rank atlas serialization attempted to stack different coefficient shapes.
The serialization bug was fixed in the subsequent clean strict/relaxed runs
listed above.

## Implementation Fixes Added In This Round

- Added true low-rank texture fields to the adapter atlas: `teacher_texture_basis`, `teacher_texture_support`, and `teacher_texture_energy`.
- Implemented `low_rank_view_texture_k4` as view-dependent weights times per-face/per-UV texture bases, not as a direct RGB hyperparameter variant.
- Added runner/analyzer support for `--teacher_distilled_basis_mode low_rank_view_texture_k4`.
- Added read-only diagnostics:
  - `scripts/car_model/analyze_v169_teacher_signal_projection.py`
  - `scripts/car_model/analyze_v169_policy_val_upper_bound.py`
- Fixed atlas serialization for mixed low-rank faces: `teacher_basis_coefficients` is padded to the maximum rank/feature shape and `teacher_basis_output_channels` records the actual rank/channel count.
- Fixed low-rank summary rank reporting: the report now includes actual `rank`/`mean_rank` rather than unconditionally reporting `4`.
- Added `low_rank_view_texture_rich_k4`, a new v169 diagnostic representation
  that keeps rank-4 surface factorization but predicts mixture weights from
  UV, normal, camera direction, parent RGB, depth, and alpha context instead of
  only `[1, camera_xyz, normal_dot_camera]`.
- Validation: `py_compile`, `git diff --check`, policy-val rich diagnostic, and
  a clean rich flowers exact run passed at the protocol level.

## Limitations

- `/data` and `/dev/shm` are effectively full; storage is a live blocker/risk.
- Teacher signal exists, but the fit signal is much sparser than the raw teacher-parent residual: active fit coverage is only about `34%`, and raw-to-fit changed pixels are about `45.5%`.
- Same-cache projection proxy is not held-out proof and does not estimate SSIM/LPIPS.
- Policy-val mean gains are positive, but SSIM/LPIPS positive-view fractions remain only `0.5` for both base carrier and true low-rank K4 best rows.
- The clean K4 and rich-K4 exact runs prove that the current low-rank carrier
  family is not enough to beat Phase-J flowers. Richer view/context features
  increase support but worsen LPIPS tail risk.
- The immediate research bottleneck is robust perceptual transfer, not just
  fitting more bins. A next successful method likely needs a tail-aware
  perceptual objective or a different carrier/decoder, rather than another
  scalar/rank-4 residual atlas variant.
- Full9 is not authorized by the v169 prompt until flowers exact beats Phase-J on PSNR, SSIM, and LPIPS.

## Next Commands

Status and storage checks:

```bash
df -h /data /dev/shm /tmp
quota -s || true
du -sh /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact /dev/shm/peilincai_spcarnet_v169_diagnostics /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626 2>/dev/null
pgrep -af 'peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact|v168-direct-teacher-lowcopy|ecsr_apply_surface_residual_region_texture_adapter|run_vnext_certified_residual_texture_scene' || true
```

Monitor the current exact run without launching a duplicate:

```bash
tail -f /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/logs/02_certified_texture.log
python - <<'PY'
import json
from pathlib import Path
p = Path('/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json')
d = json.loads(p.read_text())
print('status', d.get('status'), 'updated_at', d.get('updated_at'))
for c in d.get('commands', []):
    print(c.get('name'), c.get('returncode'), c.get('log_path'))
PY
```

After the exact run finishes, parse final flowers metrics and compare all axes against Phase-J:

```bash
sed -n '1,220p' /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md
find /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers -maxdepth 4 -type f \( -name '*metrics*' -o -name '*eval*' -o -name '*summary*' \) -print | sort
```

If the exact run fails or falls back, keep the next step diagnostic-only: compare the true low-rank K4 policy-val winner against exact-run selected policy, then decide whether to launch a single flowers exact K4 run only after storage is cleared.
