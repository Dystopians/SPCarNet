# v329b Perceptual/Geometry Follow-up and v330 Local-Support Probe

Date: 2026-07-01

## Purpose

This follow-up closes two evidence gaps left by the v329b fixed rollback
certificate log:

- fresh clean-frontier perceptual/qualitative metrics for `clean26000`,
  `v322c`, `v327b`, and `v329b`;
- geometry/triangle accounting for the compact parent used by the current
  support-transport frontier.

It also records the first v330 local-support probe as a negative result. That
probe is important because it shows that the existing local-support policy does
not solve the remaining treehill/stump opportunity gap.

## Frontier / Perceptual Evaluation

Command:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/build_support_transport_frontier_comparison.py \
  --method clean26000=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method v322c=outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701 \
  --method v327b=outputs/carnet/spcarnet_v327b_selected_full9_20260701 \
  --method v329b=outputs/carnet/spcarnet_v329b_selected_full9_20260701 \
  --output_dir outputs/carnet/spcarnet_v329b_frontier_comparison_full9_20260701 \
  --scenes bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill \
  --panel_scenes bonsai,room,garden,treehill \
  --max_panels_per_scene 2 \
  --lpips_max_side 512 \
  --panel_max_side 640 \
  --crop_size 256 \
  --device cuda \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v329b_frontier_lpips_dists_qualitative_full9
```

W&B offline run:

```text
outputs/carnet/spcarnet_v329b_frontier_comparison_full9_20260701/wandb/offline-run-20260701_025348-z1y9qt4d
```

Committed evidence:

```text
docs/car_model/results/v329b_frontier_lpips_qualitative_summary.json
docs/car_model/results/v329b_frontier_lpips_qualitative_summary.md
docs/car_model/results/v329b_frontier_panels/bonsai_00035_frontier_panel.png
docs/car_model/results/v329b_frontier_panels/room_00009_frontier_panel.png
docs/car_model/results/v329b_frontier_panels/garden_00017_frontier_panel.png
docs/car_model/results/v329b_frontier_panels/treehill_00011_frontier_panel.png
```

Aggregate metrics:

| method | scenes | PSNR | MAE | LPIPS | DISTS | dPSNR vs clean | dMAE vs clean | dLPIPS vs clean | dDISTS vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| v322c | 9 | 27.587073 | 0.028173 | 0.087735 | 0.057659 | +0.393430 | -0.000940 | -0.002472 | -0.002243 |
| v327b | 9 | 27.587183 | 0.028174 | 0.087733 | 0.057660 | +0.393540 | -0.000938 | -0.002475 | -0.002242 |
| v329b | 9 | 27.588444 | 0.028173 | 0.087733 | 0.057664 | +0.394801 | -0.000939 | -0.002474 | -0.002238 |

Interpretation:

- v329b is still clearly above the local clean MeshSplatting frontier on PSNR,
  MAE, LPIPS, and DISTS.
- v329b is slightly better than v322c/v327b on PSNR.
- v329b is not all-axis dominant over v327b: LPIPS is slightly worse than v327b
  and DISTS is slightly worse than both v322c and v327b.
- Therefore this follow-up improves evidence completeness, but it does not
  prove final paper-level closure.

## Geometry Accounting

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/build_support_transport_geometry_accounting.py \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --compact_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --output_json docs/car_model/results/v329b_frontier_geometry_accounting_summary.json \
  --output_md docs/car_model/7-01-v329b-Frontier-Geometry-Accounting.md \
  --scenes bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill
```

Aggregate geometry:

| scenes | clean triangles | support-transport triangles | total triangle reduction | clean vertices | support-transport vertices | total vertex reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 91019714 | 84219015 | 7.471677% | 28914623 | 27795247 | 3.871315% |

All per-scene topology error counts are `0`. This confirms that v329b preserves
the compact-parent geometry advantage, but the geometry reduction comes from
the inherited compact parent rather than from the new fixed rollback
certificate itself.

## v330 Local-Support Probe

The first v330 probe tested whether the existing local-support policy can
unlock the remaining treehill/stump candidate opportunities after v329b.

Runs:

```text
outputs/carnet/spcarnet_v330a_localsupport_arbitration_treehill_20260701
outputs/carnet/spcarnet_v330a_localsupport_arbitration_stump_20260701
outputs/carnet/spcarnet_v330b_localsupport_enabled_treehill_20260701
outputs/carnet/spcarnet_v330b_localsupport_enabled_stump_20260701
```

W&B offline runs:

```text
outputs/carnet/spcarnet_v330a_localsupport_arbitration_treehill_20260701/wandb/offline-run-20260701_025717-r6fyrxxf
outputs/carnet/spcarnet_v330a_localsupport_arbitration_stump_20260701/wandb/offline-run-20260701_025715-55sfmvqo
outputs/carnet/spcarnet_v330b_localsupport_enabled_treehill_20260701/wandb/offline-run-20260701_025847-fw4r8dlo
outputs/carnet/spcarnet_v330b_localsupport_enabled_stump_20260701/wandb/offline-run-20260701_025844-xjbtakef
```

Result:

| probe | scene | local-support enabled in final report | selected output changed vs v329b | selected PSNR gain | selected SSIM gain | verdict |
|---|---|---|---|---:|---:|---|
| v330a | treehill | false | no | 0.104664074413 | 0.001673645443 | accepted too many source views |
| v330a | stump | false | no | 0.057029761393 | 0.001208242029 | accepted too many source views |
| v330b | treehill | false | no | 0.104664074413 | 0.001673645443 | did not improve source SSIM enough |
| v330b | stump | false | no | 0.057029761393 | 0.001208242029 | did not clear source min delta |

Interpretation:

- v330a/v330b are negative probes, not promoted methods.
- The existing local-support policy does not change the selected outputs on
  treehill or stump, even after relaxing the source accept fraction.
- This supports the next design conclusion: the remaining gap likely needs a
  stronger per-view candidate arbitration/promotion certificate, not another
  scalar relaxation of the current local-support gate.

## Verdict

Final status: NOT COMPLETE.

Evidence completeness improved: v329b now has clean-frontier PSNR/MAE/LPIPS/DISTS
and compact-geometry accounting. But the method is still not a paper-final
endpoint because perceptual dominance over v327b is mixed and the v330
local-support probe failed to unlock the remaining high-opportunity views.
