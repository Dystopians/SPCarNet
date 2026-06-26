# vNext Full9 Effective-Margin Gate Summary

Date: 2026-06-26

This is the retained full9 validation of `ours_26000_vnext_effective_margin_gate`.
It reruns the vNext certified residual surface texture pipeline with the stricter
policy-val effective-margin gate enabled and W&B offline logging.

## Evidence Paths

- generated summary: `vnext_manifest_summary.md`
- generated JSON: `vnext_manifest_summary.json`
- runner summary: `vnext_manifest_runner_summary.md`
- artifact root: `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500`
- run root: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500`
- W&B root: `/dev/shm/peilincai_wandb_vnext_full9_margin_gate_20260626_1500`

## Protocol Result

| item | value |
|---|---:|
| completed scenes | `9 / 9` |
| failed scenes | `0 / 9` |
| missing inputs | `0 / 9` |
| protocol pass | `9 / 9` |
| accepted nonzero | `1 / 9` |
| fallback/no-op | `8 / 9` |
| mean changed fraction | `0.001371507` |
| mean PSNR | `25.067410` |
| mean SSIM | `0.741259` |
| mean LPIPS | `0.306695` |

## Scene Decisions

| scene | decision | policy | alpha | changed fraction | PSNR | SSIM | LPIPS |
|---|---|---|---:|---:|---:|---:|---:|
| bicycle | rejected | fallback_noop | `0.000000` | `0.000000000` | `23.293507` | `0.659651` | `0.332269` |
| bonsai | rejected | fallback_noop | `0.000000` | `0.000000000` | `28.864376` | `0.896010` | `0.259334` |
| counter | accepted | accepted_atlas | `0.125000` | `0.012343567` | `26.751171` | `0.862042` | `0.251955` |
| flowers | rejected | fallback_noop | `0.000000` | `0.000000000` | `19.519194` | `0.490780` | `0.424170` |
| garden | rejected | fallback_noop | `0.000000` | `0.000000000` | `24.741003` | `0.754049` | `0.248023` |
| kitchen | rejected | fallback_noop | `0.000000` | `0.000000000` | `27.816387` | `0.876443` | `0.199201` |
| room | rejected | fallback_noop | `0.000000` | `0.000000000` | `28.739004` | `0.884790` | `0.249916` |
| stump | rejected | fallback_noop | `0.000000` | `0.000000000` | `25.043329` | `0.689480` | `0.349850` |
| treehill | rejected | fallback_noop | `0.000000` | `0.000000000` | `20.838715` | `0.558089` | `0.445541` |

## Comparison to Current Baselines

| method | PSNR | SSIM | LPIPS | interpretation |
|---|---:|---:|---:|---|
| clean MeshSplatting | `25.151682` | `0.749018` | `0.287621` | local same-protocol clean baseline |
| v106 POD-MoE base-preserve | `25.831280` | `0.760830` | `0.268435` | current verified representation-quality line |
| vNext fixed-policy cleanup | `25.067699` | `0.741260` | `0.306689` | previous vNext full9 protocol closure |
| vNext effective-margin gate | `25.067410` | `0.741259` | `0.306695` | stricter safety gate; mostly fallback/no-op |

Delta of effective-margin gate vs clean MeshSplatting:

| metric | delta |
|---|---:|
| PSNR | `-0.084272` |
| SSIM | `-0.007759` |
| LPIPS | `+0.019074` |

## Interpretation

This run is a reliability milestone, not a quality milestone. The stricter gate
successfully prevents low-effect or weak lower-tail residual texture candidates
from modifying target renders. It also exposes the current vNext bottleneck:
under a fixed, no-target-GT protocol, the residual surface texture has meaningful
target impact on only `counter`, while the other eight scenes collapse to exact
parent/no-op fallback.

The method therefore should not be promoted as a paper endpoint. Phase-J remains
the teacher/upper-bound endpoint for the strong RGB story, and v106 remains the
stronger representation-quality line. The next research step must improve the
representation itself, not only tighten the selector.

## Required Next Method Work

1. Move from low-capacity residual texture selection to a stronger train-only
   teacher-distilled surface representation with enough target impact.
2. Preserve the same no-target-GT certificate: policy-val effect margins,
   lower-tail checks, native-frame contract checks, and exact no-op fallback.
3. Reuse this full9 gate summary as the promotion threshold: any new method must
   beat clean MeshSplatting and v106 on the same selected full9 protocol, not
   merely pass safety fallback.
