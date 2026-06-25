# v104a Hard-Triad Summary

Date: 2026-06-25

Status: complete existing hard-triad artifacts found for `counter`, `kitchen`, and `bonsai`. No new GPU jobs were run.

The prior persisted v104a summary under `outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_counter_20260625/` was counter-only. The complete hard-triad artifacts used here already existed under `/dev/shm`.

## Mean Metrics

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 |
| v103 affine min_count=1 | 28.384418 | 0.879855 | 0.226611 |
| v104a view-affine min_count=1 | 28.823045 | 0.884927 | 0.219492 |
| v101/v102a endpoint ceiling | 30.167395 | 0.913355 | 0.163709 |

## Mean Deltas

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104a minus clean | +1.001192 | +0.006625 | -0.017402 |
| v104a minus v103 | +0.438627 | +0.005072 | -0.007120 |
| v104a minus v101/v102a | -1.344350 | -0.028428 | +0.055783 |

## Per-Scene Metrics

| scene | v104a PSNR | v104a SSIM | v104a LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean |
|---|---:|---:|---:|---:|---:|---:|
| counter | 27.492378 | 0.867344 | 0.239003 | +0.740604 | +0.005288 | -0.013000 |
| kitchen | 28.765291 | 0.881528 | 0.188096 | +0.946739 | +0.005076 | -0.011089 |
| bonsai | 30.211466 | 0.905910 | 0.231375 | +1.316233 | +0.009510 | -0.028118 |

| scene | dPSNR vs v103 | dSSIM vs v103 | dLPIPS vs v103 | dPSNR vs v101/v102a | dSSIM vs v101/v102a | dLPIPS vs v101/v102a |
|---|---:|---:|---:|---:|---:|---:|
| counter | +0.284178 | +0.003939 | -0.004173 | -0.950529 | -0.026352 | +0.052446 |
| kitchen | +0.455139 | +0.003975 | -0.006421 | -1.432104 | -0.034565 | +0.056093 |
| bonsai | +0.576565 | +0.007303 | -0.010765 | -1.650417 | -0.024366 | +0.058809 |

## Artifact Check

| scene | field `.pt` | manifest | render report | results entry | renders/gt | complete |
|---|---|---|---|---|---:|---|
| counter | present, 432.289 MiB | present | present | present | 30/30 | yes |
| kitchen | present, 426.379 MiB | present | present | present | 35/35 | yes |
| bonsai | present, 428.313 MiB | present | present | present | 37/37 | yes |

No required v104a aggregation artifact was missing. The requested summary directory itself was absent before this run and was created for these two files.

## Field And Render Stats

| scene | valid triangles | all triangles | valid fraction | accumulated pixels | build sec | render sec | mean surface valid | mean abs delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 2716436 | 9644247 | 0.281664 | 48475697 | 447.035 | 50.493 | 0.999042 | 0.008267 |
| kitchen | 3076099 | 9512393 | 0.323378 | 56628652 | 829.495 | 52.002 | 0.999505 | 0.009361 |
| bonsai | 3405889 | 9555533 | 0.356431 | 59912951 | 633.217 | 62.412 | 0.999617 | 0.009795 |

All manifests reported `solve_failures: 0`.

## Verdict

v104a hard-triad is complete and improves over both clean MeshSplatting and v103 on PSNR, SSIM, and LPIPS for every scene and on the hard-triad mean. It still remains below the v101/v102a endpoint ceiling on all three mean metrics.

## Commands And Results

```bash
ls -la outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_hardtriad_20260625
```

Result before writing:

```text
ls: cannot access 'outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_hardtriad_20260625': No such file or directory
```

```bash
find outputs/carnet/meshsplatopt /dev/shm -maxdepth 5 \( -path '*v104*' -o -path '*view_affine*' \) 2>/dev/null | sort | head -n 300
```

Key result:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/bonsai/detached_model/test/ours_26000_v104a_view_affine_min1_bonsai
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/test/ours_26000_v104a_view_affine_min1_counter
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/kitchen/detached_model/test/ours_26000_v104a_view_affine_min1_kitchen
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/bonsai/v104_view_affine_min1_field.manifest.json
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/bonsai/v104_view_affine_min1_field.pt
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.manifest.json
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.pt
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/kitchen/v104_view_affine_min1_field.manifest.json
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/kitchen/v104_view_affine_min1_field.pt
outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_counter_20260625/v104a_counter_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v104_view_affine_field_counter_20260625/v104a_counter_summary.md
```

```bash
python - <<'PY'
import json
for scene in ['counter','kitchen','bonsai']:
    path=f'/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/{scene}/detached_model/results.json'
    data=json.load(open(path))
    key=f'ours_26000_v104a_view_affine_min1_{scene}'
    print(scene, key, data[key])
PY
```

Result:

```text
counter ours_26000_v104a_view_affine_min1_counter {'LPIPS': 0.23900321125984192, 'PSNR': 27.49237823486328, 'SSIM': 0.8673436641693115}
kitchen ours_26000_v104a_view_affine_min1_kitchen {'LPIPS': 0.18809644877910614, 'PSNR': 28.765291213989258, 'SSIM': 0.8815281391143799}
bonsai ours_26000_v104a_view_affine_min1_bonsai {'SSIM': 0.9059099555015564, 'PSNR': 30.21146583557129, 'LPIPS': 0.23137512803077698}
```

```bash
python - <<'PY'
import json
from pathlib import Path
scenes=['counter','kitchen','bonsai']
for scene in scenes:
    report=Path(f'/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/{scene}/detached_model/test/ours_26000_v104a_view_affine_min1_{scene}/render_py_endpoint_report.json')
    frames=json.load(open(report))['frames']
    print(scene, len(frames))
PY
```

Result:

```text
counter 30
kitchen 35
bonsai 37
```

```bash
df -h /data /dev/shm
```

Result:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1p1   28T   27T  430M 100% /data
tmpfs           252G  248G  4.5G  99% /dev/shm
```

No evaluation command was launched. No source files were modified.
