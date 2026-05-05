# Final Stage F82 Qualitative Gallery Report

Decision: `QUALITATIVE_GALLERY_READY_WITH_PER_VIEW_AUDIT`.

This report adds a fair qualitative inspection entry for F82. The gallery script aligns the strongest clean-long baseline render with the F82 fixed-policy render at identical held-out test views and selects views mechanically from per-view PSNR deltas. This avoids cherry-picking: each scene includes the worst, intermediate, and best F82-vs-clean views.

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/final_build_stageF82_qualitative_gallery.py \
  --per-scene 5
```

Output:

- `outputs/carnet/meshsplatopt/final_stageF82_qualitative_gallery/gallery.html`
- `outputs/carnet/meshsplatopt/final_stageF82_qualitative_gallery/gallery_manifest.md`
- `outputs/carnet/meshsplatopt/final_stageF82_qualitative_gallery/selected_views.json`
- `outputs/carnet/meshsplatopt/final_stageF82_qualitative_gallery/all_views.json`

## Current Per-View Audit

| scene | common views | selected views | min dPSNR | median dPSNR | max dPSNR |
|---|---:|---:|---:|---:|---:|
| bonsai | 37 | 5 | -0.0864 | +0.0911 | +0.3620 |
| courtyard | 5 | 5 | -0.1626 | +0.1064 | +0.4199 |
| room | 39 | 5 | +0.2969 | +0.7952 | +2.0845 |
| counter | 30 | 5 | +0.0690 | +0.2311 | +0.6004 |

Interpretation:

- F82 remains a scene-level all-metric clean-long win across the reported validation table.
- Room and counter are also per-view robust under this PSNR audit: the selected min-delta views are still positive.
- Bonsai and courtyard still have a small number of held-out views where PSNR is below clean, even though the scene-level metrics win. These views should be prioritized for any next visual-quality repair pass.

This qualitative audit is not a new headline claim. It is a paper-facing inspection tool and a weakness detector for visual examples.
