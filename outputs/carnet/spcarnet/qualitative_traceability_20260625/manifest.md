# SPCarNet Qualitative Traceability Manifest

This manifest ties the current presentation panels to scene/view IDs, metric deltas, crop coordinates, and source image paths. It is intended for mentor/PPT provenance and reviewer-facing appendix preparation.

## Summary

- panels: `3`
- rows: `16`
- figures_existing: `3`
- all_source_images_exist: `True`

## Panels

| panel | purpose | figure | examples |
|---|---|---|---:|
| `phasej_where_it_helps` | Phase-J local error-reduction showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | 6 |
| `compact_ela_outdoor_detail` | Compact-ELA outdoor detail showcase | `assets/spcarnet_m360_outdoor_detail_showcase.png` | 5 |
| `compact_ela_fullframe_gallery` | Compact-ELA full-frame gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` | 5 |

## Examples

| panel | rank | scene | view | dPSNR | dSSIM | dLPIPS | local dPSNR | local MAE drop | source images ok |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `phasej_where_it_helps` | 1 | bonsai | 00001.png | 6.634024 | 0.045201 | -0.087806 | 11.786805 | 78.603240 | `True` |
| `phasej_where_it_helps` | 2 | kitchen | 00011.png | 3.433716 | 0.025010 | -0.057845 | 10.477992 | 71.403360 | `True` |
| `phasej_where_it_helps` | 3 | room | 00011.png | 3.496552 | 0.021969 | -0.065559 | 10.358264 | 67.681553 | `True` |
| `phasej_where_it_helps` | 4 | counter | 00013.png | 2.165871 | 0.040742 | -0.066509 | 6.024988 | 54.877190 | `True` |
| `phasej_where_it_helps` | 5 | garden | 00006.png | 1.737812 | 0.047882 | -0.067825 | 4.260363 | 44.359905 | `True` |
| `phasej_where_it_helps` | 6 | flowers | 00014.png | 1.118814 | 0.075413 | -0.102840 | 2.154481 | 25.346069 | `True` |
| `compact_ela_outdoor_detail` | 1 | flowers | 00014.png | 0.989775 | 0.061620 | -0.068225 | 2.053088 | 24.158761 | `True` |
| `compact_ela_outdoor_detail` | 2 | garden | 00008.png | 1.273155 | 0.043212 | -0.055140 | 2.699023 | 27.640206 | `True` |
| `compact_ela_outdoor_detail` | 3 | treehill | 00010.png | 0.593367 | 0.049067 | -0.088149 | 3.026837 | 31.986421 | `True` |
| `compact_ela_outdoor_detail` | 4 | bicycle | 00021.png | 1.127735 | 0.038535 | -0.061511 | 1.878877 | 17.540336 | `True` |
| `compact_ela_outdoor_detail` | 5 | stump | 00007.png | 0.255608 | 0.012164 | -0.020815 | 0.808748 | 12.780879 | `True` |
| `compact_ela_fullframe_gallery` | 1 | garden | 00019.png | 1.581549 | 0.047953 | -0.061766 |  |  | `True` |
| `compact_ela_fullframe_gallery` | 2 | flowers | 00014.png | 0.989775 | 0.061620 | -0.068225 |  |  | `True` |
| `compact_ela_fullframe_gallery` | 3 | treehill | 00010.png | 0.593367 | 0.049067 | -0.088149 |  |  | `True` |
| `compact_ela_fullframe_gallery` | 4 | bicycle | 00019.png | 1.009087 | 0.045258 | -0.065687 |  |  | `True` |
| `compact_ela_fullframe_gallery` | 5 | bonsai | 00001.png | 2.785767 | 0.006315 | -0.000703 |  |  | `True` |

## Scope Note

- The manifest records provenance for existing presentation figures; it does not regenerate images.
- `phasej_where_it_helps` is the recommended main qualitative figure for the current Phase-J headline.
- `compact_ela_outdoor_detail` and `compact_ela_fullframe_gallery` are supporting qualitative evidence for official-style Compact-ELA.
