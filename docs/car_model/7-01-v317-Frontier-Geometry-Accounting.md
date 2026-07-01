# Support-Transport Geometry Accounting

This ledger compares the local clean MeshSplatting topology against the compact parent topology used by the current support-transport frontier.

Important protocol note: v305, v315d, and v316c inherit the same compact parent topology. Their support-transport policy changes render/color corrections, not the mesh triangle or vertex count.

## Aggregate

| scenes | clean triangles | support-transport triangles | total triangle reduction | clean vertices | support-transport vertices | total vertex reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 91019714 | 84219015 | 7.471677% | 28914623 | 27795247 | 3.871315% |

## Per Scene

| scene | clean triangles | support-transport triangles | triangle reduction | clean vertices | support-transport vertices | vertex reduction | compact parent removed fraction | topology errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 9422930 | 8309749 | 11.813534% | 3490855 | 3318902 | 4.925813% | 2.000003% | 0 |
| bonsai | 10834182 | 9555533 | 11.801989% | 3409579 | 3295557 | 3.344167% | 2.000001% | 0 |
| counter | 9850919 | 9644247 | 2.097997% | 2537250 | 2478825 | 2.302690% | 1.999996% | 0 |
| flowers | 9649601 | 8509358 | 11.816478% | 3605171 | 3414899 | 5.277752% | 1.999996% | 0 |
| garden | 11568056 | 11166587 | 3.470497% | 3414016 | 3315236 | 2.893367% | 2.000004% | 0 |
| kitchen | 9716239 | 9512393 | 2.097993% | 2451717 | 2391146 | 2.470554% | 1.999995% | 0 |
| room | 11173063 | 10938652 | 2.098001% | 2840131 | 2777389 | 2.209123% | 2.000002% | 0 |
| stump | 9277087 | 8180134 | 11.824326% | 3558228 | 3383973 | 4.897241% | 2.000006% | 0 |
| treehill | 9527637 | 8402362 | 11.810641% | 3607676 | 3419320 | 5.220979% | 2.000003% | 0 |

## Interpretation

- The current frontier is not only a render-quality tweak: it keeps the compact-parent geometry advantage while improving render metrics over the local clean baseline.
- Geometry reduction is scene-dependent because the compact parent is conservative on indoor scenes and more aggressive on sparse/outdoor-heavy scenes.
- This does not prove final paper closure by itself; it closes the geometry-accounting gap for the current evidence package.
