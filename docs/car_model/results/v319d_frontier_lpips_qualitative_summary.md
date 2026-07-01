# Support-Transport Frontier LPIPS and Qualitative Evidence

LPIPS net: `alex`, max side: `512`.
DISTS status: `computed_piq_DISTS_reduction_mean`.

## Aggregate

| method | scenes | PSNR | MAE | LPIPS | DISTS | dPSNR vs ref | dMAE vs ref | dLPIPS vs ref | dDISTS vs ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| v315d | 9 | 27.582989 | 0.028182 | 0.087739 | 0.057679 | +0.389346 | -0.000930 | -0.002469 | -0.002223 |
| v319c | 9 | 27.583642 | 0.028181 | 0.087746 | 0.057678 | +0.389999 | -0.000932 | -0.002461 | -0.002224 |
| v319d | 9 | 27.580252 | 0.028191 | 0.087746 | 0.057673 | +0.386609 | -0.000922 | -0.002461 | -0.002229 |

## Selected Panels

- `bicycle/00000.png`: ![](v319d_frontier_panels/bicycle/00000_frontier_panel.png)
- `bicycle/00005.png`: ![](v319d_frontier_panels/bicycle/00005_frontier_panel.png)
- `flowers/00010.png`: ![](v319d_frontier_panels/flowers/00010_frontier_panel.png)
- `flowers/00014.png`: ![](v319d_frontier_panels/flowers/00014_frontier_panel.png)
- `garden/00006.png`: ![](v319d_frontier_panels/garden/00006_frontier_panel.png)
- `garden/00017.png`: ![](v319d_frontier_panels/garden/00017_frontier_panel.png)
- `stump/00007.png`: ![](v319d_frontier_panels/stump/00007_frontier_panel.png)
- `stump/00005.png`: ![](v319d_frontier_panels/stump/00005_frontier_panel.png)
- `treehill/00013.png`: ![](v319d_frontier_panels/treehill/00013_frontier_panel.png)
- `treehill/00001.png`: ![](v319d_frontier_panels/treehill/00001_frontier_panel.png)
