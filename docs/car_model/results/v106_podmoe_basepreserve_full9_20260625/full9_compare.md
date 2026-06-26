# v106 Candidate Report Comparison

- inputs: `9`
- rows: `18`
- deltas: candidate minus reference; lower LPIPS is better.

## Candidate vs v104c

| scene | method | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c | field variant | POD base | expert certificate |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| bicycle | v104c_shrink_view_affine_min1_minviews1 | 23.717649 | 0.674972 | 0.313503 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| bonsai | v104c_shrink_view_affine_min1_minviews1 | 30.310877 | 0.907367 | 0.230186 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| counter | v104c_shrink_view_affine_min1_minviews1 | 27.498068 | 0.867420 | 0.238986 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| flowers | v104c_shrink_view_affine_min1_minviews1 | 20.075844 | 0.531076 | 0.374473 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| garden | v104c_shrink_view_affine_min1_minviews1 | 25.788094 | 0.799263 | 0.174584 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| kitchen | v104c_shrink_view_affine_min1_minviews1 | 28.770449 | 0.881590 | 0.188021 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| room | v104c_shrink_view_affine_min1_minviews1 | 29.597836 | 0.891837 | 0.230664 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| stump | v104c_shrink_view_affine_min1_minviews1 | 25.459311 | 0.714599 | 0.282213 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| treehill | v104c_shrink_view_affine_min1_minviews1 | 21.243763 | 0.578418 | 0.384298 | +0.000000 | +0.000000 | +0.000000 |  |  |  |
| counter | v106_podmoe_basepreserve | 27.499645 | 0.867521 | 0.238847 | **+0.001577** | **+0.000102** | **-0.000139** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| kitchen | v106_podmoe_basepreserve | 28.772043 | 0.881652 | 0.187815 | **+0.001595** | **+0.000062** | **-0.000206** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| bonsai | v106_podmoe_basepreserve | 30.316090 | 0.907520 | 0.230050 | **+0.005213** | **+0.000154** | **-0.000136** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| bicycle | v106_podmoe_basepreserve | 23.719175 | 0.675086 | 0.313405 | **+0.001526** | **+0.000115** | **-0.000098** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| flowers | v106_podmoe_basepreserve | 20.077723 | 0.531240 | 0.374393 | **+0.001879** | **+0.000163** | **-0.000080** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| garden | v106_podmoe_basepreserve | 25.790945 | 0.799382 | 0.174480 | **+0.002851** | **+0.000119** | **-0.000104** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| room | v106_podmoe_basepreserve | 29.600351 | 0.891889 | 0.230616 | **+0.002516** | **+0.000051** | **-0.000048** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| stump | v106_podmoe_basepreserve | 25.460457 | 0.714661 | 0.282135 | **+0.001146** | **+0.000061** | **-0.000078** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |
| treehill | v106_podmoe_basepreserve | 21.245092 | 0.578518 | 0.384177 | **+0.001329** | **+0.000099** | **-0.000121** | pod_moe | base_preserving_boundary | weighted_normal_equation_lambda_star |

## Mean Gap by Method

| method | rows | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|---:|
| v104c_shrink_view_affine_min1_minviews1 | 9 | +0.677417 | +0.011709 | -0.019073 | +0.000000 | +0.000000 | +0.000000 |
| v106_podmoe_basepreserve | 9 | +0.679598 | +0.011812 | -0.019185 | **+0.002181** | **+0.000103** | **-0.000112** |

## Key POD Stats

| scene | method | valid triangles | mixture triangles | fallback only | detail triangles | boundary triangles | gate mean | debt guard | mean abs delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | v104c_shrink_view_affine_min1_minviews1 | 2088547.000000 |  |  |  |  |  |  | 0.009774 |
| bonsai | v104c_shrink_view_affine_min1_minviews1 | 3405888.000000 |  |  |  |  |  |  | 0.010236 |
| counter | v104c_shrink_view_affine_min1_minviews1 | 2716449.000000 |  |  |  |  |  |  | 0.008246 |
| flowers | v104c_shrink_view_affine_min1_minviews1 | 1853840.000000 |  |  |  |  |  |  | 0.014429 |
| garden | v104c_shrink_view_affine_min1_minviews1 | 3311331.000000 |  |  |  |  |  |  | 0.011178 |
| kitchen | v104c_shrink_view_affine_min1_minviews1 | 3076129.000000 |  |  |  |  |  |  | 0.009386 |
| room | v104c_shrink_view_affine_min1_minviews1 | 3200462.000000 |  |  |  |  |  |  | 0.006765 |
| stump | v104c_shrink_view_affine_min1_minviews1 | 2037135.000000 |  |  |  |  |  |  | 0.005460 |
| treehill | v104c_shrink_view_affine_min1_minviews1 | 1839078.000000 |  |  |  |  |  |  | 0.010198 |
| counter | v106_podmoe_basepreserve | 2716466.000000 |  |  | 312907.000000 | 1365871.000000 |  |  | 0.008257 |
| kitchen | v106_podmoe_basepreserve | 3076122.000000 |  |  | 267510.000000 | 1584254.000000 |  |  | 0.009398 |
| bonsai | v106_podmoe_basepreserve | 3405868.000000 |  |  | 393300.000000 | 1801550.000000 |  |  | 0.010259 |
| bicycle | v106_podmoe_basepreserve | 2088538.000000 |  |  | 197321.000000 | 1049943.000000 |  |  | 0.009785 |
| flowers | v106_podmoe_basepreserve | 1853826.000000 |  |  | 227992.000000 | 954099.000000 |  |  | 0.014452 |
| garden | v106_podmoe_basepreserve | 3311331.000000 |  |  | 142440.000000 | 1606718.000000 |  |  | 0.011201 |
| room | v106_podmoe_basepreserve | 3200713.000000 |  |  | 378450.000000 | 1646706.000000 |  |  | 0.006774 |
| stump | v106_podmoe_basepreserve | 2037233.000000 |  |  | 159936.000000 | 921436.000000 |  |  | 0.005471 |
| treehill | v106_podmoe_basepreserve | 1839082.000000 |  |  | 175623.000000 | 862288.000000 |  |  | 0.010227 |
