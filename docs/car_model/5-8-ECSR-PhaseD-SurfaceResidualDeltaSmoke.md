# ECSR Phase-D Surface Residual Delta Smoke

This is the first Version-2 representation-attached residual test. It
writes a bounded residual RGB delta into per-vertex SH DC coefficients
using only Phase-A train evidence. It does not edit rendered images and
does not use held-out test views for policy acceptance.

| scene | policy status | policy-val views | faces | vertices | mean delta RGB | dPolicy L1 | test PSNR | dPSNR | dSSIM | dLPIPS |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | POLICY_ACCEPT_TEST_REGRESSION | 2,7,13,16,21,29,32,41 | 59 | 167 | 0.025000 | -0.000005 | 23.0804 | -0.2131 | -0.0235 | +0.0499 |
| flowers | REJECT_POLICY_VAL | 2,3,4,8,13,18,20,28 | 54 | 158 | 0.025000 | +0.000006 | 19.5225 | -0.1494 | -0.0209 | +0.0294 |
| treehill | POLICY_ACCEPT_TEST_REGRESSION | 3,6,7,11,20,29,30,31 | 58 | 172 | 0.025000 | -0.000001 | 20.8396 | -0.0844 | -0.0061 | +0.0394 |
| garden | POLICY_ACCEPT_TEST_REGRESSION | 14,15,20,26,30,32,39,41 | 52 | 128 | 0.024660 | -0.000001 | 24.7420 | -0.2862 | -0.0260 | +0.0467 |

Policy-val accepted: `3 / 4`
Final smoke accepted after held-out diagnostic: `0 / 4`

## Interpretation

The checkpoint-level residual attachment path is now implemented and
renderer-compatible. The current naive DC-only rule is intentionally
bounded and fixed across scenes. The smoke exposes a stronger problem:
mean train-policy L1 is too weak as the only gate, because it accepts
several tiny local deltas that still regress held-out RGB. The next
representation-level recovery needs local-mask policy metrics, strength
selection, and a least-squares or learned residual solve with smoothness
instead of a direct top-support DC offset.
